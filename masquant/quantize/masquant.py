# coding=utf-8 
# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_llama_layer_v2 import QuantLlamaDecoderLayerV2
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from models.int_minicpm_layer import QuantMiniCPMDecoderLayerV2
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
import numpy as np

from quantize.utils import (let_parameters, lwc_parameters, get_mas_parameters,
                            mas_state_dict, register_scales_and_zeros, smooth_and_quant_temporary, truncate_scale,
                            smooth_and_quant_inplace, clear_temp_variable, set_quant_state, reuse_scale)
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")
import transformers


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     


def compute_sqnr_simple(original, quantized):
    original = original.float()
    quantized = quantized.float()
    error = original - quantized
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(error ** 2)
    sqnr = 10 * torch.log10(signal_power / noise_power)
    
    return sqnr.item()

def compute_sqnr_simple_multimodal(original, quantized, multi_modal_mask=None):
    (audio_mask, image_mask, text_mask) = multi_modal_mask
    
    original_text = original.float() * text_mask
    original_audio = original.float() * audio_mask
    original_vision = original.float() * image_mask

    quantized_text = quantized.float() * text_mask
    quantized_audio = quantized.float() * audio_mask
    quantized_vision = quantized.float() * image_mask

    quant_error_text = original_text -  quantized_text
    quant_error_audio = original_audio -  quantized_audio
    quant_error_vision = original_vision -  quantized_vision
    
    signal_power_text = torch.mean(original_text ** 2)
    signal_power_audio = torch.mean(original_audio ** 2)
    signal_power_vision = torch.mean(original_vision ** 2)
        
    noise_power_text = torch.mean(quant_error_text ** 2)
    noise_power_audio = torch.mean(quant_error_audio ** 2)
    noise_power_vision = torch.mean(quant_error_vision ** 2)
    
    sqnr_text = 10 * torch.log10(signal_power_text / noise_power_text)
    sqnr_audio = 10 * torch.log10(signal_power_audio / noise_power_audio)
    sqnr_vision = 10 * torch.log10(signal_power_vision / noise_power_vision)

    sqnr = sqnr_text + sqnr_audio + sqnr_vision * 0.5
    
    return sqnr.item()

def check_gradient(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
                return False
            if torch.isinf(param.grad).any():
                print(f"Inf gradient in {name}")
                return False
    return True


def _parse_protected_layers(spec, num_layers):
    protected = set()
    for item in str(spec).split(','):
        item = item.strip().lower()
        if not item:
            continue
        if item == 'first':
            protected.add(0)
        elif item == 'last':
            protected.add(num_layers - 1)
        elif item.isdigit():
            protected.add(int(item))
    return protected


def apply_mixed_precision_policy(qlayer, args, layer_idx, num_layers):
    if not getattr(args, 'mixed_precision', False):
        return
    low_bits = args.mp_low_bits if args.mp_low_bits is not None else args.wbits
    high_bits = args.mp_high_bits
    protected_layers = _parse_protected_layers(args.mp_protect_layers, num_layers)
    protected_modules = {m.strip() for m in args.mp_protect_modules.split(',') if m.strip()}
    for name, module in qlayer.named_modules():
        if isinstance(module, QuantLinear):
            is_protected_module = any(key in name for key in protected_modules)
            bits = high_bits if (layer_idx in protected_layers or is_protected_module) else low_bits
            module.weight_quantizer.change_n_bits(bits)


def _masked_mean(value, mask):
    denom = mask.sum().clamp(min=1)
    return (value * mask).sum() / denom


def sensitivity_aware_modal_loss(
    quant_out,
    fp_out,
    multi_modal_mask,
    tau=0.05,
    auto_weight=False,
    grad_info=None,
    layer_idx=0,
):
    audio_mask, image_mask, text_mask = multi_modal_mask
    diff = (quant_out - fp_out).abs()
    modal_losses = []
    modal_names = []
    text_loss = _masked_mean(diff, text_mask)
    modal_losses.append(text_loss)
    modal_names.append('text')
    if image_mask is not None and image_mask.any():
        modal_losses.append(_masked_mean(diff, image_mask))
        modal_names.append('vision')
    if audio_mask is not None and audio_mask.any():
        modal_losses.append(_masked_mean(diff, audio_mask))
        modal_names.append('audio')

    losses = torch.stack(modal_losses)
    if auto_weight:
        weights = torch.softmax(losses.detach() / max(tau, 1e-6), dim=0) * len(modal_losses)
    else:
        weights = torch.ones_like(losses)
        for idx, name in enumerate(modal_names):
            if name == 'vision':
                weights[idx] = 0.5
                if grad_info is not None:
                    weights[idx] = 0.126
            elif name == 'audio':
                weights[idx] = 1.0
                if grad_info is not None:
                    weights[idx] = (
                        grad_info[f'layers.{layer_idx}']['aud_avg_grad']
                        / grad_info[f'layers.{layer_idx}']['cap_avg_grad']
                    )
    return (weights.to(losses.device) * losses).mean()

def masquant(
    lm,
    args,
    dataloader,
    act_scales,
    logger=None,
    grad_info = None
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device

    if 'Qwen2.5-Omni' in args.model:
        use_cache = model.config.text_config.use_cache
        model.config.text_config.use_cache = False
    elif "internvl" in args.model.lower():
        use_cache = getattr(model.language_model.config, "use_cache", False)
        model.language_model.config.use_cache = False
    elif 'MiniCPM' in args.model or 'llama' in args.model or "onevision" in args.model.lower():
        use_cache = getattr(model.config, "use_cache", False)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    is_llama = False
    if "minicpm" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # MiniCPM 不需要移动 rotary_emb，因为它在 QuantMiniCPMAttentionV2 中创建
        
        print(f"当前使用 MiniCPM 模型，使用 QuantMiniCPMDecoderLayerV2")
        DecoderLayer = QuantMiniCPMDecoderLayerV2
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if transformers.__version__ == "4.31.0":
            print("当前 transformers 版本是 4.31.0, 使用旧的构造函数")
            DecoderLayer = QuantLlamaDecoderLayer
        else:
            print(f"当前 transformers 版本是 {transformers.__version__}， 使用新的构造函数")     
            DecoderLayer = QuantLlamaDecoderLayerV2   
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    elif 'Qwen2.5-Omni' in args.model:
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        model.visual = model.visual.to(dev)
        model.visual.rotary_pos_emb = model.visual.rotary_pos_emb.to(dev)
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
        model.audio_tower = model.audio_tower.to(dev)
        
        for layer in model.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(dev)

        from models.int_qwen_omni_layer import QuantQwenDecoderLayerV2

        DecoderLayer = QuantQwenDecoderLayerV2
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
            # "down_proj": "fc2"
        }
        layer_name_prefix = "model.layers"
    elif 'Qwen2.5-VL' in args.model:
        is_llama = True
        layers = model.language_model.layers
        model.language_model.embed_tokens = model.language_model.embed_tokens.to(dev)
        model.language_model.norm = model.language_model.norm.to(dev)
        model.visual = model.visual.to(dev)
        model.visual.rotary_pos_emb = model.visual.rotary_pos_emb.to(dev)
        model.language_model.rotary_emb = model.language_model.rotary_emb.to(dev)
        
        for layer in model.language_model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(dev)

        from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2

        DecoderLayer = QuantQwenDecoderLayerV2
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
            # "down_proj": "fc2"
        }
        layer_name_prefix = "model.language_model.layers"
    elif "internvl" in args.model.lower():
        is_llama = True
        # InternVLChatModel -> language_model (InternLM2) -> model.layers
        # Do not shadow outer `lm` (LMClass); keep tokenizer / device access.
        lang_model = model.language_model
        layers = lang_model.model.layers
        if hasattr(lang_model.model, "tok_embeddings"):
            lang_model.model.tok_embeddings = lang_model.model.tok_embeddings.to(dev)
        else:
            lang_model.model.embed_tokens = lang_model.model.embed_tokens.to(dev)
        lang_model.model.norm = lang_model.model.norm.to(dev)
        model.vision_model = model.vision_model.to(dev)
        model.mlp1 = model.mlp1.to(dev)
        from models.int_internvl_layer import QuantInternVLDecoderLayerV2
        DecoderLayer = QuantInternVLDecoderLayerV2
        pairs = {
            "wqkv": "qkv",
            "wo": "out",
            "w1": "fc1",
        }
        layer_name_prefix = "language_model.model.layers"
        print("当前使用 InternVL2，使用 QuantInternVLDecoderLayerV2")
    elif ("llava-onevision" in args.model.lower()) or ("llava_onevision" in args.model.lower()) or (
        "onevision" in args.model.lower() and "llava" in args.model.lower()
    ):
        is_llama = True
        # LlavaQwenForCausalLM -> model.model.layers
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
        # Vision tower / projector may stay on CPU under some device_map loads.
        if hasattr(model, "get_vision_tower"):
            vt = model.get_vision_tower()
            if vt is not None:
                vt.to(dev)
        for attr in ("mm_projector", "vision_resampler"):
            if hasattr(model.model, attr) and getattr(model.model, attr) is not None:
                setattr(model.model, attr, getattr(model.model, attr).to(dev))
        from models.int_minicpm_layer import QuantMiniCPMDecoderLayerV2
        DecoderLayer = QuantMiniCPMDecoderLayerV2
        pairs = {
            "q_proj": "qkv",
            "o_proj": "out",
            "up_proj": "fc1",
        }
        layer_name_prefix = "model.layers"
        print("当前使用 LLaVA-OneVision，使用 QuantMiniCPMDecoderLayerV2 (Qwen2)")
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/InternVL/LLaVA-OneVision now")
    
    layers[0] = layers[0].to(dev)
    # 关于数据类型的说明：模型是用 float16 来 load 的，那么权重和激活也就是 float16 的，但是scale 是 float64，所以权重乘以 scale 的时候，要转换成 float64 再乘，结果再转换成 float16，激活也是同样的逻辑:
    # ws = (w.to(torch.float64) * S).to(torch.float16) ; xs = (x.to(torch.float64) / S).to(torch.float16) ; y = ws * xs + bias
    if args.deactive_amp and args.epochs>0:
        dtype = torch.bfloat16
        traincast = nullcontext
    else:
        dtype = torch.bfloat16
        traincast = torch.cuda.amp.autocast
    inps = []
    cache = {"i": 0}
    position_ids_cache = []
    attention_mask_cache = []
    position_embeddings_cache = []
    multi_modal_mask_cache = []
        
    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            # For MiniCPM, manually create multi_modal_mask if not present
            if 'multi_modal_mask' not in kwargs and 'MiniCPM' in args.model:
                # MiniCPM uses <image> tokens (id might vary)
                # We need to create the mask based on input shape
                # Since MiniCPM processes vision through resampler before LLM,
                # we assume all vision tokens come first, then text tokens
                # For now, create a dummy mask that treats everything as text
                # This will be replaced with proper mask generation
                batch_size, seq_len, hidden_dim = inp.shape if inp.dim() == 3 else (1, inp.shape[0], inp.shape[1])
                device = inp.device
                
                # Create masks: for MiniCPM without explicit image token ids,
                # we set all tokens as text (conservative approach)
                # TODO: Improve this by detecting actual image token positions
                image_mask = torch.zeros((batch_size, seq_len, hidden_dim), dtype=torch.bool, device=device)
                all_true = torch.ones((batch_size, seq_len, hidden_dim), dtype=torch.bool, device=device)
                text_mask = all_true & ~image_mask
                audio_mask = None
                
                multi_modal_mask_cache.append((audio_mask, image_mask, text_mask))
                kwargs['multi_modal_mask'] = (audio_mask, image_mask, text_mask)
            elif 'multi_modal_mask' not in kwargs and hasattr(model, "_mas_mm_mask") and model._mas_mm_mask is not None:
                mm = model._mas_mm_mask
                # LLaVA expands IMAGE_TOKEN_INDEX into many vision tokens before layer-0;
                # rebuild mask to match Catcher activation length when needed.
                if ("llava-onevision" in args.model.lower()) or ("onevision" in args.model.lower()):
                    mm = _expand_llava_mm_mask_to_inp(mm, inp, getattr(model, "_mas_input_ids", None))
                multi_modal_mask_cache.append(mm)
                kwargs["multi_modal_mask"] = mm
            elif 'multi_modal_mask' in kwargs:
                multi_modal_mask_cache.append(kwargs['multi_modal_mask'])
            
            if torch.isnan(inp).any():
                import pdb;pdb.set_trace()
            inps.append(inp.squeeze(0))
            cache["i"] += 1
            if 'position_embeddings' in kwargs:
                position_embeddings_cache.append(kwargs["position_embeddings"])
            attention_mask_cache.append(kwargs.get("attention_mask"))
            if self.is_llama and "position_ids" in kwargs:
                position_ids_cache.append(kwargs["position_ids"])
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    def _expand_llava_mm_mask_to_inp(mm_mask, inp, input_ids):
        """Expand pre-IMAGE_TOKEN mask to post-multimodal-merge sequence length."""
        audio_mask, image_mask, text_mask = mm_mask
        if inp.dim() == 2:
            # [S, H] -> treat as batch 1
            seq_len, hidden = inp.shape
            bsz = 1
        else:
            bsz, seq_len, hidden = inp.shape
        if image_mask is not None and image_mask.shape[1] == seq_len:
            # already aligned; only fix hidden broadcast if needed
            if image_mask.shape[-1] != hidden:
                image_mask = image_mask[..., :1].expand(bsz, seq_len, hidden).contiguous()
                text_mask = ~image_mask
                audio_mask = torch.zeros_like(image_mask, dtype=torch.bool)
            return audio_mask, image_mask, text_mask

        from llava.constants import IMAGE_TOKEN_INDEX
        device = inp.device
        if input_ids is None:
            # Fallback: treat all tokens as text (LET still runs; modality split weak)
            image_mask = torch.zeros((bsz, seq_len, hidden), dtype=torch.bool, device=device)
            text_mask = torch.ones_like(image_mask)
            audio_mask = torch.zeros_like(image_mask)
            return audio_mask, image_mask, text_mask

        ids = input_ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        expanded = []
        for b in range(bsz):
            row = ids[b] if b < ids.shape[0] else ids[0]
            s0 = row.numel()
            n_img = int((row == IMAGE_TOKEN_INDEX).sum().item())
            n_text = s0 - n_img
            n_vision = max(seq_len - n_text, 0)
            if n_img <= 0 or n_vision <= 0:
                expanded.append(torch.zeros(seq_len, dtype=torch.bool, device=device))
                continue
            # Distribute vision tokens across IMAGE_TOKEN placeholders (left-to-right).
            base, rem = divmod(n_vision, n_img)
            parts = []
            img_i = 0
            for tok in row.tolist():
                if tok == IMAGE_TOKEN_INDEX:
                    span = base + (1 if img_i < rem else 0)
                    parts.append(torch.ones(span, dtype=torch.bool, device=device))
                    img_i += 1
                else:
                    parts.append(torch.zeros(1, dtype=torch.bool, device=device))
            pos = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.bool, device=device)
            if pos.numel() < seq_len:
                pos = torch.cat([pos, torch.zeros(seq_len - pos.numel(), dtype=torch.bool, device=device)])
            elif pos.numel() > seq_len:
                pos = pos[:seq_len]
            expanded.append(pos)
        image_1d = torch.stack(expanded, dim=0)  # [B, S]
        image_mask = image_1d.unsqueeze(-1).expand(bsz, seq_len, hidden).contiguous()
        text_mask = ~image_mask
        audio_mask = torch.zeros_like(image_mask, dtype=torch.bool)
        return audio_mask, image_mask, text_mask

    def _stash_mm_mask_from_batch(batch):
        """Build MAS (audio, image, text) mask from calibration input_ids."""
        if "input_ids" not in batch:
            model._mas_mm_mask = None
            model._mas_input_ids = None
            return
        input_ids = batch["input_ids"]
        model._mas_input_ids = input_ids.detach().cpu() if torch.is_tensor(input_ids) else input_ids
        if "internvl" in args.model.lower():
            from models.internvl2.constants import IMG_CONTEXT_TOKEN
            image_token_id = lm.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN) if hasattr(lm, "tokenizer") else None
            if image_token_id is None:
                image_token_id = getattr(model, "img_context_token_id", None)
            if image_token_id is None:
                image_token_id = 92546
            image_mask = input_ids == image_token_id
        elif ("llava-onevision" in args.model.lower()) or ("onevision" in args.model.lower()):
            from llava.constants import IMAGE_TOKEN_INDEX
            # Pre-expansion placeholder; Catcher expands to post-merge length.
            image_mask = input_ids == IMAGE_TOKEN_INDEX
        else:
            model._mas_mm_mask = None
            return
        # Expand to [B, S, H] to match QuantLinear.get_masked_value (exact shape).
        if "internvl" in args.model.lower():
            hidden = int(model.language_model.config.hidden_size)
        else:
            hidden = int(getattr(model.config, "hidden_size", image_mask.shape[-1] if image_mask.dim() > 2 else 0) or 0)
            if hidden <= 0 and hasattr(model, "model") and hasattr(model.model, "config"):
                hidden = int(getattr(model.model.config, "hidden_size", 0) or 0)
            if hidden <= 0:
                hidden = 1
        image_mask = image_mask.unsqueeze(-1).expand(-1, -1, hidden).contiguous().to(dev)
        audio_mask = torch.zeros_like(image_mask, dtype=torch.bool)
        text_mask = ~image_mask
        model._mas_mm_mask = (audio_mask, image_mask, text_mask)

    with torch.no_grad():
        if args.epochs > 0:
            for batch in dataloader:
                if cache["i"] >= args.nsamples:
                    break
                try:
                    if 'Qwen2.5-Omni' in args.model or 'Qwen2.5-VL' in args.model:
                        inputs = {k: v.to(dev) for k, v in batch.items()}
                        model(**inputs)
                    elif 'MiniCPM' in args.model:
                        inputs = {k: v.to(dev) for k, v in batch.items()}
                        model(**inputs)
                    elif "internvl" in args.model.lower():
                        _stash_mm_mask_from_batch(batch)
                        inputs = {}
                        for k, v in batch.items():
                            if torch.is_tensor(v):
                                if v.is_floating_point():
                                    inputs[k] = v.to(device=dev, dtype=model.dtype)
                                else:
                                    inputs[k] = v.to(dev)
                            else:
                                inputs[k] = v
                        # set img context id for feature inject
                        if hasattr(lm, "tokenizer"):
                            from models.internvl2.constants import IMG_CONTEXT_TOKEN
                            model.img_context_token_id = lm.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
                        model(**inputs)
                    elif ("llava-onevision" in args.model.lower()) or ("onevision" in args.model.lower()):
                        _stash_mm_mask_from_batch(batch)
                        input_ids = batch["input_ids"].to(dev)
                        attention_mask = batch["attention_mask"].to(dev)
                        images = batch["images"]
                        if torch.is_tensor(images):
                            images = images.to(dev)
                        elif isinstance(images, list):
                            images = [t.to(dev) for t in images]
                        model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            images=images,
                            image_sizes=batch.get("image_sizes"),
                            modalities=batch.get("modalities"),
                        )
                    else:                
                        model(batch["input_ids"].to(dev), batch['attention_mask'])
                except ValueError:
                    pass
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "internvl" in args.model.lower():
        lm_inner = model.language_model.model
        if hasattr(lm_inner, "tok_embeddings"):
            lm_inner.tok_embeddings = lm_inner.tok_embeddings.cpu()
        if hasattr(lm_inner, "embed_tokens"):
            lm_inner.embed_tokens = lm_inner.embed_tokens.cpu()
        lm_inner.norm = lm_inner.norm.cpu()
        # Vision unused after Catcher; free VRAM for long-seq LET.
        if hasattr(model, "vision_model") and model.vision_model is not None:
            model.vision_model = model.vision_model.cpu()
        if hasattr(model, "mlp1") and model.mlp1 is not None:
            model.mlp1 = model.mlp1.cpu()
    elif ("llava-onevision" in args.model.lower()) or ("onevision" in args.model.lower()):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model, "get_vision_tower"):
            vt = model.get_vision_tower()
            if vt is not None:
                vt.to("cpu")
        for attr in ("mm_projector", "vision_resampler"):
            if hasattr(model.model, attr) and getattr(model.model, attr) is not None:
                setattr(model.model, attr, getattr(model.model, attr).cpu())
    elif "llama" in args.net.lower() or "mixtral" in args.net.lower() or 'qwen' in args.net.lower() or 'minicpm' in args.net.lower():
        if 'Qwen2.5-VL' in args.model:
            model.language_model.embed_tokens = model.language_model.embed_tokens.cpu()
            model.language_model.norm = model.language_model.norm.cpu()
        else:
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    def _to_cpu(obj):
        if obj is None:
            return None
        if torch.is_tensor(obj):
            return obj.detach().cpu() if obj.requires_grad else obj.cpu()
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_cpu(x) for x in obj)
        return obj

    def _to_dev(obj):
        if obj is None:
            return None
        if torch.is_tensor(obj):
            return obj.to(dev)
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_dev(x) for x in obj)
        return obj

    # Keep activation / mask caches on CPU; pin one sample at a time during LET.
    inps = [_to_cpu(x) for x in inps]
    attention_mask_cache[:] = [_to_cpu(x) for x in attention_mask_cache]
    position_embeddings_cache[:] = [_to_cpu(x) for x in position_embeddings_cache]
    position_ids_cache[:] = [_to_cpu(x) for x in position_ids_cache]
    multi_modal_mask_cache[:] = [_to_cpu(x) for x in multi_modal_mask_cache]
    torch.cuda.empty_cache()

    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input

    def _forward_sample(module, act, j):
        """Run one calib sample on GPU; caches stay on CPU."""
        x = _to_dev(act)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        kwargs = {}
        if attention_mask_cache:
            kwargs["attention_mask"] = _to_dev(attention_mask_cache[j])
        if position_embeddings_cache:
            kwargs["position_embeddings"] = _to_dev(position_embeddings_cache[j])
        elif position_ids_cache:
            kwargs["position_ids"] = _to_dev(position_ids_cache[j])
        if multi_modal_mask_cache:
            kwargs["multi_modal_mask"] = _to_dev(multi_modal_mask_cache[j])
        return module(x, **kwargs)[0]

    loss_func = torch.nn.MSELoss()

    if args.resume:
        mas_parameters = torch.load(args.resume)
        if not isinstance(mas_parameters, dict) or not all(i in mas_parameters for i in range(len(layers))):
            have = sorted(mas_parameters.keys()) if isinstance(mas_parameters, dict) else type(mas_parameters)
            raise ValueError(
                f"Incomplete MAS checkpoint for resume: need layers 0..{len(layers)-1}, "
                f"got {have} from {args.resume}. "
                f"Delete/quarantine this file and retrain (overnight find_mas_ckpt may have picked an interrupted run)."
            )
    else:
        mas_parameters = {}
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            if 'Qwen2.5-Omni' in args.model:
                qlayer = DecoderLayer(lm.model.config.text_config, layer, args, layer_idx=i)
            elif 'MiniCPM' in args.model:
                qlayer = DecoderLayer(lm.model.config, layer, args, layer_idx=i)
            elif "llama" in args.net.lower():
                qlayer = DecoderLayer(lm.model.config, layer, args)
            elif 'Qwen2.5-VL' in args.model:
                qlayer = DecoderLayer(lm.model.config, layer, args, layer_idx=i)
            elif "internvl" in args.model.lower():
                qlayer = DecoderLayer(model.language_model.config, layer, args, layer_idx=i)
            elif ("llava-onevision" in args.model.lower()) or ("onevision" in args.model.lower()):
                # LlavaQwen: language config fields live on the root config
                qlayer = DecoderLayer(model.config, layer, args, layer_idx=i)
        qlayer = qlayer.to(dev)
        apply_mixed_precision_policy(qlayer, args, i, len(layers))

        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    for j in range(args.nsamples):
                        out = _forward_sample(qlayer, fp_inps[j], j)
                        if torch.isnan(out).any():
                            raise RuntimeError(f"NaN in fp_inps forward (layer={i}, sample={j})")
                        fp_inps[j] = _to_cpu(out)
                        if args.aug_loss:
                            fp_inps_2[j] = _to_cpu(_forward_sample(qlayer, quant_inps[j], j))
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=True, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let and args.resume is None:
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act_all_in_one = act_scales[f"{layer_name_prefix}.{i}.{name}.all_in_one_scale"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            act_text = act_scales[f"{layer_name_prefix}.{i}.{name}.text_scale"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            act_audio = act_scales[f"{layer_name_prefix}.{i}.{name}.audio_scale"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            act_vision = act_scales[f"{layer_name_prefix}.{i}.{name}.vision_scale"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale_all_in_one = (act_all_in_one.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            scale_text = (act_text.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            scale_audio = (act_audio.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            alpha_vision = args.alpha #0.75
                            scale_vision = (act_vision.pow(alpha_vision)/weight.pow(1-alpha_vision)).clamp(min=1e-5)
                            print(f'register scales for {layer_name_prefix}.{i}.{name}')
                            
                            with torch.no_grad(): 
                                module.all_in_one_smooth_scale.data.copy_(scale_all_in_one)
                                module.text_smooth_scale.data.copy_(scale_text)
                                module.audio_smooth_scale.data.copy_(scale_audio)
                                module.vision_smooth_scale.data.copy_(scale_vision)
                                                            
                            # 考虑到 vision token 的激活range 相对 text 的大很多，所以 vision 的 scale 也相应要比 text 的 scale alpha 大一些，通过搜索的方式找到更合适的 scale 方法
                            # 0.5 每次增加 0.01 的方式，增加到 0.8，找到一个 loss 最低的 alpha 作为初始值
                            if args.epochs > 0 and args.auto_alpha :
                                alpha_vision_min = args.alpha
                                alpha_vision_max = 0.8
                                alpha_vision_step = 0.01                                
                                alpha_vision = alpha_vision_min
                                max_sqnr_sample_count = 1
                                sqnr_max = 0
                                best_alpha = alpha_vision_min
                                fp_out = []
                                for j in range(args.nsamples):
                                    fp_out.append(_to_cpu(_forward_sample(layers[i], quant_inps[j], j)))
                                    if j >= max_sqnr_sample_count:
                                        break            

                                for alpha_vision in np.arange(alpha_vision_min, alpha_vision_max + alpha_vision_step/2, alpha_vision_step):
                                    # 先应用当前的alpha
                                    scale_vision = (act_vision.pow(alpha_vision)/weight.pow(1-alpha_vision)).clamp(min=1e-5)
                                    with torch.no_grad(): 
                                        module.vision_smooth_scale.data.copy_(scale_vision)
                                    reuse_scale(qlayer, args)
                                    
                                    # 开始统计哪个 alpha 比较好
                                    sqnr_quant = 0
                                    for j in range(args.nsamples):
                                        quant_out = _forward_sample(qlayer, quant_inps[j], j)
                                        sqnr_quant += compute_sqnr_simple(_to_dev(fp_out[j]), quant_out)
                                        # print(f'sqnr_quant_{j}:  {sqnr_quant}, sqnr_max:  {sqnr_max}, quant_out: {quant_out.mean()}, fp_out: {fp_out[j].mean()}')
                                        if j >= max_sqnr_sample_count:
                                            break
                                    # print(f"alpha_vision 当前值: {alpha_vision}, sqnr_max: {sqnr_max}, sqnr_quant: {sqnr_quant}")
                                    if sqnr_quant > sqnr_max:
                                        best_alpha = alpha_vision
                                        sqnr_max = sqnr_quant
                                        # print(f'find new alpha: {best_alpha}')
                                print(f'{layer_name_prefix}.{i}.{name} best_alpha for vision: {best_alpha} ')
                                # 将最好的alpha应用到训练的初始化值
                                scale_vision = (act_vision.pow(best_alpha)/weight.pow(1-best_alpha)).clamp(min=1e-5)
                                with torch.no_grad(): 
                                    module.vision_smooth_scale.data.copy_(scale_vision)
                                reuse_scale(qlayer, args)
                            # 根据实际的 loss 来选择是使用三模态的 scale 还是单模态的 scale 来进行初始化.
                            if args.epochs > 0 and args.auto_scale :
                                reuse_scale(qlayer, args)
                                max_sqnr_sample_count = 10
                                sqnr_split = 0
                                sqnr_merged = 0
                                fp_out = []
                                for j in range(args.nsamples):
                                    fp_out.append(_to_cpu(_forward_sample(layers[i], quant_inps[j], j)))
                                    if j >= max_sqnr_sample_count:
                                        break                                
                                # 先计算三模态的sqnr
                                for j in range(args.nsamples):
                                    quant_out = _forward_sample(qlayer, quant_inps[j], j)
                                    sqnr_split += compute_sqnr_simple_multimodal(
                                        _to_dev(fp_out[j]), quant_out, multi_modal_mask=_to_dev(multi_modal_mask_cache[j])
                                    )
                                    if j >= max_sqnr_sample_count:
                                        break
                                # 再计算统一模态的sqnr， 注意需要先修改 scale 为统一模态的值.
                                with torch.no_grad(): 
                                    module.text_smooth_scale.data.copy_(scale_all_in_one)
                                    module.audio_smooth_scale.data.copy_(scale_all_in_one)
                                    module.vision_smooth_scale.data.copy_(scale_all_in_one)                                
                                for j in range(args.nsamples):
                                    quant_out = _forward_sample(qlayer, quant_inps[j], j)
                                    sqnr_merged += compute_sqnr_simple_multimodal(
                                        _to_dev(fp_out[j]), quant_out, multi_modal_mask=_to_dev(multi_modal_mask_cache[j])
                                    )
                                    if j >= max_sqnr_sample_count:
                                        break

                                # 如果是三模态的 sqnr更大，则表明三模态scale 对这一层更加友好，那么就将 scale 复原成三模态的值
                                print(f'--->>>> sqnr_split: {sqnr_split}, sqnr_merged: {sqnr_merged} ')
                                if sqnr_split > sqnr_merged:
                                    print(f'{layer_name_prefix}.{i}.{name} 使用三模态的 scale 来进行初始化.')
                                    with torch.no_grad(): 
                                        module.all_in_one_smooth_scale.data.copy_(scale_all_in_one)
                                        module.text_smooth_scale.data.copy_(scale_text)
                                        module.audio_smooth_scale.data.copy_(scale_audio)
                                        module.vision_smooth_scale.data.copy_(scale_vision)                                
                                else:
                                    print(f'{layer_name_prefix}.{i}.{name} 使用统一模态的 scale 来进行初始化.')


        if args.resume:
            qlayer.load_state_dict(mas_parameters[i], strict=False)
        

        if args.epochs > 0:
            # 先调用 reuse_scale 来设置参数共享，然后再创建优化器
            reuse_scale(qlayer, args)
            
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}],weight_decay=args.wd)            
            # loss_scaler = utils.NativeScalerWithGradNormCount()
            loss_scaler = utils.FlexibleScaler(dtype=dtype)
            
            max_epochs = args.epochs
            epoch = 0

            # 记录训练结束时候的 loss 值
            while epoch < max_epochs:
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):
                    # obtain output of quantization model
                    with traincast():
                        truncate_scale(qlayer, args)
                        quant_out = _forward_sample(qlayer, quant_inps[j], j)
                        fp_tgt = _to_dev(fp_inps[j])
                        if multi_modal_mask_cache:
                            audio_mask, image_mask, text_mask = _to_dev(multi_modal_mask_cache[j])
                        else:
                            audio_mask = image_mask = text_mask = None
                        
                        if args.loss_multi_modal: 
                            quant_diff = (quant_out-fp_tgt).pow(2)
                            text_diff = quant_diff*text_mask
                            audio_diff = quant_diff*audio_mask
                            vision_diff = quant_diff*image_mask
                            
                            text_loss = text_diff.sum() / text_mask.sum()
                            audio_loss = audio_diff.sum() / audio_mask.sum()
                            vision_loss = vision_diff.sum() / image_mask.sum()
                            
                            loss_multi_modal = text_loss + audio_loss + vision_loss
                            # print(f'loss_multi_modal: {loss_multi_modal}, text: {text_loss/loss_multi_modal}, vision: {vision_loss/loss_multi_modal}, audio: {audio_loss/loss_multi_modal}')
                            loss = loss_multi_modal
                        elif args.loss_multi_modal_mae: 
                            quant_diff = (quant_out-fp_tgt).abs()
                            text_diff = quant_diff*text_mask
                            audio_diff = quant_diff*audio_mask
                            vision_diff = quant_diff*image_mask
                            
                            total_mask = (text_mask.sum() + audio_mask.sum() + image_mask.sum())
                            loss = (text_diff.sum() + audio_diff.sum() + vision_diff.sum()) / total_mask
                        elif args.loss_multi_modal_mae_alpha:
                            if getattr(args, 'auto_modal_weight', False):
                                loss = sensitivity_aware_modal_loss(
                                    quant_out,
                                    fp_tgt,
                                    (audio_mask, image_mask, text_mask),
                                    tau=getattr(args, 'modal_weight_tau', 0.05),
                                    auto_weight=True,
                                    grad_info=grad_info,
                                    layer_idx=i,
                                )
                            else:
                                quant_diff = (quant_out-fp_tgt).abs()
                                text_diff = quant_diff*text_mask
                                total_mask = text_mask.sum()
                                
                                if audio_mask is not None:
                                    audio_diff = quant_diff*audio_mask
                                    total_mask += audio_mask.sum()
                                    audio_grad = 1.0
                                    if grad_info is not None:
                                        # 这个是实际通过 MBQ 的梯度信息去读取的，实际发现，效果并不好，不如取所有 layer中的 max 梯度作为各层的梯度的值，所以这个0.5478636880998574是 qwen2.5-omni-3b 的值.
                                        audio_grad = grad_info[f'layers.{i}']['aud_avg_grad'] / grad_info[f'layers.{i}']['cap_avg_grad']
                                        # audio_grad = 0.5478636880998574
                                    audio_diff = audio_grad * audio_diff
                                    
                                if image_mask is not None:
                                    vision_diff = quant_diff*image_mask
                                    total_mask += image_mask.sum()
                                    # 针对 qwen-omni 模型，这个 0.5 是瞎猜的,  text : audio : vision = 1 : 1 : 0.5 
                                    vision_grad = 0.5
                                    if grad_info is not None:
                                        # 这个 0.14992747247178426 的意义同audio_grad
                                        # 针对 qwen-vl-3b 模型，这个0.126 是通过 MBQ 进行 per-layer梯度信息采集得到的 0.12593715319890053. text: vision = 1: 0.126
                                        # 针对qwen-vl-7b 0.11022980689742588，接近于 0.11
                                        vision_grad = grad_info[f'layers.{i}']['vis_avg_grad'] / grad_info[f'layers.{i}']['cap_avg_grad']
                                        # vision_grad = 0.14992747247178426
                                        vision_grad = 0.126
                                    vision_diff = vision_grad * vision_diff
                                if audio_mask is not None and image_mask is not None:
                                    loss = (text_diff.sum() + audio_diff.sum() + vision_diff.sum()) / total_mask
                                else:
                                    loss = (text_diff.sum() + vision_diff.sum()) / total_mask
                        else:
                            loss = loss_func(fp_tgt.to(torch.float32), quant_out.to(torch.float32))
                        if args.aug_loss:
                            loss += loss_func(_to_dev(fp_inps_2[j]), quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        import pdb; pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters= get_mas_parameters(qlayer, use_shift))
                    if norm is not None:
                        norm_list.append(norm.cpu().data)
                    
                    if not check_gradient(qlayer):
                        print("梯度爆炸！训练终止。")
                        import pdb;pdb.set_trace()

                if len(loss_list) > 0:
                    loss_mean = torch.stack(loss_list).mean()
                else:
                    loss_mean = 0
                if len(norm_list) > 0:
                    norm_mean = torch.stack(norm_list).mean()
                else:
                    norm_mean = 0
                logger.info(f"layer {i} iter {epoch} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                epoch += 1
                if args.auto_epochs and loss_mean.data > 1.0 and max_epochs <= args.epochs:
                    max_epochs += max_epochs
                    logger.info(f"Extending training. New max_epochs: {max_epochs}")
            clear_temp_variable(qlayer)
            del optimizer
            # Scales can go negative after the last AdamW step; post-train
            # forward then hits input/(scale+eps)~Inf (W8A8/W4A16 crash).
            truncate_scale(qlayer, args)
        # real smooth and quantization
        # smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = _to_cpu(_forward_sample(qlayer, quant_inps[j], j))
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            mas_parameters[i] = mas_state_dict(qlayer)
            torch.save(mas_parameters, os.path.join(args.output_dir, f"mas_parameters.pth"))
        else:
            #  merge_scale的情况下，才需要进行 reuse，这说明是在跑 MAS-Quant 训练出来的模式
            # args.resume 为 False，但是 act_scale 为 True，这说明是在跑 smoothquant 原始的效果
            if os.getenv('inference_mode', 'merged_scales') == 'merged_scales' or (args.resume is None and args.act_scales is not None):
                reuse_scale(qlayer, args)
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    if 'Qwen2.5-Omni' in args.model:
        model.config.text_config.use_cache = use_cache
    elif "internvl" in args.model.lower():
        model.language_model.config.use_cache = use_cache
    elif 'MiniCPM' in args.model or 'llama' in args.model or "onevision" in args.model.lower():
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = use_cache
    return model
