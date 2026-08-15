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

# Monkey patch for MiniCPM Resampler to fix _initialize_weights issue
def patch_minicpm_resampler():
    """Patch MiniCPM's Resampler class to add missing _initialize_weights method."""
    original_import = __builtins__.__import__
    
    def custom_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if 'resampler' in name.lower():
            if hasattr(module, 'Resampler'):
                Resampler = module.Resampler
                if hasattr(Resampler, '_init_weights') and not hasattr(Resampler, '_initialize_weights'):
                    original_init = Resampler._init_weights
                    def _initialize_weights(self, module=None):
                        """Initialize weights for transformers compatibility."""
                        if module is not None:
                            original_init(self, module)
                        else:
                            self.apply(original_init)
                    Resampler._initialize_weights = _initialize_weights
        return module
    
    __builtins__.__import__ = custom_import

patch_minicpm_resampler()

import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
import argparse
import torch.nn as nn
import transformers

from datasets import load_dataset
import functools
from tqdm import tqdm
from datautils import get_loaders
try:
    from llava.model import *   # required for llava
except ImportError:
    print("If want to quantize llave models, you should manually install llava from https://github.com/haotian-liu/LLaVA")

# 仅考虑vision/audio/text 的情况，其中 vision/audio 是不包含各自的 start/end token; text 是其余 system/start/end/video 相关的.
def get_act_scales(model, dataloader, num_samples=128, model_type="qwen", image_token_index=None):
    if model_type == "internvl":
        audio_token_index = -1
        if image_token_index is None:
            image_token_index = getattr(model, "img_context_token_id", 92546)
    elif model_type == "llava_onevision":
        audio_token_index = -1
        if image_token_index is None:
            from llava.constants import IMAGE_TOKEN_INDEX
            image_token_index = IMAGE_TOKEN_INDEX
    elif isinstance(model, transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration):
        audio_token_index = -1
        image_token_index = model.config.image_token_id
    elif model_type == "minicpm":
        # MiniCPM uses Qwen2 architecture, has different token indices
        # For MiniCPM, we need to check the actual token IDs used
        # Based on the model config, image tokens are special tokens
        # We'll use a placeholder value and detect from actual data
        audio_token_index = -1  # MiniCPM doesn't have audio
        # MiniCPM image token ID - need to get from config or use common value
        # MiniCPM typically uses <image> token, which gets tokenized
        # We'll detect image tokens dynamically or use -2 as placeholder
        image_token_index = -2  # Will be detected from data
        # model is already the llm part for MiniCPM
    else:
        audio_token_index = model.config.thinker_config.audio_token_index
        image_token_index = model.config.thinker_config.image_token_index
        video_token_id = model.config.thinker_config.video_token_id
        model = model.thinker
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}
    text_mask = None; image_mask = None; audio_mask = None

    def _update_scales(name, comming_max, comming_max_text, comming_max_vision, comming_max_audio):
        if any(k.startswith(name + ".") for k in act_scales.keys()):
            act_scales[name + ".all_in_one_scale"] = torch.max(act_scales[name + ".all_in_one_scale"], comming_max)
            act_scales[name + ".text_scale"] = torch.max(act_scales[name + ".text_scale"], comming_max_text)
            act_scales[name + ".vision_scale"] = torch.max(act_scales[name + ".vision_scale"], comming_max_vision)
            act_scales[name + ".audio_scale"] = torch.max(act_scales[name + ".audio_scale"], comming_max_audio)
        else:
            act_scales[name + ".all_in_one_scale"] = comming_max
            act_scales[name + ".text_scale"] = comming_max_text
            act_scales[name + ".vision_scale"] = comming_max_vision
            act_scales[name + ".audio_scale"] = comming_max_audio

    def stat_tensor(name, tensor, text_mask=None, image_mask=None, audio_mask=None, act_scales=None):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()

        # Masks are built from LLM input_ids; skip modality split on shape mismatch
        # (e.g. residual vision hooks, or LLaVA pre-expansion placeholders).
        try:
            text_m = text_mask.squeeze().unsqueeze(-1).expand_as(tensor)
            image_m = image_mask.squeeze().unsqueeze(-1).expand_as(tensor)
            audio_m = audio_mask.squeeze().unsqueeze(-1).expand_as(tensor)
        except RuntimeError:
            _update_scales(name, comming_max, comming_max, comming_max, torch.zeros_like(comming_max))
            return

        tensor_text = tensor[text_m].view(-1, hidden_dim).abs().detach()
        if tensor_text.numel() == 0:
            comming_max_text = torch.zeros(hidden_dim)
        else:
            comming_max_text = torch.max(tensor_text, dim=0)[0].float().cpu()

        tensor_audio = tensor[audio_m].view(-1, hidden_dim).abs().detach()
        if tensor_audio.numel() == 0:
            comming_max_audio = torch.zeros(hidden_dim)
        else:
            comming_max_audio = torch.max(tensor_audio, dim=0)[0].float().cpu()

        tensor_vision = tensor[image_m].view(-1, hidden_dim).abs().detach()
        if tensor_vision.numel() == 0:
            comming_max_vision = torch.zeros(hidden_dim)
        else:
            comming_max_vision = torch.max(tensor_vision, dim=0)[0].float().cpu()

        _update_scales(name, comming_max, comming_max_text, comming_max_vision, comming_max_audio)

    def stat_input_hook(m, x, y, name, act_scales):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, text_mask=text_mask, image_mask=image_mask, audio_mask=audio_mask, act_scales=act_scales)

    hooks = []
    # Only collect LLM Linear activations; vision towers have different seq lengths.
    if model_type == "internvl":
        filter_modules = ["vision_model", "mlp1", "lm_head", "output", "tok_embeddings", "embed_tokens"]
    elif model_type == "llava_onevision":
        filter_modules = ["vision_tower", "mm_projector", "vision_resampler", "lm_head", "embed_tokens"]
    else:
        filter_modules = ["visual", "lm_head", "audio"]
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(f in name for f in filter_modules):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name, act_scales=act_scales)))

    for i in tqdm(range(num_samples), desc="generate act scale"):
        
        if transformers.__version__ == "4.31.0":
            model(dataloader[i][0].to(device))
        else:
            image_mask = (dataloader[i]['input_ids'] == image_token_index) 
            audio_mask = (dataloader[i]['input_ids'] == audio_token_index)
            all_true = torch.full(image_mask.shape, True, dtype=torch.bool)
            text_mask = all_true & ~audio_mask & ~image_mask
            batch = dataloader[i]
            if model_type == "llava_onevision":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                images = batch["images"]
                if torch.is_tensor(images):
                    images = images.to(device)
                elif isinstance(images, list):
                    images = [t.to(device) for t in images]
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    image_sizes=batch.get("image_sizes"),
                    modalities=batch.get("modalities"),
                )
            else:
                inputs = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        # InternVL vision expects pixel_values dtype == model.dtype (bf16)
                        if v.is_floating_point() and model_type == "internvl":
                            inputs[k] = v.to(device=device, dtype=model.dtype)
                        else:
                            inputs[k] = v.to(device)
                    else:
                        inputs[k] = v
                if model_type == "internvl":
                    model.img_context_token_id = image_token_index
                model(**inputs)

    for h in hooks:
        h.remove()

    print(f'act_scales: {act_scales}')
    return act_scales

from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

def build_model_and_tokenizer(model_name):
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        processor = None
    
    # 注意，我们只使用 omni 的 thinker 模块
    if 'Qwen2.5-Omni' in model_name:
        from transformers import Qwen2_5OmniForConditionalGeneration    
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": 'auto', 'enable_audio_output': False, 'attn_implementation': 'flash_attention_2'}
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **kwargs)
        model.to('cuda')
    elif 'MiniCPM' in model_name:
        # For MiniCPM in activation scale generation, we only use the llm part
        # We don't need the vision encoder since we're only quantizing the llm
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": 'auto', "trust_remote_code": True}
        full_model = AutoModel.from_pretrained(model_name, **kwargs)
        model = full_model.llm
        # For MiniCPM, use only tokenizer since we'll use text-only data
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Don't store full_model to avoid recursion issues
        print(f'we will use minicpm llm model only.')
    elif 'Qwen2.5-VL' in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration    
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": 'auto', "trust_remote_code": True, 'attn_implementation': 'flash_attention_2'}
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        model.to('cuda')
    elif "internvl" in model_name.lower():
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True}
        model = AutoModel.from_pretrained(model_name, **kwargs)
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        model.to("cuda")
    elif ("llava-onevision" in model_name.lower()) or ("llava_onevision" in model_name.lower()) or (
        "onevision" in model_name.lower() and "llava" in model_name.lower()
    ):
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        name = get_model_name_from_path(model_name)
        # device_map="auto" can leave the whole model on CPU in this env; force CUDA.
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_name, None, name, device_map="cuda", multimodal=True, attn_implementation="sdpa"
        )
        model = model.cuda().eval()
        processor = tokenizer
        processor.image_processor = image_processor
        print(f"LLaVA act-scale model device: {next(model.parameters()).device}")
    else:
        kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, processor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/nas/yuehu/models/vl/Qwen2.5-VL-3B-Instruct', help='model name')
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',
                        help='where to save the act scales')
    parser.add_argument("--calib_dataset",type=str,default="omnibench",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--cache_dir", type=str, default='./cache')
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="text-audio-vision",
        choices=[
            "text-only",
            "vision-only",
            "audio-only",
            "text-vision",
            "text-audio",
            "vision-audio",
            "text-audio-vision",
            "mas_mix_dataset"
        ],
        help="Data type to calculate activation. Options: text-only, vision-only, audio-only, text-vision, text-audio, vision-audio, mas_mix_dataset"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
        
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model)
    args.net = args.model.rstrip('/').split('/')[-1]
    args.model_family = args.net.split('-')[0]
    cache_dataloader = f'{args.cache_dir}/dataloader_{args.net}_{args.dataset_type}_{args.nsamples}.cache'
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader, weights_only=False)
        print(f"load calibration from {cache_dataloader}")
    else:     
        model_key = args.model.lower()
        if 'Qwen' in args.model or 'MiniCPM' in args.model:
            from custom_dataset import prepare_dataset, prepare_dataset_before_quant
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                AutoProcessor,
            )
            calibration_dataset = prepare_dataset(n_sample=args.nsamples, data_type=args.dataset_type)
            # Use the tokenizer/processor already created in build_model_and_tokenizer
            processor = tokenizer
            is_minicpm = 'MiniCPM' in args.model
            # For MiniCPM, we simplified to use text-only data
            dataloader = prepare_dataset_before_quant(processor, calibration_dataset, batch_size=args.batch_size, is_minicpm=is_minicpm)
        elif "internvl" in model_key or ("onevision" in model_key and "llava" in model_key) or ("llava-onevision" in model_key):
            # Reuse LMClass-shaped shim for calib helpers
            class _Shim:
                pass
            shim = _Shim()
            shim.model = model
            shim.tokenizer = tokenizer
            shim.image_processor = getattr(tokenizer, "image_processor", None)
            from custom_dataset import prepare_calib_internvl, prepare_calib_llava_onevision
            if "internvl" in model_key:
                dataloader = prepare_calib_internvl(shim, n_sample=args.nsamples)
            else:
                dataloader = prepare_calib_llava_onevision(shim, n_sample=args.nsamples)
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=args.seq_len,
            )
        torch.save(dataloader, cache_dataloader)    
    
    # Determine model type
    model_type = "qwen"  # default
    image_token_index = None
    if 'MiniCPM' in args.model:
        model_type = "minicpm"
    elif 'Qwen2.5-VL' in args.model or 'Qwen2.5-Omni' in args.model:
        model_type = "qwen"
    elif "internvl" in args.model.lower():
        model_type = "internvl"
        from models.internvl2.constants import IMG_CONTEXT_TOKEN
        image_token_index = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = image_token_index
    elif ("llava-onevision" in args.model.lower()) or ("onevision" in args.model.lower() and "llava" in args.model.lower()):
        model_type = "llava_onevision"
    
    act_scales = get_act_scales(
        model, dataloader, args.nsamples, model_type=model_type, image_token_index=image_token_index
    )
    save_path = os.path.join(args.scales_output_path,f'{args.net}-{args.dataset_type}-{args.nsamples}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

if __name__ == '__main__':
    main()
