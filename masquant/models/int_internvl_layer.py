# coding=utf-8
"""Quantized InternLM2 decoder layer for InternVL2 (packed wqkv + w1/w2/w3 MLP)."""

from collections import OrderedDict
from typing import Optional, Tuple

import math
import torch
from torch import nn
from einops import rearrange

from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from models.transformation import truncate_number


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class QuantInternVLMLP(nn.Module):
    def __init__(self, org_module, args=None, layer_idx=0):
        super().__init__()
        support_training = getattr(args, "mode", "train") == "train"
        self.w1 = QuantLinear(
            org_module.w1, args.weight_quant_params, args.act_quant_params,
            support_training=support_training, name=f"w1_{layer_idx}", layer_index=layer_idx, mode=args.mode,
        )
        self.w3 = QuantLinear(
            org_module.w3, args.weight_quant_params, args.act_quant_params,
            support_training=support_training, name=f"w3_{layer_idx}", layer_index=layer_idx, mode=args.mode,
        )
        self.w2 = QuantLinear(
            org_module.w2, args.weight_quant_params, args.act_quant_params,
            support_training=support_training, name=f"w2_{layer_idx}", layer_index=layer_idx, mode=args.mode,
        )
        self.act_fn = org_module.act_fn

    def forward(self, x, multi_modal_mask=None):
        return self.w2(self.act_fn(self.w1(x, multi_modal_mask)) * self.w3(x, multi_modal_mask), multi_modal_mask)


class QuantInternVLAttention(nn.Module):
    def __init__(self, org_module, config, args=None, layer_idx=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        support_training = getattr(args, "mode", "train") == "train"
        self.wqkv = QuantLinear(
            org_module.wqkv, args.weight_quant_params, args.act_quant_params,
            support_training=support_training, name=f"wqkv_{layer_idx}", layer_index=layer_idx, mode=args.mode,
        )
        self.wo = QuantLinear(
            org_module.wo, args.weight_quant_params, args.act_quant_params,
            support_training=support_training, name=f"wo_{layer_idx}", layer_index=layer_idx, mode=args.mode,
        )
        self.rotary_emb = org_module.rotary_emb

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        multi_modal_mask=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        qkv_states = self.wqkv(hidden_states, multi_modal_mask)
        qkv_states = rearrange(
            qkv_states,
            "b q (h gs d) -> b q h gs d",
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )
        query_states = rearrange(qkv_states[..., : self.num_key_value_groups, :], "b q h gs d -> b q (h gs) d")
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.wo(attn_output, multi_modal_mask)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


def _clone_rms_norm(norm: nn.Module) -> nn.Module:
    eps = getattr(norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(norm, "eps", 1e-6)
    device = norm.weight.device
    dtype = norm.weight.dtype
    cloned = type(norm)(norm.weight.shape[0], eps=eps)
    cloned.weight = nn.Parameter(norm.weight.detach().clone().to(device=device, dtype=dtype))
    return cloned


class QuantInternVLDecoderLayerV2(nn.Module):
    def __init__(self, config, ori_layer, args, layer_idx=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = QuantInternVLAttention(ori_layer.attention, config, args=args, layer_idx=layer_idx)
        self.feed_forward = QuantInternVLMLP(ori_layer.feed_forward, args=args, layer_idx=layer_idx)
        # Independent norm copies so layers[i].cpu() cannot yank qlayer norms off CUDA.
        self.attention_norm = _clone_rms_norm(ori_layer.attention_norm)
        self.ffn_norm = _clone_rms_norm(ori_layer.ffn_norm)
        # aliases so LET/smooth helpers can find norms if needed later
        self.input_layernorm = self.attention_norm
        self.post_attention_layernorm = self.ffn_norm

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        position_embeddings=None,
        multi_modal_mask=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            multi_modal_mask=multi_modal_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, multi_modal_mask)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)

    def smooth_and_quant_temporary(self):
        # Packed InternLM2: skip classic q/k/v LET; only temp-quant weights
        for module in self.modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight_quantizer(module.weight)
                module.temp_bias = module.bias
                module.use_temporary_parameter = True

    def clear_temp_variable(self):
        for module in self.modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    del module.temp_weight
                if hasattr(module, "temp_bias"):
                    del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        for module in self.modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter = False

    def let_parameters(self, use_shift=True):
        # Train QuantLinear smooth scales (same pattern as MiniCPM / Qwen VL).
        template = "smooth" if use_shift else "smooth_scale"
        params = [m for n, m in self.named_parameters() if n.find(template) > -1]
        return iter(params)

    def lwc_parameters(self):
        params = [m for n, m in self.named_parameters() if "bound_factor" in n]
        return iter(params)

    def omni_parameters(self, use_shift=True):
        template = "smooth" if use_shift else "smooth_scale"
        params = [
            m
            for n, m in self.named_parameters()
            if n.find("bound_factor") > -1 or n.find(template) > -1
        ]
        return iter(params)

    def omni_state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if "smooth" in name or "bound_factor" in name:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
