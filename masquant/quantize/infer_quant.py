# coding=utf-8
import torch
from quantize.int_linear import QuantLinear


def mas_quantize_model(
        model, low_rank_adapters, text_scales, vision_scales, audio_scales, args
):
    dev = "cuda"
    model_key = args.model.lower()
    if "omni" in model_key:
        layers = model.model.layers
        from models.int_qwen_omni_layer import QuantQwenDecoderLayerV2 as DecoderLayer
        cfg = model.config.text_config
    elif "internvl" in model_key:
        layers = model.language_model.model.layers
        from models.int_internvl_layer import QuantInternVLDecoderLayerV2 as DecoderLayer
        cfg = model.language_model.config
    elif ("llava-onevision" in model_key) or ("llava_onevision" in model_key) or (
        "onevision" in model_key and "llava" in model_key
    ):
        layers = model.model.layers
        from models.int_minicpm_layer import QuantMiniCPMDecoderLayerV2 as DecoderLayer
        cfg = model.config
    else:
        layers = model.model.language_model.layers
        from models.int_qwen_vl_layer import QuantQwenDecoderLayerV2 as DecoderLayer
        cfg = model.config

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        print(f"=== Start quantize layer {i} ===")
        qlayer = DecoderLayer(cfg, layer, args, layer_idx=i)
        qlayer = qlayer.to(dev)
        qlayer.set_quant_state(weight_quant=True, act_quant=True)
        layers[i] = qlayer

    filter_modules = ["visual", "vision", "lm_head", "audio", "vision_tower", "mm_projector", "mlp1"]
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear) and not any(f in name for f in filter_modules):
            if args.rank > 0 and name in low_rank_adapters["vision"].keys():
                m.Lv = low_rank_adapters["vision"][name]["L"].to(m.weight.dtype)
                m.Rv = low_rank_adapters["vision"][name]["R"].to(m.weight.dtype)
                if "audio" in low_rank_adapters.keys():
                    m.La = low_rank_adapters["audio"][name]["L"].to(m.weight.dtype)
                    m.Ra = low_rank_adapters["audio"][name]["R"].to(m.weight.dtype)
                else:
                    m.La = None
                    m.Ra = None
            m.text_smooth_scale = text_scales[name]
            m.vision_smooth_scale = vision_scales[name]
            m.audio_smooth_scale = audio_scales[name]
            target_dtype = torch.bfloat16
            cur_dtype = m.weight.dtype
            m.q_weight = m.weight_quantizer(
                (m.weight.to(target_dtype) * m.text_smooth_scale).to(cur_dtype)
            )
    return model
