# import torch
# torch.backends.cudnn.deterministic = True
# torch.backends.cuda.matmul.allow_tf32 = False # 禁用 TF32
# torch.use_deterministic_algorithms(True)
# import torch
# torch.autograd.set_detect_anomaly(True)
import argparse

# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

# patch_minicpm_resampler()

import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lmms_eval import evaluator as eval_multimodal
# from lm_eval import evaluator
# from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.masquant import masquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_llama_layer_v2 import QuantLlamaDecoderLayerV2
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
import json
import csv
import pdb

torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
]

def compute_sqnr(original, quantized):
    error = original - quantized
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(error ** 2)
    sqnr = 10 * torch.log10(signal_power / noise_power)
    
    return sqnr.item()

def compute_sqnr_per_modality(original, quantized, audio_mask, vision_mask, vision_alpha=0.5, audio_alpha=1.0):
    """
    计算每个 token 的 SQNR，然后返回平均值
    
    Args:
        original: [batch, seq_len, hidden_dim] 如 [1, 4464, 2048]
        quantized: [batch, seq_len, hidden_dim] 如 [1, 4464, 2048]
    
    Returns:
        mean_sqnr: 所有 token 的平均 SQNR
    """
    error = original.float() - quantized.float()
    # 在最后一维（hidden_dim）上计算信号和噪声功率
    # 结果 shape: [1, 4464]
    signal_power = torch.mean(original ** 2, dim=-1)  # [1, 4464]
    noise_power = torch.mean(error ** 2, dim=-1)      # [1, 4464]
    # 计算每个 token 的 SQNR
    # 添加一个小的 epsilon 避免除零
    sqnr = 10 * torch.log10(signal_power / (noise_power))  # [1, 4464]
    all_true = torch.full(vision_mask.shape, True, dtype=torch.bool).to(vision_mask.device)
    text_mask = all_true & ~audio_mask & ~vision_mask
    sqnr_vision = torch.mean(sqnr[vision_mask])
    sqnr_audio = torch.mean(sqnr[audio_mask])
    sqnr_text = torch.mean(sqnr[text_mask])
  
    # 对所有 token 求平均，注意 vision 的乘以了一个 0.5的系数!!!!
    mean_sqnr = torch.mean(torch.stack([vision_alpha*sqnr_vision, audio_alpha*sqnr_audio, sqnr_text]))
    return mean_sqnr.item()

def compute_sqnr_per_token(original, quantized):
    """
    计算每个 token 的 SQNR，然后返回所有 token 的平均 SQNR。
    
    Args:
        original (torch.Tensor): 原始浮点隐藏状态 (batch_size, sequence_length, hidden_size)
        quantized (torch.Tensor): 量化隐藏状态 (batch_size, sequence_length, hidden_size)
        
    Returns:
        float: 所有 token 的平均 SQNR (dB)
    """
    # original/quantized shape: (batch_size, sequence_length, hidden_size)

    # 1. 计算误差 (噪声)
    error = original - quantized

    # 2. 计算信号功率：对每个 token 向量在 hidden_size 维度上求均值
    # 结果 shape: (batch_size, sequence_length)
    signal_power = torch.mean(original ** 2, dim=-1)

    # 3. 计算噪声功率：对每个 token 向量在 hidden_size 维度上求均值
    # 结果 shape: (batch_size, sequence_length)
    noise_power = torch.mean(error ** 2, dim=-1)

    # 4. 计算每个 token 的 SQNR (dB)
    # 使用 torch.clamp 来避免除以零或对零取对数
    sqnr_matrix = 10 * torch.log10(signal_power / torch.clamp(noise_power, min=1e-5))
    
    # 5. 计算所有 token 的平均 SQNR，并返回 Python float
    mean_sqnr = torch.mean(sqnr_matrix)
    
    return mean_sqnr.item()


def _set_child_module(root, name, module):
    parts = name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def quantize_vision_tower_if_requested(model, args, logger=None):
    if not getattr(args, 'quantize_vit', False):
        return
    visual = None
    if hasattr(model, 'model') and hasattr(model.model, 'visual'):
        visual = model.model.visual
    elif hasattr(model, 'visual'):
        visual = model.visual
    elif hasattr(model, 'vision_model'):
        visual = model.vision_model
    if visual is None:
        if logger is not None:
            logger.warning('quantize_vit requested, but no visual tower was found.')
        return

    weight_params = dict(args.weight_quant_params)
    weight_params['n_bits'] = args.vit_wbits
    weight_params['group_size'] = None
    act_params = dict(args.act_quant_params)
    act_params['n_bits'] = args.vit_abits

    replaced = 0
    for name, module in list(visual.named_modules()):
        if isinstance(module, nn.Linear):
            qlinear = QuantLinear(
                module,
                weight_params,
                act_params,
                support_training=False,
                name=f'visual.{name}',
                mode=args.mode,
            )
            qlinear.set_quant_state(weight_quant=True, act_quant=True)
            _set_child_module(visual, name, qlinear)
            replaced += 1
    if logger is not None:
        logger.info(
            f'quantize_vit enabled: replaced {replaced} visual Linear layers '
            f'with W{args.vit_wbits}A{args.vit_abits} QuantLinear.'
        )
    else:
        print(
            f'quantize_vit enabled: replaced {replaced} visual Linear layers '
            f'with W{args.vit_wbits}A{args.vit_abits} QuantLinear.'
        )


@torch.no_grad()
def evaluate(llm, args, logger):
    results = {}
    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(llm.model.model.decoder.layers)
            input_device = llm.model.model.decoder.layers[0].device
            output_device = llm.model.model.decoder.layers[-1].device
            llm._device = input_device
            assert input_device == output_device
            llm.model.model.decoder.embed_positions.to(input_device)
            llm.model.model.decoder.embed_tokens.to(input_device)
            llm.model.model.decoder.final_layer_norm.to(output_device)
            llm.model.lm_head.to(output_device)

        elif "llama" in args.net.lower() or "mixtral" in args.net.lower() or "qwen" in args.net.lower():
            map_layers_to_multi_gpus(llm.model.model.layers)
            input_device = llm.model.model.layers[0].device
            output_device = llm.model.model.layers[-1].device
            assert input_device == output_device
            llm._device = input_device
            llm.model.model.embed_tokens.to(input_device)
            llm.model.model.norm.to(output_device)
            llm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(llm.model.transformer.h)
            input_device = llm.model.transformer.h[0].device
            output_device = llm.model.transformer.h[-1].device
            assert input_device == output_device
            llm._device = input_device
            llm.model.transformer.word_embeddings.to(input_device)
            llm.model.transformer.ln_f.to(output_device)
            llm.model.lm_head.to(output_device)
    else:
        if "opt" in args.net.lower():
            llm.model.model.decoder = llm.model.model.decoder.to(llm.device)
        elif "llama" in args.net.lower() or "mixtral" in args.net.lower() or 'qwen' in args.net.lower()  or 'minicpm' in args.net.lower():
            llm.model = llm.model.to(llm.device)
        elif "falcon" in args.net.lower():
            llm.model.transformer = llm.model.transformer.to(llm.device)
        elif "internvl" in args.model.lower() or ("onevision" in args.model.lower() and "llava" in args.model.lower()):
            # layers/embeds may sit on CPU after MAS replace; move whole model for eval
            llm.model = llm.model.to(llm.device)

    if args.eval_sqnr:
        from more_itertools import batched
        import json
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model)
        from qwen_omni_utils import process_mm_info
        USE_AUDIO_IN_VIDEO = False
        cur_index = 0
        sqnr_total = []
        csv_file_path = args.sqnr_result
        llm.model.eval()
        llm.model_origin.eval()
        with open(args.input_file) as f:
            for lines in batched(f, args.batch_size):
                datas = [json.loads(line) for line in lines]

                conversations = [data["prompt"] for data in datas]

                text = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                audios, images, videos = process_mm_info(
                    conversations, use_audio_in_video=False
                )

                inputs = processor(
                    text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                )
                inputs = inputs.to(llm.device).to(llm.model.dtype)
                
                llm.model_origin.to(llm.device)
                inputs['output_hidden_states'] = True
                output_quant = llm.model(**inputs)
                if "omni" in args.net.lower():
                    output_float = llm.model_origin.thinker(**inputs)
                else:
                    output_float = llm.model_origin.model(**inputs)
                
                all_hidden_states_float = output_float[2]
                all_hidden_states_quant = output_quant[2]
                # 对每个样本记录每一层的 SQNR
                layer_sqnr = [0] * (len(all_hidden_states_float) ) # 初始化每层 SQNR 列表
                
                for layer_index in range(1, len(all_hidden_states_float)):
                    sqnr = compute_sqnr_per_token(all_hidden_states_float[layer_index], all_hidden_states_quant[layer_index])
                    # sqnr = compute_sqnr_per_modality(all_hidden_states_float[layer_index], all_hidden_states_quant[layer_index], audio_mask, image_mask)
                    layer_sqnr[layer_index - 1] = sqnr  # 记录当前层 SQNR 
                
                # sqnr_final = compute_sqnr(output_float[0], output_quant[0])
                # layer_sqnr[layer_index] = sqnr_final
                sqnr_total.append(layer_sqnr)  # 保存当前样本的 SQNR 值
                # 把最终的logits 也计算一下 sqnr
                print(f'cur_index: {cur_index}, sqnr_layers: {layer_sqnr}')
                cur_index += 1
                
                if cur_index >= 32:  # 假设最多处理 32 个样本
                    break

        # 计算所有层的平均 SQNR
        mean_sqnr = [sum(layer)/cur_index for layer in zip(*sqnr_total)]

        # 将结果写入 CSV 文件
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        with open(csv_file_path, mode='w', newline='') as csv_file:
            # 设定表头，只包含每层的 SQNR 值
            fieldnames = [f'layer_{i}_sqnr' for i in range(1, len(mean_sqnr)+1)]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            writer.writeheader()  # 写入表头
            
            # 写入每个样本的 SQNR 值
            for layer_sqnr in sqnr_total:
                writer.writerow({f'layer_{i+1}_sqnr': sqnr for i, sqnr in enumerate(layer_sqnr)})
            
            # 写入平均值行
            writer.writerow({f'layer_{i+1}_sqnr': mean_sqnr[i] for i in range(len(mean_sqnr))})
            print({f'layer_{i+1}_sqnr': mean_sqnr[i] for i in range(len(mean_sqnr))})

        print(f'完成 SQNR 计算，结果已写入 {csv_file_path} \n')

    if args.eval_omni_task:
        from more_itertools import batched
        import json
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model)
        from qwen_omni_utils import process_mm_info
        USE_AUDIO_IN_VIDEO = False
        file_index = 0
        allocated = torch.cuda.memory_allocated(llm.device) / 1024**2
        print(f'allocated:  {allocated:.2f} MB')
        
        with open(args.input_file) as f, open(args.output_file, "w", encoding='utf-8') as fw:
            for lines in batched(f, args.batch_size):
                datas = [json.loads(line) for line in lines]
                file_index += 1

                # if file_index < 59:
                #     continue

                allocated = torch.cuda.memory_allocated(llm.device) / 1024**2
                # print(f'allocated_{file_index}:  {allocated:.2f} MB, before load data')

                conversations = [data["prompt"] for data in datas]

                text = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                audios, images, videos = process_mm_info(
                    conversations, use_audio_in_video=False
                )

                inputs = processor(
                    text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                )
                inputs = inputs.to(llm.device).to(llm.model.dtype)
                allocated = torch.cuda.memory_allocated(llm.device) / 1024**2
                # print(f'allocated_{file_index}:  {allocated:.2f} MB, load data done.')
                                
                text_ids = llm.model.generate(
                    **inputs,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    do_sample=False,
                    repetition_penalty=1.0,
                    max_new_tokens=128
                )
                allocated = torch.cuda.memory_allocated(llm.device) / 1024**2
                # print(f'allocated_{file_index}:  {allocated:.2f} MB, inference done')
                generated_ids_list = [
                    text_ids[i][len(inputs["input_ids"][i]) :] for i in range(len(datas))
                ]
                
                # generated_ids_list = text_ids[:, inputs.input_ids.size(1):]
                response_text = processor.batch_decode(
                    generated_ids_list,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for data, response in zip(datas, response_text):
                    json_data = json.dumps(data | {"response": response}, ensure_ascii=False)
                    fw.write(json_data + "\n")
                    fw.flush()

    if args.eval_ppl:
        results_ppl = {}
        csv_file_path = args.ppl_result
        # for dataset in ["wikitext2", "c4-new"]:
        for dataset in ["wikitext2"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader, weights_only=False)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=llm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // llm.seqlen
            
            
            if 'Qwen2.5-Omni' in args.model:
                use_cache = llm.model.config.text_config.use_cache
                llm.model.config.text_config.use_cache = False
            elif 'MiniCPM' in args.model:
                use_cache = llm.model.config.use_cache
                llm.model.config.use_cache = False            
            
            llm.model.eval()
            nlls = []
            # nsamples = 1
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * llm.seqlen) : ((i + 1) * llm.seqlen)].to(llm.device)
                if "opt" in args.net.lower():
                    outputs = llm.model.model.decoder(batch)
                elif "llama" in args.net.lower() or "mixtral" in args.net.lower() or 'qwen' in args.net.lower() or 'minicpm' in args.net.lower():
                    outputs = llm.model.model(batch)
                elif "falcon" in args.model:
                    outputs = llm.model.transformer(batch)

                hidden_states = outputs[0]
                if hidden_states.dtype != llm.model.lm_head.weight.dtype:
                    llm.model.lm_head.to(hidden_states.dtype)
                logits = llm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * llm.seqlen) : ((i + 1) * llm.seqlen)][
                    :, 1:
                ].to(llm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * llm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * llm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            if 'Qwen2.5-Omni' in args.model:
                llm.model.config.text_config.use_cache = use_cache
            elif 'MiniCPM' in args.model:
                llm.model.config.use_cache = use_cache
            
            results_ppl[dataset] = ppl.item()
        # 将结果写入 CSV 文件
        with open(csv_file_path, mode='w', newline='') as csv_file:
            # 设定表头，只包含每层的 SQNR 值
            fieldnames = results_ppl.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            writer.writeheader()  # 写入表头
            
            # 写入每个样本的 SQNR 值
            writer.writerow(results_ppl)

        print(f'完成 PPL 计算，结果已写入 {csv_file_path} \n')        
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            llm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit
        )
        results.update(t_results)
        logger.info(results)
        # for test of MMLU
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))
    
    if args.tasks_multimodal != "":
        model_key = args.model.lower()
        if 'Omni' in args.model:
            print(f'--------->>>>>>>>>>>>>>>>>>>> omni model !!!!!!!!!!!')
            from models.LMMClass_Omni import LMMClass
            vlm = LMMClass(args.model)
            vlm.model.thinker.model = llm.model.model
            t_results = eval_multimodal.simple_evaluate(
                vlm,
                tasks=args.tasks_multimodal.split(","),
                num_fewshot=args.num_fewshot,
                limit=None if args.limit_multimodal == 1.0 else args.limit_multimodal
            )
            results.update(t_results['results'])
            logger.info(results)
        elif "internvl" in model_key:
            from models.LMMClass_InternVL import LMMClass
            vlm = LMMClass(args.model, llm.model)
            t_results = eval_multimodal.simple_evaluate(
                vlm,
                tasks=args.tasks_multimodal.split(","),
                num_fewshot=args.num_fewshot,
                limit=None if args.limit_multimodal == 1.0 else args.limit_multimodal,
                gen_kwargs=f"max_new_tokens={args.gen_max_new_tokens}",
                cli_args=argparse.Namespace(
                    output_path=args.output_dir,
                    process_with_media=False,
                ),
            )
            results.update(t_results['results'])
            logger.info(results)
            print(f'tasks_multimodal:  {results}')
        elif ("llava-onevision" in model_key) or ("llava_onevision" in model_key) or (
            "onevision" in model_key and "llava" in model_key
        ):
            from models.LMMClass_LLaVAOV import LMMClass
            vlm = LMMClass(args.model, llm.model)
            t_results = eval_multimodal.simple_evaluate(
                vlm,
                tasks=args.tasks_multimodal.split(","),
                num_fewshot=args.num_fewshot,
                limit=None if args.limit_multimodal == 1.0 else args.limit_multimodal,
                gen_kwargs=f"max_new_tokens={args.gen_max_new_tokens}",
                cli_args=argparse.Namespace(
                    output_path=args.output_dir,
                    process_with_media=False,
                ),
            )
            results.update(t_results['results'])
            logger.info(results)
            print(f'tasks_multimodal:  {results}')
        else:
            from models.LMMClass import LMMClass
            vlm = LMMClass(args.model, llm.model)
            
            t_results = eval_multimodal.simple_evaluate(
                vlm,
                tasks=args.tasks_multimodal.split(","),
                num_fewshot=args.num_fewshot,
                limit=None if args.limit_multimodal == 1.0 else args.limit_multimodal,
                gen_kwargs=f"max_new_tokens={args.gen_max_new_tokens}",
                cli_args=argparse.Namespace(
                    output_path=args.output_dir,
                    process_with_media=False,
                ),
            )
            results.update(t_results['results'])
            logger.info(results)
            print(f'tasks_multimodal:  {results}')
    return results


def main_entry(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'infer'], help="training or inference mode")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--output_dir_postfix", type=str, default="", help="post fix for output dir")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--calib_dataset",type=str,default="omnibench",
        choices=["wikitext2", "ptb", "c4", "mix","pile", "omnibench"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--tasks_multimodal", default="")
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=16,
        help="max_new_tokens for multimodal lmms-eval generation (OCRBench typically needs 128)",
    )
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--auto_scale", action="store_true")
    parser.add_argument("--auto_alpha", action="store_true")
    parser.add_argument("--auto_epochs", action="store_true")
    parser.add_argument("--loss_multi_modal", action="store_true")
    parser.add_argument("--loss_multi_modal_mae", action="store_true")
    parser.add_argument("--loss_multi_modal_mae_alpha", action="store_true")
    parser.add_argument("--auto_modal_weight", action="store_true", help="learn layer-wise modality loss weights from calibration errors")
    parser.add_argument("--modal_weight_tau", type=float, default=0.05, help="temperature for auto modality loss weights")
    parser.add_argument("--adaptive_modal_scale", action="store_true", help="use token-adaptive soft mixing of text/vision/audio scales")
    parser.add_argument("--adaptive_scale_tau", type=float, default=1.0, help="temperature for token-adaptive scale routing")
    parser.add_argument("--mixed_precision", action="store_true", help="enable heuristic sensitivity-guided mixed precision for QuantLinear weights")
    parser.add_argument("--mp_high_bits", type=int, default=6, help="bitwidth for protected/sensitive modules under mixed precision")
    parser.add_argument("--mp_low_bits", type=int, default=None, help="bitwidth for non-protected modules under mixed precision; defaults to --wbits")
    parser.add_argument("--mp_protect_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj", help="comma-separated module names to protect with high bits")
    parser.add_argument("--mp_protect_layers", type=str, default="first,last", help="comma-separated protected layer ids or first,last")
    parser.add_argument("--quantize_vit", action="store_true", help="experimental: fake-quantize vision tower Linear layers for end-to-end MLLM quantization")
    parser.add_argument("--vit_wbits", type=int, default=8, help="vision tower weight bits when --quantize_vit is enabled")
    parser.add_argument("--vit_abits", type=int, default=8, help="vision tower activation bits when --quantize_vit is enabled")
    parser.add_argument("--ppl_result", default="ppl_result.csv")
    parser.add_argument("--eval_sqnr", action="store_true")
    parser.add_argument("--sqnr_result", default="sqnr_result.csv")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-2)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--disable_zero_point",default=False, action="store_true", help="quantization without zero_point")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--limit_multimodal", type=float, default=1.0)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    parser.add_argument("--input_file", default="")
    parser.add_argument("--output_file", default="")    
    parser.add_argument("--grad_info_path", default="")    
    parser.add_argument("--eval_omni_task", action="store_true")
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
        help="Data type to calculate activation. Options: text-only, vision-only, audio-only, text-vision, text-audio, vision-audio"
    )
    inference_mode = os.getenv('inference_mode', 'merged_scales')
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    os.environ["MASQ_ADAPTIVE_MODAL_SCALE"] = "1" if args.adaptive_modal_scale else "0"
    os.environ["MASQ_ADAPTIVE_SCALE_TAU"] = str(args.adaptive_scale_tau)
  
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    grad_info = None
    if len(args.grad_info_path) > 0:
        grad_info = torch.load(args.grad_info_path, weights_only=False)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=4) or (args.abits<16 and args.abits>=4):
        args.deactive_amp = True

    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]

    # init logger
    if args.output_dir:
        from datetime import datetime
        current_time = datetime.now()
        formatted_with_ms = current_time.strftime("%m%d-%H%M%S.%f")

        # args.output_dir = f'{args.output_dir}/{args.net}-{args.dataset_type}-{args.epochs}epochs-w{args.wbits}a{args.abits}-{args.output_dir_postfix}-{formatted_with_ms}'
        args.output_dir = f'{args.output_dir}/{args.net}-{args.epochs}epochs-w{args.wbits}a{args.abits}-{args.output_dir_postfix}-{formatted_with_ms}-{inference_mode}'
        print(f'->>>>> output_dir is: {args.output_dir}/mas_parameters.pth ')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    logger.info(f'inference_mode:  {inference_mode}')

    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    llm = LMClass(args)
    llm.seqlen = 2048
    llm.model.eval()
    for param in llm.model.parameters():
        param.requires_grad = False

    

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": True,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        llm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}-{args.dataset_type}-{args.nsamples}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}-{args.dataset_type}.pt'

    quantize_vision_tower_if_requested(llm.model, args, logger)

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.net}_{args.dataset_type}_{args.nsamples}.cache'
        print(f'try to load cache from: {cache_dataloader}')
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
                processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
                is_minicpm = 'MiniCPM' in args.model
                dataloader = prepare_dataset_before_quant(processor, calibration_dataset, batch_size=args.batch_size, is_minicpm=is_minicpm)
            elif "internvl" in model_key:
                from custom_dataset import prepare_calib_internvl
                dataloader = prepare_calib_internvl(llm, n_sample=args.nsamples)
            elif ("llava-onevision" in model_key) or ("llava_onevision" in model_key) or (
                "onevision" in model_key and "llava" in model_key
            ):
                from custom_dataset import prepare_calib_llava_onevision
                dataloader = prepare_calib_llava_onevision(llm, n_sample=args.nsamples)
            else:
                dataloader, _ = get_loaders(
                    args.calib_dataset,
                    nsamples=args.nsamples,
                    seed=args.seed,
                    model=args.model,
                    seqlen=llm.seqlen,
                )
            torch.save(dataloader, cache_dataloader)
            
        act_scales = None
        if args.let and (args.resume is None):
            act_scales = torch.load(args.act_scales)
        masquant(
            llm,
            args,
            dataloader,
            act_scales,
            logger,
            grad_info
        )
        logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in llm.model.named_modules():
            if isinstance(module, QuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer) or isinstance(module,QuantLlamaDecoderLayerV2) :
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        llm.model.save_pretrained(args.save_dir)  
        llm.tokenizer.save_pretrained(args.save_dir) 
    evaluate(llm, args,logger)


if __name__ == "__main__":
    print(sys.argv)
    main_entry()
