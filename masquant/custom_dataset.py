
from datasets import load_dataset
from typing import Dict
from qwen_omni_utils import process_mm_info
import json

def prepare_dataset(n_sample: int = 8, data_type: str = 'text-vision') -> list[list[dict]]:
    from datasets import load_dataset

    if data_type == 'text-only':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=f"train[:{n_sample}]")
        return [
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },            
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": sample['text']},
                    ],
                }
            ]
            for sample in dataset
        ]
    elif data_type == 'audio-only':
        dataset_json = '/nas/yuehu/NEW/qwen_compressor/dataset/libri_test_other.jsonl'
        prefix_path = "file:///nas/yuehu/assets/omni_data"
        conversations = []
        with open(dataset_json, "r") as json_file:
            lines = json_file.readlines()
            for line in lines:
                dataset = json.loads(line)
                prompt = dataset["prompt"]
                for item in prompt:
                    if item["role"] == "user":
                        item["content"] = [entry for entry in item["content"] if entry["type"] != "text"]
                
                conversations.append(prompt)
        return conversations[:n_sample]
    elif data_type == 'vision-only':
        dataset_json = '/nas/yuehu/NEW/qwen_compressor/dataset/sharegpt4v_instruct_gpt4-vision_cap100k_filtered_coco_image.json'
        with open(dataset_json, "r") as json_file:
            dataset = json.load(json_file)
            
        prefix_path = "file:///nas/yuehu/assets/dataset/"

        dataset_ret = []
        for i in range(n_sample):
            data_item = dataset[i]

            conversations = data_item["conversations"]
            for conv in conversations:
                if conv["from"] == "human":
                    user_text = conv["value"]
                    if "<image>" in user_text:
                        user_text = user_text.replace("<image>", "")
                    if "\n" in user_text:
                        user_text = user_text.replace("\n", "")
                if conv["from"] == "gpt":
                    asst_text = conv["value"]
            image_path = data_item["image"]
            item = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": prefix_path + image_path}
                    ]
                }
            ]
            dataset_ret.append(item)

        return dataset_ret
    elif data_type == 'text-audio':
        dataset_json = '/nas/yuehu/NEW/qwen_compressor/dataset/libri_test_other.jsonl'
        prefix_path = "file:///nas/yuehu/assets/omni_data"
        conversations = []
        with open(dataset_json, "r") as json_file:
            lines = json_file.readlines()
            for line in lines:
                dataset = json.loads(line)
                conversations.append(dataset["prompt"])
        return conversations[:n_sample]
    elif data_type == 'text-vision':
        dataset_json = '/root/autodl-tmp/hf_home/datasets/coco/sharegpt4v_coco_only.json'
        with open(dataset_json, "r") as json_file:
            dataset = json.load(json_file)
            
        prefix_path = "/root/autodl-tmp/hf_home/datasets/"

        dataset_ret = []
        for i in range(n_sample):
            data_item = dataset[i]

            conversations = data_item["conversations"]
            for conv in conversations:
                if conv["from"] == "human":
                    user_text = conv["value"]
                    if "<image>" in user_text:
                        user_text = user_text.replace("<image>", "")
                    if "\n" in user_text:
                        user_text = user_text.replace("\n", "")
                if conv["from"] == "gpt":
                    asst_text = conv["value"]
            image_path = data_item["image"]
            item = [
                # {
                #     "role": "system",
                #     "content": [
                #         {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                #     ],
                # },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": prefix_path + image_path},
                        {"type": "text", "text": user_text}
                    ],
                }
            ]
            dataset_ret.append(item)

        return dataset_ret
    elif data_type == 'text-audio-vision' or data_type == 'mas_mix_dataset' :
        dataset_json = 'data/jsonls/omnibench.jsonl'
        if data_type == 'mas_mix_dataset':
            dataset_json = 'data/jsonls/mas_mix_dataset.jsonl'        
        conversations = []
        with open(dataset_json, "r") as json_file:
            lines = json_file.readlines()
            for line in lines:
                dataset = json.loads(line)
                conversations.append(dataset["prompt"])
        return conversations[:n_sample]        
    else:
        print(f'data_type: {data_type} is not supported yet.')
        return []


def batched(iterable, n: int, process_func):
    # batched('ABCDEFG', 3) → ABC DEF G
    assert n >= 1, "batch size must be at least one"
    from itertools import islice
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if process_func is None:
            yield batch
        else:
            yield [process_func(item) for item in batch]

def preprocess_dataset(sample: Dict) -> Dict:
    return sample

def prepare_dataset_before_quant(processor, calibration_dataset, batch_size: int = 1, is_qwen_vl: bool = False, is_minicpm: bool = False):
    import torch
    from PIL import Image
    import requests
    from io import BytesIO
    
    calib_data = []
    for batch in batched(calibration_dataset, batch_size, process_func=preprocess_dataset):
        if is_minicpm:
            # For MiniCPM-V, we need both image and text to compute proper activation scales
            # We'll process the full multimodal input
            try:
                inputs = processor(batch, return_tensors="pt", max_slice_nums=9)
                calib_data.append(inputs)
            except Exception as e:
                print(f"Error processing MiniCPM input: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            text = processor.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
            if is_qwen_vl == False:
                audios, images, videos = process_mm_info(batch, use_audio_in_video=False)
                inputs = processor(text=text, images=images, videos=videos, audio=audios, padding=True, return_tensors="pt")
            else:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(batch)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            calib_data.append(inputs)
    return calib_data


def _load_sharegpt_items(n_sample: int):
    import json
    from PIL import Image

    dataset_json = "/root/autodl-tmp/hf_home/datasets/coco/sharegpt4v_coco_only.json"
    prefix_path = "/root/autodl-tmp/hf_home/datasets/"
    with open(dataset_json, "r") as f:
        dataset = json.load(f)
    items = []
    for i in range(min(n_sample, len(dataset))):
        data_item = dataset[i]
        image = Image.open(prefix_path + data_item["image"]).convert("RGB")
        items.append((image, data_item))
    return items


def prepare_calib_internvl(llm, n_sample: int = 8):
    """Build InternVL2 calibration batches for MAS Catcher / act-scale collection."""
    import torch
    from models.process_models import get_process_model

    wrapper = get_process_model("internvl2")(llm.model, llm.tokenizer)
    # Vision tower weights are bf16/fp16; transform() yields float32 by default.
    pix_dtype = getattr(llm.model, "dtype", None) or torch.bfloat16
    calib_data = []
    for image, data_item in _load_sharegpt_items(n_sample):
        sample = wrapper.preprocess_data([image], data_item)
        batch = {
            "input_ids": sample["input_ids"].unsqueeze(0),
            "attention_mask": sample["attention_mask"].unsqueeze(0),
            "pixel_values": sample["pixel_values"].to(dtype=pix_dtype),
            "image_flags": sample["image_flags"],
        }
        calib_data.append(batch)
    return calib_data


def prepare_calib_llava_onevision(llm, n_sample: int = 8):
    """Build LLaVA-OneVision calibration batches."""
    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token

    image_processor = getattr(llm, "image_processor", None)
    if image_processor is None:
        image_processor = llm.model.get_vision_tower().image_processor
    pad_id = (
        llm.tokenizer.pad_token_id
        if llm.tokenizer.pad_token_id is not None
        else llm.tokenizer.eos_token_id
    )
    calib_data = []
    for image, data_item in _load_sharegpt_items(n_sample):
        user_text = ""
        for conv in data_item["conversations"]:
            if conv["from"] == "human":
                user_text = conv["value"].replace("<image>", "").strip()
                break
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + user_text
        input_ids = tokenizer_image_token(
            prompt, llm.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)
        image_tensor = process_images([image], image_processor, llm.model.config)
        if isinstance(image_tensor, list):
            images = [t.to(dtype=torch.float16) for t in image_tensor]
        else:
            images = image_tensor.to(dtype=torch.float16)
        calib_data.append(
            {
                "input_ids": input_ids,
                "attention_mask": input_ids.ne(pad_id),
                "images": images,
                "image_sizes": [image.size],
                "modalities": ["image"],
            }
        )
    return calib_data

