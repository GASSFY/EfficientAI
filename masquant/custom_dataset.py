
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

