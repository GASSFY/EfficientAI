import sys
from typing import Dict

import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from transformers.trainer_pt_utils import LabelSmoother
from torchvision.transforms.functional import InterpolationMode

from .constants import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    SIGLIP_MEAN,
    SIGLIP_STD,
)
from .conversation import get_conv_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def build_transform(is_train, input_size, pad2square=False, normalize_type="imagenet"):
    if normalize_type == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == "clip":
        mean, std = CLIP_MEAN, CLIP_STD
    elif normalize_type == "siglip":
        mean, std = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError

    ops = [T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img)]
    if pad2square:
        ops.append(T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in mean))))
    ops.extend(
        [
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return T.Compose(ops)


def preprocess(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1,
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}"
                conversation = conversation.replace("<image>", image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding=False if group_by_length or use_packed_ds else "max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i != 0 and not tokenizer.legacy:
                instruction_len -= 1
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len
            if i != 0 and not tokenizer.legacy:
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_TOKEN_ID
            print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. dataset={ds_name}")
            sys.stdout.flush()

    return dict(input_ids=input_ids, labels=targets, attention_mask=input_ids.ne(tokenizer.pad_token_id))


def preprocess_mpt(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1,
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}"
                conversation = conversation.replace("<image>", image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding=False if group_by_length or use_packed_ds else "max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx : conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_TOKEN_ID
        for turn in re_turns:
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids) + 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len
        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_TOKEN_ID
            print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. dataset={ds_name}")
            sys.stdout.flush()

    return dict(input_ids=input_ids, labels=targets, attention_mask=input_ids.ne(tokenizer.pad_token_id))


def preprocess_phi3(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1,
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}"
                conversation = conversation.replace("<image>", image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    tokenizer.padding_side = "right"
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding=False if group_by_length or use_packed_ds else "max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(int(tokenizer.pad_token_id)).sum())
        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx : conv_idx + 2]))
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        target[target == endoftext_id] = IGNORE_TOKEN_ID
        for i, turn in enumerate(re_turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids) if i == 0 else len(tokenizer(turn).input_ids) - 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids) - (1 if i == 0 else 2)
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len
        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_TOKEN_ID
            print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. dataset={ds_name}")
            sys.stdout.flush()
    return dict(input_ids=input_ids, labels=targets, attention_mask=input_ids.ne(tokenizer.pad_token_id))


def preprocess_internlm(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1,
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            sentence["value"] = sentence["value"].strip()
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}"
                conversation = conversation.replace("<image>", image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding=False if group_by_length or use_packed_ds else "max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        parts = conversation.split(conv.roles[1])
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1
        target[cur_len : cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len += temp_len
        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            cur_len += len(tokenizer(part1).input_ids) - 1
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len : cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len += temp_len
        cur_len += len(tokenizer(parts[-1]).input_ids) - 1
        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_TOKEN_ID
            print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. dataset={ds_name}")
            sys.stdout.flush()

    return dict(input_ids=input_ids, labels=targets, attention_mask=input_ids.ne(tokenizer.pad_token_id))


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

