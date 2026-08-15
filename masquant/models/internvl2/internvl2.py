# coding=utf-8
"""InternVL2 family wrapper (aligned with asdq.models.internvl2)."""

from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
from PIL import Image

from models.base import BaseModel
from models.registry import MODEL_REGISTRY
from .constants import IMG_CONTEXT_TOKEN
from .dataset import (
    build_transform,
    dynamic_preprocess,
    preprocess,
    preprocess_internlm,
    preprocess_mpt,
    preprocess_phi3,
)


@MODEL_REGISTRY.register("internvl2")
class InternVL2(BaseModel):
    def __init__(self, model, tokenizer, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.template_name = "internlm2-chat"
        self.num_image_token = self.model.num_image_token
        self.image_size = 448
        self.pad2square = False
        self.dynamic_image_size = True
        self.use_thumbnail = True
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 1
        self.normalize_type = "imagenet"
        self.group_by_length = True

    def fetch_vit(self):
        return self.model.vision_model

    def fetch_llm(self):
        return self.model.language_model

    def fetch_proj(self):
        return self.model.mlp1

    def get_blocks(self):
        return self.model.language_model.model.layers

    def get_preprocess_function(self):
        if self.template_name == "Hermes-2":
            return preprocess_mpt
        if self.template_name == "internlm2-chat":
            return preprocess_internlm
        if self.template_name == "phi3-chat":
            return preprocess_phi3
        return preprocess

    def get_transform(self):
        return build_transform(
            is_train=False,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )

    def vision_preprocess(self, image):
        transform = self.get_transform()
        if len(image) == 1:
            img = image[0]
            if self.dynamic_image_size:
                images = dynamic_preprocess(
                    img,
                    min_num=self.min_dynamic_patch,
                    max_num=self.max_dynamic_patch,
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                num_tiles = [len(images)]
            else:
                images = [img]
                num_tiles = [1]
            pixel_values = torch.stack([transform(i) for i in images])
            num_patches = pixel_values.size(0)
        else:
            images, num_tiles = [], []
            for img in image:
                if self.dynamic_image_size:
                    tiles = dynamic_preprocess(
                        img,
                        min_num=self.min_dynamic_patch,
                        max_num=max(1, self.max_dynamic_patch // len(image)),
                        image_size=self.image_size,
                        use_thumbnail=self.use_thumbnail,
                    )
                    images += tiles
                    num_tiles.append(len(tiles))
                else:
                    images.append(img)
                    num_tiles.append(1)
            pixel_values = torch.stack([transform(i) for i in images])
            num_patches = pixel_values.size(0)
        return pixel_values, num_patches, num_tiles

    def language_preprocess(self, text):
        return self.tokenizer(text)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        lm = self.fetch_llm()
        device = next(lm.parameters()).device
        if inputs_embeds is not None:
            return lm(
                inputs_embeds=inputs_embeds.to(device),
                attention_mask=attention_mask.to(device) if attention_mask is not None else None,
                use_cache=use_cache,
                return_dict=return_dict,
            )
        return lm(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            use_cache=use_cache,
            return_dict=return_dict,
        )

    def preprocess_data(self, images, data_item):
        if images is not None:
            pixel_values, num_patches, num_tiles = self.vision_preprocess(images)
            preprocess_function = self.get_preprocess_function()
            if "<image>" not in data_item["conversations"][0]["value"]:
                data_item = deepcopy(data_item)
                data_item["conversations"][0]["value"] = (
                    "<image>\n" + data_item["conversations"][0]["value"]
                )
            ret = preprocess_function(
                self.template_name,
                [deepcopy(data_item["conversations"])],
                self.tokenizer,
                [self.num_image_token * num_patches],
                group_by_length=self.group_by_length,
                ds_name="sharegpt4v",
            )
            return dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                attention_mask=ret["attention_mask"][0],
                pixel_values=pixel_values,
                image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            )
        image = Image.new("RGB", (224, 224), (255, 255, 255))
        pixel_values, num_patches, _ = self.vision_preprocess([image])
        preprocess_function = self.get_preprocess_function()
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            text_only=True,
            group_by_length=self.group_by_length,
            ds_name="sharegpt4v",
        )
        return dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
        )

    @torch.no_grad()
    def generate_input(self, data_samples) -> Tuple[Dict, Dict]:
        viz_dev = next(self.model.vision_model.parameters()).device
        input_ids = data_samples["input_ids"].to(viz_dev)
        attention_mask = data_samples["attention_mask"].to(viz_dev)
        labels = data_samples["labels"].to(viz_dev)
        pixel_values = data_samples["pixel_values"].to(dtype=self.model.dtype, device=viz_dev)
        image_flags = data_samples["image_flags"].to(viz_dev).squeeze(-1)

        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        bsz, seq_len, dim = input_embeds.shape
        flat_embeds = input_embeds.reshape(bsz * seq_len, dim)
        flat_ids = input_ids.reshape(bsz * seq_len)
        selected = flat_ids == self.model.img_context_token_id
        try:
            flat_embeds[selected] = vit_embeds.reshape(-1, dim)
        except Exception:
            vit_flat = vit_embeds.reshape(-1, dim)
            flat_embeds[selected] = vit_flat[: int(selected.sum())]
        input_embeds = flat_embeds.reshape(bsz, seq_len, dim)

        vision_mask = selected.reshape(bsz, seq_len)
        answer_mask = labels != -100
        return (
            {
                "inputs_embeds": input_embeds,
                "labels": labels,
                "attention_mask": attention_mask,
            },
            {"vision_mask": vision_mask, "caption_mask": answer_mask},
        )

    def to_cuda(self):
        self.model = self.model.cuda()

    def to_cpu(self):
        self.model = self.model.cpu()
