# coding=utf-8
"""lmms-eval wrapper for LLaVA-OneVision family."""

import copy
from typing import List, Tuple

import torch
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


class LMMClass(lmms):
    def __init__(self, model_path, quant_model=None, conv_template="qwen_1_5"):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_path
        self.batch_size_per_gpu = 1
        self.use_cache = True
        self.conv_template = conv_template

        model_name = get_model_name_from_path(model_path)
        if quant_model is None:
            # Avoid accelerate device_map hooks (break later .cpu()/.to(cuda) in MAS).
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                model_path,
                None,
                model_name,
                device_map=None,
                multimodal=True,
                attn_implementation="sdpa",
            )
            self._model = model.to(self._device)
        else:
            # Reuse quantized weights; only need tokenizer / image_processor from builder.
            tokenizer, _unused, image_processor, max_length = load_pretrained_model(
                model_path,
                None,
                model_name,
                device_map="cpu",
                multimodal=True,
                attn_implementation="sdpa",
            )
            del _unused
            torch.cuda.empty_cache()
            self._model = quant_model.to(self._device)
        self._tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self._max_length = max_length
        self.model = self._model
        self._config = self._model.config
        self.model.eval()
        self.vocab_size = getattr(self.tokenizer, "vocab_size", None)
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for LLaVA-OneVision")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for LLaVA-OneVision")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            gen_kwargs = dict(all_gen_kwargs[0])
            context = contexts[0]

            image_tensor = None
            if len(visuals) > 0 and isinstance(visuals[0], Image.Image):
                image_tensor = process_images(visuals, self.image_processor, self._config)
                if isinstance(image_tensor, list):
                    image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device="cuda")
                if DEFAULT_IMAGE_TOKEN not in context:
                    context = DEFAULT_IMAGE_TOKEN + "\n" + context

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to("cuda")
            pad_token_ids = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            attention_masks = input_ids.ne(pad_token_ids)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if not gen_kwargs.get("do_sample", False):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)
            gen_kwargs.pop("until", None)
            gen_kwargs.pop("image_aspect_ratio", None)

            with torch.inference_mode():
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    image_sizes=[v.size for v in visuals] if visuals else None,
                    use_cache=self.use_cache,
                    **gen_kwargs,
                )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            ans = text_outputs[0].strip()
            res.append(ans)
            self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), ans)
            pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res
