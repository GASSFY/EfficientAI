# coding=utf-8
"""lmms-eval wrapper for InternVL2 family."""

from typing import List, Tuple

import torch
import torchvision.transforms as T
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from models.internvl2.constants import IMG_CONTEXT_TOKEN, IMAGENET_MEAN, IMAGENET_STD


def _patch_internlm_generate(model):
    """transformers>=4.50: InternLM2ForCausalLM may miss GenerationMixin.generate."""
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin

    lm = model.language_model
    cls = lm.__class__
    if not issubclass(cls, GenerationMixin):
        lm.__class__ = type(cls.__name__, (cls, GenerationMixin), {})
    if getattr(lm, "generation_config", None) is None:
        try:
            lm.generation_config = GenerationConfig.from_model_config(lm.config)
        except Exception:
            lm.generation_config = GenerationConfig()


def build_transform(input_size=448):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=True):
    # simplified single-tile path for eval smoke; matches ASDQ max_dynamic_patch=1 default
    images = [image.resize((image_size, image_size))]
    if use_thumbnail and max_num > 1:
        images.append(image.resize((image_size, image_size)))
    return images[:1]


class LMMClass(lmms):
    def __init__(self, model_path, quant_model=None):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_path
        self.batch_size_per_gpu = 1
        self.use_cache = True
        self.image_size = 448

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        if quant_model is None:
            self._model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval().cuda()
        else:
            self._model = quant_model.to(self._device)
        self.model = self._model
        self._config = self._model.config
        _patch_internlm_generate(self.model)
        self.model.eval()
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.vocab_size = self.tokenizer.vocab_size
        self.transform = build_transform(self.image_size)
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return getattr(self._config, "max_position_embeddings", 4096)

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for InternVL2")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for InternVL2")

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
            gen_kwargs = all_gen_kwargs[0]
            context = contexts[0]
            if "<image>" in context:
                context = context.replace("<image>", "").strip()

            pixel_values = None
            if len(visuals) > 0 and isinstance(visuals[0], Image.Image):
                tiles = dynamic_preprocess(visuals[0], image_size=self.image_size)
                vision_device = next(self.model.vision_model.parameters()).device
                vision_dtype = next(self.model.vision_model.parameters()).dtype
                pixel_values = torch.stack([self.transform(t) for t in tiles]).to(
                    dtype=vision_dtype, device=vision_device
                )

            generation_config = dict(
                max_new_tokens=gen_kwargs.get("max_new_tokens", 128),
                do_sample=bool(gen_kwargs.get("temperature", 0) and gen_kwargs.get("temperature", 0) > 0),
                num_beams=gen_kwargs.get("num_beams", 1),
            )
            if generation_config["do_sample"]:
                generation_config["temperature"] = gen_kwargs.get("temperature", 0.0)
                if gen_kwargs.get("top_p", None) is not None:
                    generation_config["top_p"] = gen_kwargs["top_p"]
            # InternVL chat API
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                context,
                generation_config,
            )
            if isinstance(response, tuple):
                response = response[0]
            res.append(str(response).strip())
            self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), res[-1])
            pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res
