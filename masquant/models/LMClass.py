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

import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModel
)
class LMClass(BaseLM):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model
        model_key = args.model.lower()
        is_llava_ov = ("llava-onevision" in model_key) or ("llava_onevision" in model_key) or (
            "onevision" in model_key and "llava" in model_key
        )
        is_internvl = "internvl" in model_key

        if is_llava_ov:
            # LLaVA-NeXT builder owns config/tokenizer loading
            config = None
            self.tokenizer = None
        else:
            config = AutoConfig.from_pretrained(
                args.model, attn_implementation=args.attn_implementation, trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.model, use_fast=False, legacy=False, trust_remote_code=True
            )
        # 注意，我们只使用 omni 的 thinker 模块
        if 'Qwen2.5-Omni' in args.model:
            # from transformers import Qwen2_5OmniForConditionalGeneration
            from models.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
            import copy
            kwargs = {"device_map": 'auto', 'enable_audio_output': False, 'attn_implementation': 'flash_attention_2', 'torch_dtype': torch.bfloat16}
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model, **kwargs)
            # import pdb;pdb.set_trace()
            # model.half()
            if args.eval_sqnr:
                self.model_origin = copy.deepcopy(model)
            else:
                self.model_origin = model
            self.model_config_origin = model.config
            
            self.model = model.thinker
            print(f'we will use omni-thinker model only.')
        elif 'Qwen2.5-VL' in args.model:
            from models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            import copy
            kwargs = {"torch_dtype": torch.bfloat16, "device_map": 'auto', "trust_remote_code": True, 'attn_implementation': 'flash_attention_2'}
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
            if args.eval_sqnr:
                self.model_origin = copy.deepcopy(model)
            else:
                self.model_origin = model
            self.model_config_origin = model.config
            
            self.model = model
        elif 'MiniCPM' in args.model:
            import copy
            kwargs = {"torch_dtype": 'auto', "device_map": 'auto', "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
            model = AutoModel.from_pretrained(args.model, **kwargs)
            self.model_origin = copy.deepcopy(model)
            self.model_config_origin = model.config            
            self.model = model.llm
            processor = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            print(f'we will use minicpm llm model only.')
        elif is_internvl:
            import copy
            kwargs = {
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            model = AutoModel.from_pretrained(args.model, **kwargs)
            model = model.eval().cuda()
            from models.LMMClass_InternVL import _patch_internlm_generate
            _patch_internlm_generate(model)
            if getattr(args, "eval_sqnr", False):
                self.model_origin = copy.deepcopy(model)
            else:
                self.model_origin = model
            self.model_config_origin = model.config
            self.model = model  # full InternVLChatModel
            self.seqlen = getattr(model.language_model.config, "max_position_embeddings", 4096)
            print("loaded InternVL2 full model for MAS-Quant")
        elif is_llava_ov:
            import copy
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model

            model_name = get_model_name_from_path(args.model)
            # Avoid accelerate device_map hooks (they break per-layer .cpu()/.to(cuda) in MAS).
            tokenizer, model, image_processor, _max_length = load_pretrained_model(
                args.model,
                None,
                model_name,
                device_map=None,
                multimodal=True,
                attn_implementation=getattr(args, "attn_implementation", None) or "sdpa",
            )
            model = model.to(self._device)
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            if getattr(args, "eval_sqnr", False):
                self.model_origin = copy.deepcopy(model)
            else:
                self.model_origin = model
            self.model_config_origin = model.config
            self.model = model  # full LlavaQwenForCausalLM
            self.seqlen = getattr(model.config, "tokenizer_model_max_length", None) or getattr(
                model.config, "max_position_embeddings", 4096
            )
            print("loaded LLaVA-OneVision full model for MAS-Quant")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)
            self.seqlen = self.model.config.max_position_embeddings
            
            self.model_origin = self.model
            self.model_config_origin = self.model.config            
        
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.seqlen

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )