"""
Conversation prompt templates.
"""

import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple, Union


class SeparatorStyle(IntEnum):
    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    name: str
    system_template: str = "{system_message}"
    system_message: str = ""
    roles: Tuple[str] = ("USER", "ASSISTANT")
    messages: List[List[str]] = ()
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        if self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )


conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."
    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    return conv_templates[name].copy()


register_conv_template(
    Conversation(
        name="Hermes-2",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
        stop_token_ids=[2, 6, 7, 8],
        stop_str="<|endoftext|>",
    )
)

register_conv_template(
    Conversation(
        name="internlm2-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
        stop_token_ids=[2, 92543, 92542],
    )
)

register_conv_template(
    Conversation(
        name="phi3-chat",
        system_template="<|system|>\n{system_message}",
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|user|>\n", "<|assistant|>\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|end|>",
        stop_token_ids=[2, 32000, 32007],
    )
)

