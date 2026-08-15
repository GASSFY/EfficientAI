# coding=utf-8
"""Register process-model wrappers and look them up by name (ASDQ-style)."""

from models.internvl2 import InternVL2  # noqa: F401 - registers "internvl2"
from models.llava_onevision import LLaVA_onevision  # noqa: F401 - registers "llava_onevision"
from models.registry import MODEL_REGISTRY


def get_process_model(model_name: str):
    return MODEL_REGISTRY[model_name]
