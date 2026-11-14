"""Loads the local Qwen3-8B chat model in 4-bit (bitsandbytes NF4) for the explainer."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"


@dataclass
class LoadedLLM:
    model: object
    tokenizer: object
    model_id: str


def load_llm(model_id: str = DEFAULT_MODEL_ID, device_map=None) -> LoadedLLM:
    # pinned to cuda:0: device_map="auto" underestimates free memory on an 8 GB GPU and offloads layers
    # to CPU, which bitsandbytes 4-bit does not support without an fp32-cpu-offload flag
    if device_map is None:
        device_map = {"": 0}
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # a base (pretrain-only) model has no chat template and would silently produce unstructured text
    if tokenizer.chat_template is None:
        raise ValueError(
            f"{model_id} has no chat_template -- this looks like a base/pretrain-only "
            f"checkpoint, not an instruct/chat one. Qwen3's naming convention doesn't "
            f"always suffix chat checkpoints with '-Instruct' the way Qwen2.5 did; "
            f"check the model card on the Hub for the correct chat-tuned repo id."
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map=device_map,
    )
    model.eval()
    return LoadedLLM(model=model, tokenizer=tokenizer, model_id=model_id)
