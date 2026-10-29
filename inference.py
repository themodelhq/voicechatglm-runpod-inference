"""Lazy model loader using the previously tested 4-bit NF4 Transformers path."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger("voicechatglm-inference")
_lock = threading.Lock()
_engine: "InferenceEngine | None" = None


@dataclass
class InferenceEngine:
    model: Any
    tokenizer: Any
    model_id: str
    revision: str


def get_engine(model_id: str, revision: str) -> InferenceEngine:
    global _engine
    if _engine is not None:
        if _engine.model_id != model_id or _engine.revision != revision:
            raise RuntimeError("This worker is already loaded with a different model/revision")
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this deployment configuration")
        log.info("Loading model %s at revision %s", model_id, revision)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=os.getenv("BNB_4BIT_QUANT_TYPE", "nf4"),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() == "true",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            token=os.getenv("HF_TOKEN") or None,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            token=os.getenv("HF_TOKEN") or None,
        )
        model.eval()
        _engine = InferenceEngine(model=model, tokenizer=tokenizer, model_id=model_id, revision=revision)
        log.info("Model loaded; CUDA device=%s", torch.cuda.get_device_name(0))
        return _engine


def generate_reply(*, engine: InferenceEngine, message: str, system_prompt: str | None, max_new_tokens: int) -> str:
    system = system_prompt.strip() if isinstance(system_prompt, str) and system_prompt.strip() else (
        "You are VoiceChatGLM, a helpful AI assistant. Answer the user's request clearly and accurately."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
    tokenizer = engine.tokenizer
    # This is the chat-template setting used in the prior tested Qwen3 path.
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    if isinstance(input_ids, dict):
        model_inputs = {k: (v.to(engine.model.device) if hasattr(v, "to") else v) for k, v in input_ids.items()}
    else:
        model_inputs = {"input_ids": input_ids.to(engine.model.device)}
    input_length = model_inputs["input_ids"].shape[-1]
    temperature = float(os.getenv("GENERATION_TEMPERATURE", "0.65"))
    top_p = float(os.getenv("GENERATION_TOP_P", "0.9"))
    with torch.inference_mode():
        output = engine.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, input_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
