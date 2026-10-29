"""RunPod Serverless handler for VoiceChatGLM Qwen3-8B (4-bit NF4)."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import runpod
from inference import get_engine, generate_reply

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("voicechatglm-runpod")

MODEL_ID = os.getenv("VOICECHATGLM_MODEL_ID", "themodelhq/voicechatglm-qwen3-8b")
MODEL_REVISION = os.getenv(
    "VOICECHATGLM_MODEL_REVISION",
    "9f25d05035bfda0de9b36dfc5911ddcdb378f3c8",
)
SERVICE_KEY = os.getenv("VOICECHATGLM_API_KEY", "")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1200"))


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod job handler. Expected job input: {message, ...}.

    If VOICECHATGLM_API_KEY is configured, require input.service_key to match it.
    RunPod's own endpoint Authorization token is still required at the HTTP edge.
    """
    started = time.monotonic()
    payload = job.get("input") or {}
    if not isinstance(payload, dict):
        return {"error": {"code": "invalid_input", "message": "input must be a JSON object"}}

    if SERVICE_KEY:
        supplied = payload.get("service_key")
        if not isinstance(supplied, str) or not _constant_time_equal(supplied, SERVICE_KEY):
            return {"error": {"code": "unauthorized", "message": "Invalid service credential"}}

    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return {"error": {"code": "invalid_input", "message": "message is required and must be a non-empty string"}}

    try:
        max_new_tokens = int(payload.get("max_new_tokens", MAX_NEW_TOKENS))
    except (TypeError, ValueError):
        return {"error": {"code": "invalid_input", "message": "max_new_tokens must be an integer"}}
    max_allowed = int(os.getenv("MAX_ALLOWED_NEW_TOKENS", "1200"))
    if max_new_tokens < 1 or max_new_tokens > max_allowed:
        return {"error": {"code": "invalid_input", "message": f"max_new_tokens must be between 1 and {max_allowed}"}}

    system_prompt = payload.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        return {"error": {"code": "invalid_input", "message": "system_prompt must be a string"}}

    try:
        engine = get_engine(MODEL_ID, MODEL_REVISION)
        answer = generate_reply(
            engine=engine,
            message=message.strip(),
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        return {
            "ok": True,
            "model": MODEL_ID,
            "conversation_id": payload.get("conversation_id"),
            "message": answer,
            "usage": {"elapsed_seconds": round(time.monotonic() - started, 3)},
        }
    except Exception as exc:  # Do not return stack traces or secret-bearing exception text.
        log.exception("Inference job failed")
        return {"error": {"code": "inference_failed", "message": "Model inference failed; inspect private worker logs."}}


def _constant_time_equal(left: str, right: str) -> bool:
    import hmac
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
