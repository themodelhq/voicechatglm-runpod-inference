# VoiceChatGLM RunPod Serverless inference worker

This package is a starting deployment artifact for the already-tested VoiceChatGLM Qwen3-8B model, using 4-bit NF4 quantization and the Qwen chat template. It is separate from `VoiceChatGLM_Frontend_v1.3.0.zip` and does not modify Render or the frontend.

## Important status

- Source/package scaffold prepared; no GPU build or live RunPod deployment was performed here.
- The previous Colab/T4 test does not guarantee this exact container image, dependency set, model revision, or RunPod hardware combination will work unchanged.
- Confirm the base image tag and build on RunPod before launching a paid worker.
- This is a **RunPod Serverless handler**, not a FastAPI HTTP server. RunPod invokes `handler(job)` using its serverless job protocol.
- The output shape is `{ok, model, conversation_id, message, usage}` on success and `{error: {code, message}}` on controlled errors.

## Model and runtime

Defaults preserve the repository and revision from the prior tested configuration:

- `VOICECHATGLM_MODEL_ID=themodelhq/voicechatglm-qwen3-8b`
- `VOICECHATGLM_MODEL_REVISION=9f25d05035bfda0de9b36dfc5911ddcdb378f3c8`
- `load_in_4bit=true`, NF4, FP16 compute, double quantization enabled
- `enable_thinking=False` in the tokenizer chat template

If the repository is private/gated, set `HF_TOKEN` as a secret in the RunPod endpoint environment. Use a read-only Hugging Face token. Do not bake credentials into the Docker image.

## Build and deploy

1. Push this directory to a private GitHub repository, or build and push its Docker image to a container registry supported by RunPod.
2. Select **Deploy your own production API → Use your own Repository** in RunPod.
3. Configure the endpoint to use this repository/image and the included Dockerfile.
4. Select a GPU with sufficient VRAM. The previously tested T4 had 14.56 GiB visible memory and loaded the quantized model at about 5.55 GiB model memory; actual inference needs extra memory for KV cache and context. Do not assume a GPU tier will fit without testing.
5. Add environment variables from `.env.example` using the RunPod dashboard's secret fields. Do not commit `.env`.
6. Start with a small test request and conservative output length. Check logs and endpoint costs before further use.

## RunPod Serverless request shape

RunPod calls the worker with a job whose `input` is this JSON object:

```json
{
  "message": "Say hello in one sentence.",
  "max_new_tokens": 96,
  "conversation_id": "optional-id",
  "system_prompt": "optional system instructions",
  "service_key": "only-required-if-VOICECHATGLM_API_KEY-is-configured"
}
```

RunPod's own HTTP API access token must be sent as `Authorization: Bearer <RUNPOD_API_KEY>` when calling the endpoint. If `VOICECHATGLM_API_KEY` is configured, `input.service_key` is an additional secret checked by the worker. The worker never returns that key.

## Compatibility with the existing Render backend

The historical VoiceChatGLM model adapter contract is documented as `POST /v1/chat` with an `X-API-Key` header and a JSON body containing `message`, optional `conversation_id`, `max_new_tokens`, and optional `system_prompt`.

A RunPod Serverless endpoint instead expects the RunPod bearer credential and wraps job data inside `{"input": {...}}`. Therefore this worker is **not a drop-in URL replacement** for the current Render adapter. Before connecting it, update/verify the Render model adapter to:

- send `Authorization: Bearer <RUNPOD_API_KEY>` to the RunPod endpoint URL;
- wrap the chat fields under `input`;
- optionally include `input.service_key` if the worker-level `VOICECHATGLM_API_KEY` is configured;
- unwrap RunPod's response envelope and map `output.message` (exact envelope can vary by RunPod endpoint mode) to the response shape expected by the application;
- never expose either credential to the browser or logs.

Do not set Render's `VOICECHATGLM_API_URL` to the Hugging Face model page URL. Use the actual RunPod Serverless endpoint URL after deployment. Do not change production Render variables until staging tests pass.

## Readiness and security

RunPod Serverless handles the worker lifecycle; this package does not expose public `/health` or `/ready` routes. Check worker status and logs in RunPod. The historical FastAPI service exposed `/health`, `/ready`, and `/v1/chat`; if preserving those exact HTTP paths without changing Render's adapter is mandatory, deploy a separate authenticated HTTP service on a RunPod Pod or add a carefully secured gateway. Do not expose an unauthenticated public model server.

Security notes:

- Use RunPod endpoint access controls and keep its API key only on Render's server side.
- If `VOICECHATGLM_API_KEY` is set, configure the same strong secret in Render and RunPod, and pass it only server-to-server in the job payload.
- Use a private repository if desired; HF_TOKEN must have read-only permissions.
- Never log request secrets, HF tokens, or raw authorization headers.
- `max_new_tokens` is validated against `MAX_ALLOWED_NEW_TOKENS`.
- This implementation serializes model loading per worker but does not provide conversation history persistence; `conversation_id` is passed through for correlation only.

## Validation checklist

- [ ] Docker image builds successfully with the selected RunPod base image.
- [ ] Worker starts and downloads the specified revision.
- [ ] GPU and bitsandbytes CUDA support are detected.
- [ ] Health/worker readiness is visible in RunPod dashboard.
- [ ] Valid request produces a response.
- [ ] Missing/invalid `service_key` fails when configured.
- [ ] Oversized/invalid `max_new_tokens` is rejected.
- [ ] Render adapter sends RunPod's required bearer auth and `input` envelope.
- [ ] Revoked/rotated RunPod API credentials are handled safely.
- [ ] Staging end-to-end test passes before changing production.

## Cost control

Building this package does not require launching a GPU. GPU time, endpoint idle/minimum billing, storage and image registry charges depend on the selected RunPod configuration. Check current prices and set provider-side spending alerts/limits where available before deploying.
