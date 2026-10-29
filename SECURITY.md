# Security notes

- Never commit `.env`, HF tokens, RunPod API keys, or `VOICECHATGLM_API_KEY`.
- Store secrets in RunPod endpoint environment settings and Render secret environment settings.
- Restrict Hugging Face token to read-only access to the model repository.
- The RunPod Serverless endpoint's `Authorization: Bearer` credential protects the public RunPod API. The optional worker-level `service_key` is a second layer, not a replacement for RunPod authentication.
- Do not expose raw exception strings to clients; logs may contain stack traces, so keep logs private and do not log request bodies.
- Rotate credentials if they are ever committed, copied into chat, or exposed in logs.
