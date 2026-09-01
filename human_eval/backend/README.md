# Human Evaluation Backend

The backend deliberately stays small: FastAPI, one JSON file per session, separate WAV files, and one blinded WebSocket proxy to the existing `demo_app.py` Voice service. There is no database, ORM, queue, or separate model adapter process.

Run instructions, model routing, allocation policy, collected metrics, API contracts, and the multi-turn open question are documented in the main [Human Evaluation README](../README.md).

Optional environment variables:

- `HUMAN_EVAL_DATA_DIR` — JSON/audio root; defaults to `human_eval/backend/data`.
- `HUMAN_EVAL_CONVERSATION_SECONDS` — server limit; defaults to `120`.
- `MINICPM_DEMO_URL`, `MINICPM_DEMO_TOKEN` — upstream Voice service override.
- `OPENAI_API_KEY` — post-hoc transcription for turns without upstream text.
- `HUMAN_EVAL_ASR_MODEL` — ASR model; defaults to `gpt-transcribe`.
- `HUMAN_EVAL_REQUIRE_TRANSCRIPTS` — set to `1` for formal/public collection; readiness fails when ASR is not configured.

For a public deployment, mount persistent storage at `/data`, set `HUMAN_EVAL_DATA_DIR=/data`, and run one backend replica. The browser does not retain study records; JSON and WAV files live on that server-side persistent disk.

See [DATA_SCHEMA.md](./DATA_SCHEMA.md) for the persisted record shape.
