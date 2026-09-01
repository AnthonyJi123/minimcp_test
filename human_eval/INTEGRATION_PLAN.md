# Human Evaluation Integration Plan

The local end-to-end path is implemented:

1. FastAPI assigns two tasks and two blinded model arms per task.
2. The browser streams 16 kHz PCM through the study WebSocket.
3. The gateway connects to the existing `demo_app.py` Voice WebSocket with the assigned `probe_on` setting.
4. Session JSON, WAV audio, transcripts, model telemetry, ratings, and comparisons are persisted by the backend.

Before formal recruitment:

- Resolve the multi-turn context question in [README.md](./README.md).
- Run a small pilot covering manual finish, two-minute timeout, model failure, refresh, and all task/arm orders.
- Decide consent copy, data retention, access rules, and transcript redaction.
- Add authentication to the admin export route and TLS/origin restrictions for deployment.
- Deploy one backend replica with a persistent disk and `HUMAN_EVAL_DATA_DIR=/data`; enable `HUMAN_EVAL_REQUIRE_TRANSCRIPTS=1`.
- Inspect exported sessions for missing audio/transcripts and confirm assignment balance.

Do not add a database or queue unless multiple backend processes or recruitment scale require them.
