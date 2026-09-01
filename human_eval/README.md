# MiniCPM Human Evaluation

English, blinded voice evaluation of MiniCPM and MiniCPM+. The participant UI and FastAPI backend are under `human_eval/`.

## Current status

- The browser checks the microphone, speaker, consent, and model readiness.
- Each participant receives two tasks. Each task contains one MiniCPM conversation and one MiniCPM+ conversation in blinded, balanced order.
- Each conversation lasts at most two minutes and uses one fixed model arm.
- The backend saves assignments, audio, transcripts, model telemetry, ratings, and pairwise feedback.
- Full transcripts work when `OPENAI_API_KEY` is set. Formal mode can require this with `HUMAN_EVAL_REQUIRE_TRANSCRIPTS=1`.
- Routing correctness is reviewed per MiniCPM+ turn; it is not inferred from the task type.

## Open questions before launch

1. **Multi-turn memory:** `demo_app.py` resets model state after every turn. Decide how to retain conversation history for S3 and pass that history to the expert.
2. **Transcript and audio policy:** confirm that both may be stored; define consent language, retention, deletion, and redaction of personal information.
3. **Production storage:** choose the host, persistent disk, backup policy, and data owner. The proposed small-study setup is one backend replica with `HUMAN_EVAL_DATA_DIR=/data`.
4. **Access control:** add authentication to `/api/admin/*`, restrict origins, and use HTTPS before sharing a public link.
5. **Routing review:** decide who labels each MiniCPM+ turn as expected `local`, `escalate`, or `not_applicable`, and whether a second reviewer is needed.
6. **Study policy:** set the recruitment target, stopping rule, exclusion criteria, and treatment of incomplete sessions.
7. **ASR quality:** pilot English transcription accuracy and decide how failed or incorrect transcripts are corrected.
8. **Provider logs:** confirm whether the model and ASR providers retain request audio or text.

## Run and test locally

One-time setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r human_eval/backend/requirements.txt
```

Start the app:

```bash
source .venv/bin/activate
python3 human_eval/serve.py
```

Open <http://127.0.0.1:4173/>.

For full transcripts:

```bash
export OPENAI_API_KEY='your-key'
export HUMAN_EVAL_REQUIRE_TRANSCRIPTS=1
python3 human_eval/serve.py
```

To start another test in the same browser:

1. Restart `serve.py` if `scenarios.json` or backend code changed. Scenario definitions are loaded when the server starts.
2. Open browser developer tools and run:

   ```js
   sessionStorage.removeItem("humanEvalUserId");
   location.reload();
   ```

3. Use a hard refresh (`Command+Shift+R`) if UI files changed. `scenarios.json` already uses `cache: "no-store"`.

Both the server restart and the cleared browser user ID are required to test a new scenario version. Existing sessions keep the scenario assigned when they were created.

Do not delete old records to reset the UI. Local output is stored here:

- Session JSON: `human_eval/backend/data/sessions/`
- User and model WAV: `human_eval/backend/data/audio/`
- Server logs: the terminal running `serve.py`

Useful exports:

- `/api/admin/export/sessions.jsonl`
- `/api/admin/export/tasks.jsonl`
- `/api/admin/export/conversations.jsonl`
- `/api/admin/export/turns.jsonl`

Run checks:

```bash
python3 -m unittest human_eval.backend.tests.test_core human_eval.backend.tests.test_api
node --check human_eval/app.js
```

## Study design and participant flow

| Task | Participant action | Expected MiniCPM+ behavior | Main measure |
|---|---|---|---|
| S1: simple fact | Ask two related, self-contained stable facts | Stay local | Quality, latency, unnecessary escalation |
| S2: current information | Ask two related, self-contained questions about now or today | Escalate when fresh information is needed | Freshness, correctness, responsiveness |
| S3: reasoning and context | Give options, constraints, and an update across at least three turns | Escalate when deeper reasoning is needed | Constraint reasoning and context retention |

Each participant completes:

1. Device check and consent.
2. Task 1: conversation A, rating, conversation B, rating, pairwise choice.
3. Task 2: the same sequence.
4. Completion page.

Participants may finish a conversation early. Model identity and escalation status remain hidden.

Assignments continuously target these task-pair proportions: 25% S1+S2, 25% S1+S3, and 50% S2+S3. Scenario and model order are balanced without requiring a fixed participant count.

## Human feedback

After each conversation, participants rate 1–5:

- Helpfulness
- Correctness
- Instruction following
- Context consistency
- Clarity and spoken conciseness
- Overall content quality
- Turn-taking naturalness
- Responsiveness

After each task pair, they select conversation A, conversation B, or about the same; choose reasons; and may add a comment.

## Stored data and metrics

- **Identity and assignment:** user/session/task/conversation IDs, capability, scenario, order, model arm, probe setting, and threshold tier.
- **Feedback:** eight ratings, conversation comment, pairwise preference, reasons, task comment, and submission times.
- **Interaction:** user/model WAV, transcripts and source, expert answer, turn count, interruption state, and conversation end reason.
- **Routing:** local/escalated action, threshold, EOT score and series, plus manual expected action and correctness review.
- **Latency:** speech end, gate decision, first model audio, response completion, expert, relay, stall, EOT-read, and ASR timing.
- **Audio quality:** input/output duration, speech detection, RMS/VAD statistics, and short or missing audio flags.
- **Guardrails:** timeout, crash, disconnect, empty response, interruption, missing transcript, and routing-review status.

Raw JSON is the audit record. Flattened JSONL exports are intended for analysis. Exact fields are in [backend/DATA_SCHEMA.md](./backend/DATA_SCHEMA.md).

## Production data location

The browser keeps only a user ID in `sessionStorage`. All study records are written by the FastAPI server.

For a public deployment, run one backend replica, mount a persistent disk at `/data`, and set:

```bash
HUMAN_EVAL_DATA_DIR=/data
HUMAN_EVAL_REQUIRE_TRANSCRIPTS=1
OPENAI_API_KEY=...
```

The server writes `/data/sessions/*.json` and `/data/audio/<conversation_id>/*.wav`. An ephemeral disk is unsafe because a restart or redeploy may erase it. Do not commit participant data to Git.

## API summary

- `POST /api/study-sessions`: create or resume an assignment.
- `GET /api/model/readiness`: warm and check the model; formal mode also checks ASR configuration.
- `WS /api/conversations/{conversation_id}/stream`: proxy blinded audio to the assigned model.
- `POST /api/conversations/{conversation_id}/finalize`: save the conversation result.
- `PUT /api/conversations/{conversation_id}/rating`: save conversation ratings.
- `PUT /api/tasks/{task_id}/comparison`: save pairwise feedback.
- `POST /api/study-sessions/{session_id}/complete`: complete the study.
- `GET /api/admin/export/{table}.jsonl`: export analysis data.
- `PUT /api/admin/turns/{turn_id}/routing-review`: save a transcript-informed routing label.
