# Persisted Human-Evaluation Data

Each `sessions/{session_id}.json` document is the complete audit record for one participant. Audio stays in `audio/{conversation_id}/`. Analysis should normally use the flattened `sessions`, `tasks`, `conversations`, and `turns` JSONL exports.

## Session and assignment

- `user_id`, `session_id`, `study_version`, `schema_version`
- `status`, `created_at`, `started_at`, `completed_at`, `updated_at`
- private `pair_cell`, assignment time, and reservation expiry
- task order, `task_id`, capability `S1`/`S2`/`S3`, scenario ID, and whether escalation is expected for that capability
- conversation order, `conversation_id`, assigned `model`, `probe_on`, and `threshold_tier`

## Primary outcome data

- Per-conversation rating metrics and free-form feedback
- Pairwise preference (`first`, `second`, or `same`), selected reasons, and feedback
- Rating/comparison submission timestamps
- Completion status and completion code
- Task and scenario identity needed to compare matched X/Y prompts

## Interaction data

For every model turn:

- User WAV path, byte count, sample rate, transcript, transcript status, and source (`upstream_asr` or `posthoc_asr`)
- Model WAV path, byte count, sample rate, final transcript, and expert transcript when escalated
- Input-stream start, user-speech start/end, gate decision, first model audio, and response-complete timestamps
- Derived speech-end-to-gate, speech-end-to-first-audio, and speech-end-to-complete latency
- Model-reported first-audio, expert, stall, relay, EOT-read, and optional post-hoc ASR latency
- Input/output duration, speech-detected flag, input RMS mean/max, mean VAD threshold, and silence before EOT

## MiniCPM+ escalation data

- Whether the turn was eligible for escalation and whether it escalated
- Threshold tier, numerical threshold, final EOT score, per-chunk score series, and EOT read time
- Total observed escalation and local-routing counts across MiniCPM+ turns
- Per-turn routing review: expected action, actual action, correctness, reviewer, note, and timestamp
- Reviewed/correct/incorrect/unreviewed routing counts
- Number of S1 MiniCPM+ conversations with any escalation and expected-task MiniCPM+ conversations with zero escalation; these are screening signals, not correctness labels

## Guardrail and data-quality data

- Conversation end reason and status
- Timeout, model crash, disconnect, interruption, and empty-response flags
- Input and output audio anomaly flags
- Structured error timestamp/message
- Completed conversation count, total turn count, MiniCPM+ turn count, and anomaly totals

Null interpretation:

- `user.transcript=null` with `transcript_status=not_configured` means post-hoc ASR was not configured; the WAV remains available.
- MiniCPM+ `routing_review.status=unreviewed` means correctness is intentionally unknown. Routing correctness is never inferred from the task capability alone.
- Expert, stall, and relay fields do not apply to local turns and are omitted in new records.
- Missing core speech timestamps or derived latency fields indicate a collection defect, not “not applicable.” For older affected records, the turn export estimates speech end from gate time minus EOT-read time and sets `speech_end_estimated=true`; raw files remain unchanged.

Analysis exports:

- `sessions.jsonl`: one row per participant, assignment, duration, counts, and anomaly totals.
- `tasks.jsonl`: one row per capability task and pairwise judgment.
- `conversations.jsonl`: one row per model arm with rating metrics, end state, escalation count, and guardrails.
- `turns.jsonl`: one row per user–assistant turn with transcripts, audio references, routing, latency, and audio quality.
