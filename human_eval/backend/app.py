"""FastAPI entry point for the human-evaluation backend.

Run with:
    uvicorn human_eval.backend.app:app --host 127.0.0.1 --port 4173
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .core import (
    JsonSessionStore,
    analysis_rows,
    create_assignment,
    public_session,
    utc_now,
)
from .model_gateway import (
    ConversationRecorder,
    DemoModelGateway,
    transcript_collection_settings,
)


HUMAN_EVAL_DIR = Path(__file__).resolve().parent.parent
SCENARIOS = json.loads((HUMAN_EVAL_DIR / "scenarios.json").read_text(encoding="utf-8"))
DATA_DIR = Path(os.getenv("HUMAN_EVAL_DATA_DIR", Path(__file__).parent / "data"))
CONVERSATION_LIMIT_SECONDS = int(os.getenv("HUMAN_EVAL_CONVERSATION_SECONDS", "120"))

store = JsonSessionStore(DATA_DIR)
model_gateway = DemoModelGateway()
assignment_lock = threading.Lock()
model_warm_task: asyncio.Task[dict[str, Any]] | None = None
app = FastAPI(title="MiniCPM Human Evaluation", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:4173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, max_length=128)
    force_new: bool = False


class RatingRequest(BaseModel):
    metrics: Dict[str, Union[float, int, bool]]
    feedback: str = Field(default="", max_length=5000)


class ComparisonRequest(BaseModel):
    preference: str
    reasons: list[str] = Field(default_factory=list)
    feedback: str = Field(default="", max_length=5000)


class FinalizeConversationRequest(BaseModel):
    end_reason: str
    timeout: bool = False
    crash: bool = False
    disconnect: bool = False
    error: Optional[str] = Field(default=None, max_length=2000)


class RoutingReviewRequest(BaseModel):
    expected_action: Literal["local", "escalate", "not_applicable"]
    reviewer: str = Field(min_length=1, max_length=128)
    note: str = Field(default="", max_length=2000)


def _not_found(label: str, identifier: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{label} not found: {identifier}")


async def ensure_model_ready() -> dict[str, Any]:
    """Share one in-flight GPU warm-up across pages and conversations."""
    global model_warm_task
    if model_warm_task is None:
        model_warm_task = asyncio.create_task(model_gateway.wait_until_ready())
    task = model_warm_task
    try:
        return await asyncio.shield(task)
    finally:
        if task.done() and model_warm_task is task:
            model_warm_task = None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "study_version": SCENARIOS["version"],
        **transcript_collection_settings(),
    }


@app.get("/api/model/readiness")
async def model_readiness() -> dict[str, Any]:
    transcript_settings = transcript_collection_settings()
    if (
        transcript_settings["transcripts_required"]
        and not transcript_settings["posthoc_asr_configured"]
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Full transcript collection is required, but OPENAI_API_KEY "
                "is not configured on the study backend"
            ),
        )
    try:
        payload = await ensure_model_ready()
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Model warm-up failed: {error}")
    return {
        "ready": True,
        "busy": bool(payload.get("busy")),
        "load_s": payload.get("load_s"),
        **transcript_settings,
    }


@app.put("/api/admin/turns/{turn_id}/routing-review")
def save_routing_review(
    turn_id: str, request: RoutingReviewRequest
) -> dict[str, Any]:
    """Save a transcript-informed routing label; never infer it from task alone."""
    result: dict[str, Any] = {}

    def update(
        _task: dict[str, Any], conversation: dict[str, Any], turn: dict[str, Any]
    ) -> None:
        if conversation.get("model") != "minicpm_plus":
            raise ValueError("Routing review only applies to MiniCPM+ turns")
        actual_action = (
            "escalate" if turn.get("gate", {}).get("escalated") else "local"
        )
        correct = (
            None
            if request.expected_action == "not_applicable"
            else actual_action == request.expected_action
        )
        review = {
            "status": "reviewed",
            "expected_action": request.expected_action,
            "actual_action": actual_action,
            "correct": correct,
            "note": request.note,
            "reviewer": request.reviewer,
            "reviewed_at": utc_now(),
        }
        turn["routing_review"] = review
        result.update(review)

    try:
        store.mutate_turn(turn_id, update)
    except KeyError:
        raise _not_found("Turn", turn_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    return result


@app.post("/api/study-sessions")
def create_or_resume_session(request: CreateSessionRequest) -> dict[str, Any]:
    with assignment_lock:
        existing = store.list_sessions()
        if request.user_id and not request.force_new:
            for session in reversed(existing):
                if session.get("user_id") == request.user_id:
                    if session.get("status") != "completed":
                        store.save(session)  # renew the active reservation
                    return public_session(session)
        session = create_assignment(SCENARIOS, existing, request.user_id)
        store.create(session)
    print(
        "[assignment]",
        json.dumps(
            {
                "session_id": session["session_id"],
                "user_id": session["user_id"],
                "pair_cell": session["assignment"]["pair_cell"],
                "tasks": [
                    {
                        "capability": task["capability"],
                        "sequence_cell": task["sequence_cell"],
                        "models": [item["model"] for item in task["conversations"]],
                    }
                    for task in session["tasks"]
                ],
            }
        ),
        flush=True,
    )
    return public_session(session)


@app.get("/api/study-sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    try:
        return public_session(store.get(session_id))
    except (KeyError, ValueError):
        raise _not_found("Session", session_id)


@app.put("/api/conversations/{conversation_id}/rating")
def save_rating(conversation_id: str, request: RatingRequest) -> dict[str, bool]:
    try:
        store.mutate_conversation(
            conversation_id,
            lambda _task, conversation: conversation.update(
                {
                    "rating": {
                        "metrics": request.metrics,
                        "feedback": request.feedback,
                        "submitted_at": utc_now(),
                    }
                }
            ),
        )
    except (KeyError, ValueError):
        raise _not_found("Conversation", conversation_id)
    return {"saved": True}


@app.put("/api/tasks/{task_id}/comparison")
def save_comparison(task_id: str, request: ComparisonRequest) -> dict[str, bool]:
    if request.preference not in {"first", "second", "same"}:
        raise HTTPException(status_code=422, detail="Invalid preference")
    try:
        store.mutate_task(
            task_id,
            lambda task: task.update(
                {
                    "comparison": {
                        "preference": request.preference,
                        "reasons": request.reasons,
                        "feedback": request.feedback,
                        "submitted_at": utc_now(),
                    }
                }
            ),
        )
    except (KeyError, ValueError):
        raise _not_found("Task", task_id)
    return {"saved": True}


def _finish_conversation(
    conversation_id: str,
    end_reason: str,
    *,
    timeout: bool = False,
    crash: bool = False,
    disconnect: bool = False,
    error: Optional[str] = None,
) -> None:
    def update(_task: dict[str, Any], conversation: dict[str, Any]) -> None:
        if crash:
            status = "failed"
        elif end_reason in {"user_finished", "time_limit"}:
            status = "completed"
        else:
            status = "abandoned"
        conversation.update(
            {
                "status": status,
                "ended_at": utc_now(),
                "end_reason": end_reason,
            }
        )
        if error:
            conversation.setdefault("errors", []).append(
                {"timestamp": utc_now(), "message": error[:2000]}
            )
        anomalies = conversation.setdefault("anomalies", {})
        anomalies["timeout"] = anomalies.get("timeout", False) or timeout
        anomalies["crash"] = anomalies.get("crash", False) or crash
        anomalies["disconnect"] = anomalies.get("disconnect", False) or disconnect

    store.mutate_conversation(conversation_id, update)


def _participant_model_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Keep routing/gate telemetry server-side while forwarding voice UX events."""
    event_type = payload.get("type")
    if event_type == "hello":
        return {"type": "ready"}
    if event_type == "turn":
        return {"type": "turn"}
    allowed_fields = {
        "phase": ("v",),
        "audio": ("sr", "pcm"),
        "speech": ("on",),
        "duck": (),
        "resume": (),
        "interrupt": (),
    }
    if event_type not in allowed_fields:
        return None
    return {
        "type": event_type,
        **{key: payload.get(key) for key in allowed_fields[event_type]},
    }


@app.post("/api/conversations/{conversation_id}/finalize")
def finalize_conversation(
    conversation_id: str, request: FinalizeConversationRequest
) -> dict[str, bool]:
    try:
        _finish_conversation(
            conversation_id,
            request.end_reason,
            timeout=request.timeout,
            crash=request.crash,
            disconnect=request.disconnect,
            error=request.error,
        )
    except (KeyError, ValueError):
        raise _not_found("Conversation", conversation_id)
    return {"saved": True}


@app.post("/api/study-sessions/{session_id}/complete")
def complete_session(session_id: str) -> dict[str, str]:
    try:
        session = store.get(session_id)
    except (KeyError, ValueError):
        raise _not_found("Session", session_id)
    conversations = [
        conversation
        for task in session["tasks"]
        for conversation in task["conversations"]
    ]
    final_statuses = {"completed", "failed", "abandoned"}
    if not all(item.get("status") in final_statuses and item.get("rating") for item in conversations):
        raise HTTPException(status_code=409, detail="All conversations and ratings must be finalized")
    if not all(task.get("comparison") for task in session["tasks"]):
        raise HTTPException(status_code=409, detail="Both task comparisons must be complete")

    completion_code = f"DONE-{session_id[-8:].upper()}"

    def mark_complete(target: dict[str, Any]) -> None:
        target.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "completion_code": completion_code,
            }
        )

    store.mutate(session_id, mark_complete)
    return {"completion_code": completion_code}


@app.get("/api/admin/export.jsonl")
def export_sessions() -> Response:
    # TODO: require study-admin authentication before any non-local deployment.
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in store.list_sessions())
    return Response(content=body, media_type="application/x-ndjson")


@app.get("/api/admin/export/{table}.jsonl")
def export_analysis_table(table: str) -> Response:
    # TODO: require study-admin authentication before any non-local deployment.
    if table not in {"sessions", "tasks", "conversations", "turns"}:
        raise HTTPException(status_code=404, detail="Unknown analysis table")
    body = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in analysis_rows(store.list_sessions(), table)
    )
    return Response(content=body, media_type="application/x-ndjson")


@app.websocket("/api/conversations/{conversation_id}/stream")
async def conversation_stream(client: WebSocket, conversation_id: str) -> None:
    await client.accept()
    try:
        _, task, conversation = store.find_conversation(conversation_id)
    except (KeyError, ValueError):
        await client.close(code=4404, reason="Conversation not found")
        return

    recorder = ConversationRecorder(
        store, conversation_id, conversation["model"], task["capability"]
    )
    upstream_url = model_gateway.websocket_url(conversation["model"])
    finish_reason = "disconnect"

    try:
        import websockets

        await ensure_model_ready()
        async with websockets.connect(
            upstream_url, max_size=None, open_timeout=60
        ) as upstream:
            async def client_to_model() -> str:
                while True:
                    message = await client.receive()
                    if message.get("type") == "websocket.disconnect":
                        return "disconnect"
                    if message.get("bytes") is not None:
                        chunk = message["bytes"]
                        recorder.record_client_audio(chunk)
                        await upstream.send(chunk)
                        continue
                    text = message.get("text")
                    if text is None:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = {}
                    if payload.get("type") == "finish_conversation":
                        return "user_finished"
                    recorder.record_client_event(payload)
                    await upstream.send(text)

            async def model_to_client() -> str:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await client.send_bytes(message)
                        continue
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        await client.send_text(message)
                        continue
                    recorder.record_server_event(payload)
                    participant_event = _participant_model_event(payload)
                    if participant_event:
                        await client.send_json(participant_event)
                    if payload.get("type") == "turn":
                        await recorder.finalize_turn(payload)
                return "upstream_closed"

            async def time_limit() -> str:
                await asyncio.sleep(CONVERSATION_LIMIT_SECONDS)
                await client.send_json({"type": "auto_finish", "reason": "time_limit"})
                return "time_limit"

            tasks = {
                asyncio.create_task(client_to_model()),
                asyncio.create_task(model_to_client()),
                asyncio.create_task(time_limit()),
            }
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finish_reason = next(iter(done)).result()
            for pending_task in pending:
                pending_task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    except WebSocketDisconnect:
        finish_reason = "disconnect"
    except Exception as error:
        finish_reason = "crash"
        print(
            f"[model-error] {conversation_id}: {type(error).__name__}: {error}",
            flush=True,
        )
        try:
            await client.send_json({"type": "error", "message": "Model connection failed"})
        except Exception:
            pass
        _finish_conversation(
            conversation_id,
            "crash",
            crash=True,
            error=f"{type(error).__name__}: {error}",
        )
        return
    finally:
        if finish_reason != "crash":
            try:
                _finish_conversation(
                    conversation_id,
                    finish_reason,
                    timeout=finish_reason == "time_limit",
                    disconnect=finish_reason in {"disconnect", "upstream_closed"},
                )
            except (KeyError, ValueError):
                pass
        try:
            await client.close()
        except Exception:
            pass


# Keep the participant UI and API on one origin without exposing backend/data.
@app.get("/", include_in_schema=False)
def participant_ui() -> FileResponse:
    return FileResponse(HUMAN_EVAL_DIR / "index.html")


@app.get("/app.js", include_in_schema=False)
def participant_javascript() -> FileResponse:
    return FileResponse(HUMAN_EVAL_DIR / "app.js", media_type="text/javascript")


@app.get("/styles.css", include_in_schema=False)
def participant_styles() -> FileResponse:
    return FileResponse(HUMAN_EVAL_DIR / "styles.css", media_type="text/css")


@app.get("/scenarios.json", include_in_schema=False)
def participant_scenarios() -> FileResponse:
    return FileResponse(HUMAN_EVAL_DIR / "scenarios.json", media_type="application/json")
