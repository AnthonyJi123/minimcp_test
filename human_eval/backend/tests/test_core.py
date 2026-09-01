from __future__ import annotations

import asyncio
import base64
import json
import random
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from human_eval.backend.core import (
    JsonSessionStore,
    analysis_rows,
    create_assignment,
    public_session,
    recompute_summary,
)
from human_eval.backend.model_gateway import ConversationRecorder, DemoModelGateway


SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "scenarios.json"


class AssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

    def test_32_participant_block_is_balanced(self) -> None:
        rng = random.Random(7)
        sessions = []
        for index in range(32):
            sessions.append(
                create_assignment(self.scenarios, sessions, f"user-{index}", rng=rng)
            )

        pair_counts = Counter(row["assignment"]["pair_cell"] for row in sessions)
        self.assertEqual(pair_counts, {"s1_s2": 8, "s1_s3": 8, "s2_s3": 16})

        sequences = defaultdict(Counter)
        for session in sessions:
            for task in session["tasks"]:
                sequences[task["capability"]][task["sequence_cell"]] += 1
                self.assertEqual(
                    {item["model"] for item in task["conversations"]},
                    {"minicpm", "minicpm_plus"},
                )
        for counts in sequences.values():
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        capability_codes = {
            task["capability_key"]: task["capability"]
            for session in sessions
            for task in session["tasks"]
        }
        self.assertEqual(capability_codes["simple_guardrail"], "S1")
        self.assertEqual(capability_codes["realtime"], "S2")
        self.assertEqual(capability_codes["context_reasoning"], "S3")

    def test_arbitrary_participant_count_stays_close_to_target_ratio(self) -> None:
        rng = random.Random(11)
        sessions = []
        for index in range(11):
            sessions.append(
                create_assignment(self.scenarios, sessions, f"user-{index}", rng=rng)
            )
        counts = Counter(row["assignment"]["pair_cell"] for row in sessions)
        normalized = [counts["s1_s2"], counts["s1_s3"], counts["s2_s3"] / 2]
        self.assertLessEqual(max(normalized) - min(normalized), 1)

    def test_expired_or_failed_sessions_do_not_hold_quota(self) -> None:
        stale = create_assignment(self.scenarios, [], "stale", rng=random.Random(2))
        stale["assignment"]["reservation_expires_at"] = "2000-01-01T00:00:00+00:00"
        failed = create_assignment(self.scenarios, [], "failed", rng=random.Random(3))
        failed["status"] = "failed"

        expected = create_assignment(self.scenarios, [], "next", rng=random.Random(9))
        actual = create_assignment(
            self.scenarios, [stale, failed], "next", rng=random.Random(9)
        )
        self.assertEqual(actual["assignment"]["pair_cell"], expected["assignment"]["pair_cell"])
        self.assertEqual(
            [(task["capability"], task["sequence_cell"]) for task in actual["tasks"]],
            [(task["capability"], task["sequence_cell"]) for task in expected["tasks"]],
        )

    def test_public_assignment_hides_model_fields(self) -> None:
        session = create_assignment(self.scenarios, [], "user-1", rng=random.Random(3))
        public = public_session(session)
        serialized = json.dumps(public)
        for hidden in ("model", "probe_on", "threshold_tier", "sequence_cell"):
            self.assertNotIn(f'"{hidden}"', serialized)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

    def test_session_and_summary_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonSessionStore(directory)
            session = create_assignment(self.scenarios, [], "user-1", rng=random.Random(1))
            conversation = session["tasks"][0]["conversations"][0]
            conversation["model"] = "minicpm_plus"
            conversation["turns"].append(
                {
                    "turn_id": "turn_test",
                    "gate": {"escalated": True},
                    "anomalies": {"empty_response": False},
                }
            )
            recompute_summary(session)
            store.create(session)
            loaded = store.get(session["session_id"])
            self.assertEqual(loaded["user_id"], "user-1")
            self.assertEqual(loaded["summary"]["turn_count"], 1)
            self.assertEqual(loaded["summary"]["escalation_count"], 1)
            self.assertNotIn(
                "inappropriate_escalation_conversation_count", loaded["summary"]
            )


class ModelGatewayTests(unittest.TestCase):
    def test_model_arms_map_to_demo_probe_switch(self) -> None:
        gateway = DemoModelGateway()
        self.assertIn("probe_on=0", gateway.websocket_url("minicpm"))
        self.assertIn("probe_on=1", gateway.websocket_url("minicpm_plus"))
        self.assertIn("tier=balanced", gateway.websocket_url("minicpm_plus"))

    def test_turn_telemetry_and_audio_are_persisted(self) -> None:
        scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            store = JsonSessionStore(directory)
            session = create_assignment(scenarios, [], "user-1", rng=random.Random(2))
            task = session["tasks"][0]
            conversation = next(
                item for item in task["conversations"] if item["model"] == "minicpm_plus"
            )
            store.create(session)
            recorder = ConversationRecorder(
                store,
                conversation["conversation_id"],
                conversation["model"],
                task["capability"],
            )
            recorder.record_server_event(
                {"type": "hello", "tier": "balanced", "thr": 0.62, "probe_on": True}
            )
            recorder.record_client_audio(b"\x00\x00" * 2000)
            recorder.record_server_event({"type": "speech", "on": True})
            recorder.record_server_event(
                {"type": "vu", "rms": 0.03, "thr": 0.01, "speech": True, "sil": 0.2}
            )
            recorder.record_server_event({"type": "speech", "on": False})
            recorder.record_server_event({"type": "score", "v": 0.41})
            recorder.record_server_event(
                {"type": "eot", "score": 0.81, "thr": 0.62, "ms": 28, "fired": True}
            )
            recorder.record_server_event(
                {
                    "type": "audio",
                    "pcm": base64.b64encode(b"\x00\x00" * 1600).decode(),
                }
            )
            asyncio.run(
                recorder.finalize_turn(
                    {
                        "type": "turn",
                        "fired": True,
                        "mode": "escalated",
                        "eot_score": 0.81,
                        "threshold": 0.62,
                        "scores": [0.41],
                        "uplink_text": "Test question",
                        "answer": "Test answer",
                        "expert_answer": "Verified answer",
                        "first_audio_ms": 300,
                        "expert_latency_s": 1.2,
                    }
                )
            )
            persisted = store.get(session["session_id"])
            saved_conversation = next(
                item
                for saved_task in persisted["tasks"]
                for item in saved_task["conversations"]
                if item["conversation_id"] == conversation["conversation_id"]
            )
            turn = saved_conversation["turns"][0]
            self.assertTrue(turn["gate"]["escalated"])
            self.assertEqual(turn["gate"]["score"], 0.81)
            self.assertEqual(turn["user"]["transcript"], "Test question")
            self.assertEqual(turn["user"]["transcript_source"], "upstream_asr")
            self.assertEqual(turn["routing_review"]["status"], "unreviewed")
            self.assertEqual(turn["routing_review"]["actual_action"], "escalate")
            self.assertIsNone(turn["routing_review"]["correct"])
            self.assertIsNotNone(turn["timestamps"]["user_speech_ended_at"])
            self.assertIn("speech_end_to_gate", turn["latency_ms"])
            self.assertTrue(turn["audio_quality"]["speech_detected"])
            self.assertEqual(turn["audio_quality"]["input_rms_mean"], 0.03)
            self.assertEqual(turn["audio_quality"]["silence_before_eot_s"], 0.2)
            self.assertTrue(Path(turn["user"]["audio_path"]).exists())
            self.assertTrue(Path(turn["model_response"]["audio_path"]).exists())

            turn_rows = analysis_rows([persisted], "turns")
            self.assertEqual(len(turn_rows), 1)
            self.assertEqual(turn_rows[0]["model"], "minicpm_plus")
            self.assertEqual(turn_rows[0]["user_transcript"], "Test question")
            self.assertEqual(turn_rows[0]["user_transcript_source"], "upstream_asr")
            self.assertEqual(turn_rows[0]["routing_review_status"], "unreviewed")
            self.assertIn("latency_speech_end_to_gate_ms", turn_rows[0])


if __name__ == "__main__":
    unittest.main()
