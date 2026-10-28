"""CPU tests for the calendar scheduling agentic RL demo (#192).

All tests are pure Python — no torch, no GPU, no network.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "agentic" / "calendar"))
import game  # noqa: E402
import dataset_generator  # noqa: E402


class TimezoneTest(unittest.TestCase):
    """Tests for fixed-offset time zone conversion."""

    def test_parse_offset_utc(self):
        self.assertEqual(game.parse_offset("UTC"), 0)

    def test_parse_offset_positive(self):
        self.assertEqual(game.parse_offset("UTC+8"), 480)
        self.assertEqual(game.parse_offset("UTC+2"), 120)

    def test_parse_offset_negative(self):
        self.assertEqual(game.parse_offset("UTC-5"), -300)
        self.assertEqual(game.parse_offset("UTC-8"), -480)

    def test_parse_offset_half_hour(self):
        self.assertEqual(game.parse_offset("UTC+5:30"), 330)

    def test_parse_offset_invalid_raises(self):
        with self.assertRaises(ValueError):
            game.parse_offset("PST")
        with self.assertRaises(ValueError):
            game.parse_offset("UTC+25")

    def test_to_utc_simple(self):
        slot = game.TimeSlot(9, 17)
        utc_start, utc_end = game.to_utc(slot, "UTC+8")
        self.assertEqual(utc_start, 1)
        self.assertEqual(utc_end, 9)

    def test_to_utc_negative_offset(self):
        slot = game.TimeSlot(8, 14)
        utc_start, utc_end = game.to_utc(slot, "UTC-5")
        self.assertEqual(utc_start, 13)
        self.assertEqual(utc_end, 19)

    def test_to_utc_cross_midnight(self):
        """A slot that starts before midnight in a positive-offset TZ wraps to previous day in UTC."""
        slot = game.TimeSlot(1, 5)
        utc_start, utc_end = game.to_utc(slot, "UTC+8")
        self.assertEqual(utc_start, 17)
        self.assertEqual(utc_end, 21)


class FindCommonSlotsTest(unittest.TestCase):
    """Tests for availability intersection."""

    def test_two_participants_same_timezone(self):
        participants = {
            "Alice": game.Participant("Alice", "UTC", (game.TimeSlot(9, 17),)),
            "Bob": game.Participant("Bob", "UTC", (game.TimeSlot(10, 18),)),
        }
        meeting = game.Meeting("m1", 1, ("Alice", "Bob"))
        slots = game.find_common_slots(meeting, participants)
        self.assertTrue(len(slots) > 0)
        self.assertIn((10, 17), slots)

    def test_two_participants_different_timezones(self):
        participants = {
            "Alice": game.Participant("Alice", "UTC+8", (game.TimeSlot(9, 17),)),
            "Bob": game.Participant("Bob", "UTC-5", (game.TimeSlot(8, 14),)),
        }
        meeting = game.Meeting("m1", 1, ("Alice", "Bob"))
        slots = game.find_common_slots(meeting, participants)
        self.assertEqual(slots, [])

    def test_different_timezones_with_overlap(self):
        participants = {
            "Alice": game.Participant("Alice", "UTC+8", (game.TimeSlot(9, 21),)),
            "Bob": game.Participant("Bob", "UTC-5", (game.TimeSlot(7, 15),)),
        }
        meeting = game.Meeting("m1", 1, ("Alice", "Bob"))
        slots = game.find_common_slots(meeting, participants)
        self.assertTrue(any(e - s >= 1 for s, e in slots))

    def test_no_solution_returns_empty(self):
        participants = {
            "Alice": game.Participant("Alice", "UTC+8", (game.TimeSlot(9, 10),)),
            "Bob": game.Participant("Bob", "UTC-8", (game.TimeSlot(9, 10),)),
        }
        meeting = game.Meeting("m1", 1, ("Alice", "Bob"))
        slots = game.find_common_slots(meeting, participants)
        self.assertEqual(slots, [])

    def test_three_participants(self):
        participants = {
            "Alice": game.Participant("Alice", "UTC", (game.TimeSlot(8, 20),)),
            "Bob": game.Participant("Bob", "UTC", (game.TimeSlot(10, 18),)),
            "Carol": game.Participant("Carol", "UTC", (game.TimeSlot(12, 16),)),
        }
        meeting = game.Meeting("m1", 2, ("Alice", "Bob", "Carol"))
        slots = game.find_common_slots(meeting, participants)
        self.assertTrue(any(e - s >= 2 for s, e in slots))


class ConflictDetectionTest(unittest.TestCase):
    """Tests for conflict detection with confirmed meetings."""

    def test_no_conflict(self):
        confirmed = {"other": (5, 7)}
        result = game.check_conflict(8, 10, confirmed)
        self.assertIsNone(result)

    def test_conflict_detected(self):
        confirmed = {"other": (5, 9)}
        result = game.check_conflict(8, 10, confirmed)
        self.assertEqual(result, "other")

    def test_conflict_excluded(self):
        confirmed = {"m1": (5, 9)}
        result = game.check_conflict(8, 10, confirmed, exclude_meeting_id="m1")
        self.assertIsNone(result)

    def test_duplicate_confirmation(self):
        """Confirming the same slot twice should not conflict with itself."""
        confirmed = {"m1": (10, 11)}
        result = game.check_conflict(10, 11, confirmed, exclude_meeting_id="m1")
        self.assertIsNone(result)


class ScoreProposalTest(unittest.TestCase):
    """Tests for the scoring function."""

    def _make_state(self, confirmed=None):
        participants = {
            "Alice": game.Participant("Alice", "UTC", (game.TimeSlot(9, 17),)),
            "Bob": game.Participant("Bob", "UTC", (game.TimeSlot(9, 17),)),
        }
        meetings = (game.Meeting("m1", 1, ("Alice", "Bob")),)
        return game.CalendarState(
            participants=participants,
            meetings=meetings,
            confirmed=confirmed or {},
        )

    def test_correct_proposal(self):
        state = self._make_state()
        score = game.score_proposal(state, "m1", 10, 11)
        self.assertEqual(score, 1.0)

    def test_wrong_duration(self):
        state = self._make_state()
        score = game.score_proposal(state, "m1", 10, 12)
        self.assertEqual(score, -1.0)

    def test_outside_availability(self):
        state = self._make_state()
        score = game.score_proposal(state, "m1", 7, 8)
        self.assertEqual(score, -1.0)

    def test_conflict_with_confirmed(self):
        state = self._make_state(confirmed={"other": (10, 12)})
        score = game.score_proposal(state, "m1", 10, 11)
        self.assertEqual(score, -1.0)

    def test_invalid_meeting_id(self):
        state = self._make_state()
        score = game.score_proposal(state, "nonexistent", 10, 11)
        self.assertEqual(score, -1.0)


class ToolExecutionTest(unittest.TestCase):
    """Tests for the tool execution functions used in multi-turn rollout."""

    def _make_state(self):
        return game.CalendarState(
            participants={
                "Alice": game.Participant("Alice", "UTC+8", (game.TimeSlot(9, 17),)),
                "Bob": game.Participant("Bob", "UTC-5", (game.TimeSlot(13, 22),)),
            },
            meetings=(game.Meeting("m1", 1, ("Alice", "Bob")),),
        )

    def test_execute_query_availability(self):
        state = self._make_state()
        result = game.execute_query_availability(state, "Alice")
        self.assertEqual(result["participant"], "Alice")
        self.assertEqual(result["timezone"], "UTC+8")
        self.assertTrue(len(result["available_slots_utc"]) > 0)
        # Alice UTC+8 9-17 → UTC 1-9
        slot = result["available_slots_utc"][0]
        self.assertEqual(slot["utc_start"], 1)
        self.assertEqual(slot["utc_end"], 9)

    def test_execute_query_availability_unknown_participant(self):
        state = self._make_state()
        result = game.execute_query_availability(state, "Unknown")
        self.assertIn("error", result)

    def test_execute_propose_slot_valid(self):
        state = self._make_state()
        # Alice UTC 1-9, Bob UTC 18-22 → no overlap, so pick a meeting where overlap exists
        state2 = game.CalendarState(
            participants={
                "Alice": game.Participant("Alice", "UTC", (game.TimeSlot(9, 17),)),
                "Bob": game.Participant("Bob", "UTC", (game.TimeSlot(10, 18),)),
            },
            meetings=(game.Meeting("m1", 1, ("Alice", "Bob")),),
        )
        result = game.execute_propose_slot(state2, "m1", 10, 11)
        self.assertTrue(result["valid"])

    def test_execute_propose_slot_invalid(self):
        state = self._make_state()
        result = game.execute_propose_slot(state, "m1", 99, 100)
        self.assertFalse(result["valid"])
        self.assertIn("error", result)

    def test_execute_confirm_slot_valid(self):
        state = game.CalendarState(
            participants={
                "Alice": game.Participant("Alice", "UTC", (game.TimeSlot(9, 17),)),
                "Bob": game.Participant("Bob", "UTC", (game.TimeSlot(10, 18),)),
            },
            meetings=(game.Meeting("m1", 1, ("Alice", "Bob")),),
        )
        result = game.execute_confirm_slot(state, "m1", 10, 11)
        self.assertTrue(result["confirmed"])

    def test_execute_confirm_slot_invalid(self):
        state = self._make_state()
        result = game.execute_confirm_slot(state, "m1", 99, 100)
        self.assertFalse(result["confirmed"])
        self.assertIn("error", result)


class RewardFlowTest(unittest.TestCase):
    """Tests for the multi-turn reward flow checking."""

    def _make_reward_record(self, tool_calls, source_record):
        """Build a fake RewardRecord-like object for testing."""
        from types import SimpleNamespace
        return SimpleNamespace(
            tool_calls=tool_calls,
            source_record=source_record,
        )

    def _make_source(self):
        return {
            "participants": {
                "Alice": {"name": "Alice", "timezone": "UTC", "available_slots": [{"start_hour": 9, "end_hour": 17}]},
                "Bob": {"name": "Bob", "timezone": "UTC", "available_slots": [{"start_hour": 10, "end_hour": 18}]},
            },
            "meetings": [{"id": "m1", "duration_hours": 1, "required_participants": ["Alice", "Bob"]}],
            "confirmed": {},
            "target_meeting_id": "m1",
        }

    def test_correct_flow_full_reward(self):
        """query → propose → confirm in correct order with valid slot → 1.0."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "agentic" / "calendar"))
        import reward as reward_mod
        calls = [
            {"name": "query_availability", "arguments": {"participant": "Alice"}},
            {"name": "query_availability", "arguments": {"participant": "Bob"}},
            {"name": "propose_slot", "arguments": {"meeting_id": "m1", "utc_start_hour": 10, "utc_end_hour": 11}},
            {"name": "confirm_slot", "arguments": {"meeting_id": "m1", "utc_start_hour": 10, "utc_end_hour": 11}},
        ]
        record = self._make_reward_record(calls, self._make_source())
        result = reward_mod.reward_fn(record)
        self.assertEqual(result, 1.0)

    def test_wrong_flow_partial_reward(self):
        """Valid slot but missing query → 0.5."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "agentic" / "calendar"))
        import reward as reward_mod
        calls = [
            {"name": "propose_slot", "arguments": {"meeting_id": "m1", "utc_start_hour": 10, "utc_end_hour": 11}},
            {"name": "confirm_slot", "arguments": {"meeting_id": "m1", "utc_start_hour": 10, "utc_end_hour": 11}},
        ]
        record = self._make_reward_record(calls, self._make_source())
        result = reward_mod.reward_fn(record)
        self.assertEqual(result, 0.5)

    def test_invalid_slot_negative_reward(self):
        """Wrong slot → -1.0."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "agentic" / "calendar"))
        import reward as reward_mod
        calls = [
            {"name": "query_availability", "arguments": {"participant": "Alice"}},
            {"name": "query_availability", "arguments": {"participant": "Bob"}},
            {"name": "propose_slot", "arguments": {"meeting_id": "m1", "utc_start_hour": 99, "utc_end_hour": 100}},
            {"name": "confirm_slot", "arguments": {"meeting_id": "m1", "utc_start_hour": 99, "utc_end_hour": 100}},
        ]
        record = self._make_reward_record(calls, self._make_source())
        result = reward_mod.reward_fn(record)
        self.assertEqual(result, -1.0)

    def test_no_confirmation_partial_reward(self):
        """Valid propose but no confirm_slot → 0.5 (correct slot, incomplete flow)."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "agentic" / "calendar"))
        import reward as reward_mod
        calls = [
            {"name": "query_availability", "arguments": {"participant": "Alice"}},
            {"name": "propose_slot", "arguments": {"meeting_id": "m1", "utc_start_hour": 10, "utc_end_hour": 11}},
        ]
        record = self._make_reward_record(calls, self._make_source())
        result = reward_mod.reward_fn(record)
        self.assertEqual(result, 0.5)

    def test_wrong_order_partial_reward(self):
        """Propose before query → 0.5."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "agentic" / "calendar"))
        import reward as reward_mod
        calls = [
            {"name": "propose_slot", "arguments": {"meeting_id": "m1", "utc_start_hour": 10, "utc_end_hour": 11}},
            {"name": "query_availability", "arguments": {"participant": "Alice"}},
            {"name": "confirm_slot", "arguments": {"meeting_id": "m1", "utc_start_hour": 10, "utc_end_hour": 11}},
        ]
        record = self._make_reward_record(calls, self._make_source())
        result = reward_mod.reward_fn(record)
        self.assertEqual(result, 0.5)


class DatasetGeneratorTest(unittest.TestCase):
    """Tests for dataset generation."""

    def test_generation_deterministic_same_seed(self):
        r1 = dataset_generator.generate_records(10, seed=42)
        r2 = dataset_generator.generate_records(10, seed=42)
        self.assertEqual(r1, r2)

    def test_generation_different_seeds(self):
        r1 = dataset_generator.generate_records(10, seed=1)
        r2 = dataset_generator.generate_records(10, seed=2)
        self.assertNotEqual(r1, r2)

    def test_held_out_split_disjoint(self):
        records = dataset_generator.generate_records(100, seed=2026)
        train, test = dataset_generator.split_held_out(records, fraction=0.2)
        train_ids = {r["id"] for r in train}
        test_ids = {r["id"] for r in test}
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

    def test_generated_records_are_valid(self):
        records = dataset_generator.generate_records(10, seed=2026)
        for raw in records:
            state = game.record_to_state(raw)
            self.assertIsInstance(state, game.CalendarState)
            self.assertTrue(len(state.participants) >= 2)
            self.assertTrue(len(state.meetings) >= 1)


class PromptFormatTest(unittest.TestCase):
    """Tests for prompt rendering."""

    def test_format_prompt_contains_key_info(self):
        state = game.CalendarState(
            participants={
                "Alice": game.Participant("Alice", "UTC+8", (game.TimeSlot(9, 17),)),
                "Bob": game.Participant("Bob", "UTC-5", (game.TimeSlot(8, 14),)),
            },
            meetings=(game.Meeting("m1", 1, ("Alice", "Bob")),),
        )
        prompt = game.format_prompt(state, "m1")
        self.assertIn("Alice", prompt)
        self.assertIn("UTC+8", prompt)
        self.assertIn("Bob", prompt)
        self.assertIn("UTC-5", prompt)
        self.assertIn("query_availability", prompt)
        self.assertIn("propose_slot", prompt)
        self.assertIn("confirm_slot", prompt)

    def test_format_prompt_does_not_leak_availability(self):
        """Prompt must NOT include availability slot details — model must discover them via tools."""
        state = game.CalendarState(
            participants={
                "Alice": game.Participant("Alice", "UTC+8", (game.TimeSlot(9, 17),)),
            },
            meetings=(game.Meeting("m1", 1, ("Alice",)),),
        )
        prompt = game.format_prompt(state, "m1")
        # The prompt should mention the timezone but NOT the specific hours.
        self.assertIn("UTC+8", prompt)
        self.assertNotIn("09:00-17:00", prompt)
        self.assertNotIn("available=[", prompt)

    def test_format_prompt_with_confirmed(self):
        state = game.CalendarState(
            participants={
                "Alice": game.Participant("Alice", "UTC", (game.TimeSlot(9, 17),)),
            },
            meetings=(game.Meeting("m1", 1, ("Alice",)),),
            confirmed={"other": (10, 12)},
        )
        prompt = game.format_prompt(state, "m1")
        self.assertIn("Already confirmed", prompt)
        self.assertIn("other", prompt)


class SerializationTest(unittest.TestCase):
    """Tests for state serialization/deserialization."""

    def test_round_trip(self):
        state = game.CalendarState(
            participants={
                "Alice": game.Participant("Alice", "UTC+8", (game.TimeSlot(9, 17),)),
            },
            meetings=(game.Meeting("m1", 1, ("Alice",)),),
            confirmed={"other": (10, 12)},
        )
        record = game.state_to_record(state, target_meeting_id="m1")
        restored = game.record_to_state(record)
        self.assertEqual(restored.participants["Alice"].timezone, "UTC+8")
        self.assertEqual(restored.confirmed["other"], (10, 12))
        self.assertEqual(restored.meetings[0].id, "m1")


class TimeSlotValidationTest(unittest.TestCase):
    """Tests for TimeSlot input validation."""

    def test_zero_duration_raises(self):
        with self.assertRaises(ValueError):
            game.TimeSlot(10, 10)

    def test_negative_range_raises(self):
        with self.assertRaises(ValueError):
            game.TimeSlot(15, 10)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            game.TimeSlot(-1, 5)
        with self.assertRaises(ValueError):
            game.TimeSlot(10, 25)


if __name__ == "__main__":
    unittest.main()