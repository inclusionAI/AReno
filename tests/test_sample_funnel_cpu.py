from __future__ import annotations

import json
import unittest

from click.testing import CliRunner

from areno.api.funnel import FunnelCounters, build_funnel, reconcile
from areno.cli.funnel import funnel_command
from areno.cli.main import main

# --------------------------------------------------------------------------
# Pure-function tests for areno.api.funnel
# --------------------------------------------------------------------------


class FunnelCoreTest(unittest.TestCase):
    """`build_funnel` / `reconcile` are pure and GPU-free."""

    def test_build_funnel_sft_marks_untracked_stages(self):
        """SFT has no rollout, so generated/length_valid serialize as None."""
        counters = FunnelCounters(
            step=0,
            source="sft",
            loaded=64,
            contract_valid=60,
            generated=None,
            length_valid=None,
            trainable_token_valid=60,
            trained=60,
            drop_reasons={"contract_valid": ["empty_or_over_budget"]},
        )
        payload = build_funnel(counters)
        self.assertEqual(payload["source"], "sft")
        self.assertIsNone(payload["stages"]["generated"])
        self.assertIsNone(payload["stages"]["length_valid"])
        self.assertEqual(payload["stages"]["trained"], 60)
        self.assertEqual(payload["drop_reasons"]["contract_valid"], ["empty_or_over_budget"])

    def test_reconcile_silent_when_consistent(self):
        """A monotonic, fully-tracked funnel emits no warnings."""
        counters = FunnelCounters(
            step=1,
            source="online_rl",
            loaded=32,
            contract_valid=30,
            generated=90,
            length_valid=88,
            trainable_token_valid=88,
            trained=88,
        )
        self.assertEqual(reconcile(counters), [])

    def test_reconcile_flags_backwards_counts(self):
        """`trained` cannot exceed the prior tracked stage."""
        counters = FunnelCounters(
            step=1,
            source="online_rl",
            loaded=32,
            contract_valid=30,
            generated=90,
            length_valid=88,
            trainable_token_valid=88,
            trained=90,  # more than length_valid
        )
        warnings = reconcile(counters)
        self.assertTrue(any("trained" in w and "exceeds" in w for w in warnings))

    def test_reconcile_flags_untracked_required_stage(self):
        """Loaded is never optional; unsetting it warns."""
        counters = FunnelCounters(step=1, source="online_rl", loaded=None, trained=10)
        warnings = reconcile(counters)
        self.assertTrue(any("loaded" in w and "not tracked" in w for w in warnings))

    def test_reconcile_allows_optional_untracked_stages(self):
        """SFT leaving generated/length_valid unset must NOT warn."""
        counters = FunnelCounters(
            step=0,
            source="sft",
            loaded=64,
            contract_valid=60,
            trainable_token_valid=60,
            trained=60,
        )
        self.assertEqual(reconcile(counters), [])

    def test_build_funnel_emits_no_sample_content_fields(self):
        """`build_funnel` output must carry only counts and reason codes."""
        counters = FunnelCounters(
            step=0,
            source="online_rl",
            loaded=10,
            trained=9,
            drop_reasons={"loaded": ["prompt_too_long"]},
        )
        payload = build_funnel(counters)
        self.assertEqual(set(payload.keys()), {"step", "source", "stages", "drop_reasons"})
        # No field exists for prompt/completion/messages/sample text.
        for forbidden in ("prompt", "completion", "messages", "final_answer", "tokens"):
            self.assertNotIn(forbidden, payload)
            self.assertNotIn(forbidden, payload["stages"])


class FunnelBarTest(unittest.TestCase):
    """ASCII bar rendering is proportional and visually stable."""

    def test_bar_full_when_value_equals_baseline(self):
        from areno.cli.funnel import _bar

        # The baseline stage always fills the full width (24 cells), no padding.
        full = _bar(50, 50)
        self.assertEqual(len(full.rstrip()), 24)
        self.assertIn("█", full)

    def test_bar_shorter_when_value_below_baseline(self):
        from areno.cli.funnel import _bar

        # 4 out of 8 baseline -> 12 of 24 cells, strictly shorter than full.
        full = _bar(8, 8)
        partial = _bar(4, 8)
        self.assertLess(len(partial.rstrip()), len(full.rstrip()))
        self.assertEqual(len(partial.rstrip()), 12)

    def test_bar_na_for_untracked_stage(self):
        from areno.cli.funnel import _bar

        # n/a stages render a dash placeholder, not a proportion bar.
        self.assertTrue(_bar(None, 10).startswith("—"))

    def test_baseline_is_global_max(self):
        from areno.cli.funnel import _baselines

        stages = {
            "loaded": 16,
            "contract_valid": 15,
            "generated": 45,
            "length_valid": 45,
            "trainable_token_valid": 45,
            "trained": 45,
        }
        # online-RL fans out, so generated (45) is the global max baseline.
        self.assertEqual(_baselines(stages), 45)
        stages_sft = {
            "loaded": 64,
            "contract_valid": 60,
            "generated": None,
            "length_valid": None,
            "trainable_token_valid": 58,
            "trained": 58,
        }
        self.assertEqual(_baselines(stages_sft), 64)


# --------------------------------------------------------------------------
# CLI tests via CliRunner with synthetic on-disk funnel artifacts
# --------------------------------------------------------------------------


def _write_funnel_file(tmp_path, records, *, pid=4242):
    """Write one JSON line per record into sample_funnel.{pid}.jsonl."""
    path = tmp_path / f"sample_funnel.{pid}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _sft_record(step, loaded, valid, trained, *, drop=None, pid=4242):
    return {
        "step": step,
        "source": "sft",
        "pid": pid,
        "stages": {
            "loaded": loaded,
            "contract_valid": valid,
            "generated": None,
            "length_valid": None,
            "trainable_token_valid": trained,
            "trained": trained,
        },
        "drop_reasons": {"contract_valid": [drop]} if drop else {},
    }


def _rl_record(step, loaded, contract, generated, length, trained, *, drop=None, pid=4242):
    return {
        "step": step,
        "source": "online_rl",
        "pid": pid,
        "stages": {
            "loaded": loaded,
            "contract_valid": contract,
            "generated": generated,
            "length_valid": length,
            "trainable_token_valid": trained,
            "trained": trained,
        },
        "drop_reasons": {"loaded": ["prompt_too_long"]}
        if drop == "loaded"
        else ({"length_valid": ["over_context_len"]} if drop == "length" else {}),
    }


class FunnelCliTest(unittest.TestCase):
    """`areno funnel` reads synthetic artifacts and renders without sample text."""

    def test_sft_fixture_per_update_and_cumulative(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = __import__("pathlib").Path(tmp)
            _write_funnel_file(
                tmp_path,
                [_sft_record(0, 32, 30, 30), _sft_record(1, 32, 28, 28, drop="empty_or_over_budget")],
            )
            result = CliRunner().invoke(funnel_command, ["--metrics-log-dir", str(tmp_path)])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("step=0", result.output)
        self.assertIn("step=1", result.output)
        self.assertIn("Cumulative", result.output)
        # Cumulative trained = 30 + 28
        self.assertIn("58", result.output)
        # Drop reason surfaced under the contract-valid stage.
        self.assertIn("empty_or_over_budget", result.output)

    def test_online_rl_fixture_with_prompt_too_long_drop(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = __import__("pathlib").Path(tmp)
            _write_funnel_file(
                tmp_path,
                [
                    _rl_record(0, 16, 15, 45, 45, 45, drop="loaded"),
                    _rl_record(1, 16, 16, 48, 48, 48),
                ],
            )
            result = CliRunner().invoke(funnel_command, ["--metrics-log-dir", str(tmp_path), "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        report = json.loads(result.output)
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["per_update"][0]["stages"]["loaded"], 16)
        self.assertEqual(report["per_update"][0]["drop_reasons"]["loaded"], ["prompt_too_long"])
        # Cumulative trained = 45 + 48
        self.assertEqual(report["cumulative"]["stages"]["trained"], 93)

    def test_max_updates_zero_omits_per_update(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = __import__("pathlib").Path(tmp)
            _write_funnel_file(
                tmp_path,
                [_sft_record(0, 8, 8, 8), _sft_record(1, 8, 8, 8)],
            )
            result = CliRunner().invoke(
                funnel_command, ["--metrics-log-dir", str(tmp_path), "--max-updates", "0", "--json"]
            )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        report = json.loads(result.output)
        # max-updates=0 -> per_update omitted; cumulative still present.
        self.assertEqual(report["per_update"], [])
        self.assertIsNotNone(report["cumulative"])

    def test_max_updates_negative_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = __import__("pathlib").Path(tmp)
            _write_funnel_file(tmp_path, [_sft_record(0, 8, 8, 8)])
            result = CliRunner().invoke(funnel_command, ["--metrics-log-dir", str(tmp_path), "--max-updates", "-1"])
        # IntRange(min=0) rejects negatives before the body runs.
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("is not in the range", result.output)

    def test_reconcile_warns_on_backwards_counts(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = __import__("pathlib").Path(tmp)
            # trained (90) exceeds length_valid (88) -> must warn.
            _write_funnel_file(tmp_path, [_rl_record(0, 32, 30, 90, 88, 90)])
            result = CliRunner().invoke(funnel_command, ["--metrics-log-dir", str(tmp_path)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("warn:", result.output)
        self.assertIn("exceeds", result.output)

    def test_malformed_json_line_warns_with_line_number(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = __import__("pathlib").Path(tmp)
            path = tmp_path / "sample_funnel.4242.jsonl"
            path.write_text(
                json.dumps(_sft_record(0, 8, 8, 8)) + "\n"
                "{not valid json\n" + json.dumps(_sft_record(1, 8, 8, 8)) + "\n",
                encoding="utf-8",
            )
            result = CliRunner().invoke(funnel_command, ["--metrics-log-dir", str(tmp_path), "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        report = json.loads(result.output)
        self.assertEqual(report["records"], 2)  # two valid records parsed
        self.assertTrue(any("line 2" in w for w in report["load_warnings"]))

    def test_missing_dir_raises_click_exception(self):
        result = CliRunner().invoke(funnel_command, ["--metrics-log-dir", "/nonexistent/areno/funnel/dir"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIsInstance(result.exception, SystemExit)
        self.assertIn("not found", result.output)

    def test_never_prints_sample_contents(self):
        """A stray prompt field on disk must never reach the terminal."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = __import__("pathlib").Path(tmp)
            path = tmp_path / "sample_funnel.4242.jsonl"
            record = _sft_record(0, 8, 8, 8)
            record["prompt"] = "SECRET_PROMPT_SHOULD_NOT_PRINT"  # sabotage
            record["completion"] = "SECRET_COMPLETION_TOO"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            result = CliRunner().invoke(funnel_command, ["--metrics-log-dir", str(tmp_path)])
            result_json = CliRunner().invoke(funnel_command, ["--metrics-log-dir", str(tmp_path), "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(result_json.exit_code, 0, msg=result_json.output)
        self.assertNotIn("SECRET_PROMPT_SHOULD_NOT_PRINT", result.output)
        self.assertNotIn("SECRET_COMPLETION_TOO", result.output)
        self.assertNotIn("SECRET_PROMPT_SHOULD_NOT_PRINT", result_json.output)

    def test_top_level_cli_lists_funnel(self):
        """`areno --help` must list the new `funnel` subcommand."""
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("funnel", result.output)

    def test_pid_selects_specific_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = __import__("pathlib").Path(tmp)
            _write_funnel_file(tmp_path, [_sft_record(0, 8, 8, 8, pid=111)], pid=111)
            _write_funnel_file(tmp_path, [_sft_record(0, 4, 4, 4, pid=222)], pid=222)
            result = CliRunner().invoke(funnel_command, ["--metrics-log-dir", str(tmp_path), "--pid", "222", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        report = json.loads(result.output)
        self.assertEqual(report["pid"], 222)
        self.assertEqual(report["cumulative"]["stages"]["trained"], 4)


if __name__ == "__main__":
    unittest.main()
