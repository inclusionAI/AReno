"""CPU regression tests for reward-function determinism.

Issue context: "check whether reward functions return different results for the
same input". Each test here builds a fixed :class:`RewardRecord`-shaped object,
calls the example ``reward_fn`` N times, and asserts all return values are
bit-for-bit equal. This guards against future regressions if someone introduces
randomness, time-based logic, or unordered iteration into a reward path.

These tests are intentionally side-effect-free: they do not modify any reward
implementation, only exercise it.

Note: ComputeGroupAdvantagesDeterminismTest loads ``compute_group_advantages``
by file path (``_load_rewards_module``) so it does not require ``pip install -e .``.
All other tests load example reward functions by file path too.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

# ── Helper: load a plain Python file without installing its package ──────────
# Used by ComputeGroupAdvantagesDeterminismTest so it can import
# `compute_group_advantages` without a full `pip install -e .`.
_REWARDS_MODULE: object | None = None


def _load_rewards_module():
    """Lazily load ``areno/api/rewards.py`` by file path."""
    global _REWARDS_MODULE
    if _REWARDS_MODULE is not None:
        return _REWARDS_MODULE
    path = REPO_ROOT / "areno" / "api" / "rewards.py"
    spec = importlib.util.spec_from_file_location("_det_rewards", path)
    assert spec is not None and spec.loader is not None
    _REWARDS_MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_REWARDS_MODULE)
    return _REWARDS_MODULE

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
N_CALLS = 10  # Same input, repeated calls; reward must be stable.


def _can_import(module_name: str) -> bool:
    """Return True iff ``module_name`` can be imported in the current env."""

    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def _load_reward(example_dir: Path, module_name: str = "reward"):
    """Load an example module by file path, mirroring the existing test pattern.

    The agentic examples do ``sys.path.insert(0, parent)`` then ``import game``
    at import time, so we temporarily put their directory on ``sys.path`` and
    clean up ``sys.modules["game"]`` afterwards to avoid cross-example bleed.
    Also cleans up any ``det_test_*`` modules left by previous calls so that
    re-importing the same example (e.g. shopping ``game`` + ``reward``) does
    not return a stale cached module.
    """

    path = example_dir / f"{module_name}.py"
    previous_game = sys.modules.pop("game", None)
    previous_dataset_gen = sys.modules.pop("dataset_generator", None)
    # Remove any det_test_* modules from prior _load_reward calls to prevent
    # stale cached modules when the same example dir is loaded under different
    # module_name (e.g. shopping/reward.py and shopping/game.py).
    stale_det = [k for k in sys.modules if k.startswith("det_test_")]
    for k in stale_det:
        del sys.modules[k]
    sys.path.insert(0, str(example_dir))
    try:
        spec = importlib.util.spec_from_file_location(f"det_test_{example_dir.name}_{module_name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(example_dir))
        sys.modules.pop("game", None)
        sys.modules.pop("dataset_generator", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game
        if previous_dataset_gen is not None:
            sys.modules["dataset_generator"] = previous_dataset_gen


def _assert_reward_stable(test_case: unittest.TestCase, reward_fn, record, expected: float | None = None):
    """Call ``reward_fn`` N times on the same record and assert equality.

    Each call result is validated as a numeric type before conversion to float,
    guarding against reward functions that accidentally return ``None`` or
    non-numeric values.

    On failure the full list of returned values is included in the assertion
    message so the developer can see *which* call(s) deviated.
    """

    outputs = []
    for i in range(N_CALLS):
        result = reward_fn(record)
        test_case.assertIsInstance(
            result, (int, float),
            f"reward_fn returned non-numeric type: {type(result)} at call {i}",
        )
        outputs.append(float(result))
    first = outputs[0]
    for i, value in enumerate(outputs[1:], 1):
        test_case.assertEqual(
            value, first,
            f"Call {i} returned {value}, call 0 returned {first}; "
            f"all {N_CALLS} outputs: {outputs}",
        )
    if expected is not None:
        test_case.assertEqual(first, expected, f"Expected {expected}, got {first}; all outputs: {outputs}")


class MathRewardDeterminismTest(unittest.TestCase):
    """math_verify reward: same (answer, completion) pair must give same score."""

    @unittest.skipUnless(
        _can_import("math_verify"),
        "math_verify not installed in this env; skip math reward determinism test",
    )
    def test_math_verify_reward_is_deterministic(self):
        """Same ground-truth and completion should always yield the same reward."""

        reward = _load_reward(EXAMPLES / "math", module_name="math_verify_reward")
        record = SimpleNamespace(answer=["42"], completion="\\boxed{42}")
        _assert_reward_stable(self, reward.reward_fn, record, expected=1.0)

        record_wrong = SimpleNamespace(answer=["42"], completion="\\boxed{7}")
        _assert_reward_stable(self, reward.reward_fn, record_wrong, expected=0.0)


class TicTacToeRewardDeterminismTest(unittest.TestCase):
    """Tic-Tac-Toe tool-call reward: minimax-based scoring must be stable."""

    def test_tool_reward_win_move_is_deterministic(self):
        """A winning move should always score 1.0 across repeated calls."""

        reward = _load_reward(EXAMPLES / "agentic" / "tictactoe")
        board = [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]]
        record = SimpleNamespace(
            source_record={"board": board},
            completion="<move>3</move>",
            tool_calls=[{"name": "choose_square", "arguments": {"square": 3}}],
        )
        _assert_reward_stable(self, reward.reward_fn, record, expected=1.0)

    def test_tool_reward_illegal_move_is_deterministic(self):
        """An illegal move should always score -1.0 across repeated calls."""

        reward = _load_reward(EXAMPLES / "agentic" / "tictactoe")
        board = [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]]
        record = SimpleNamespace(
            source_record={"board": board},
            completion="<move>1</move>",
            tool_calls=[{"name": "choose_square", "arguments": {"square": 1}}],
        )
        _assert_reward_stable(self, reward.reward_fn, record, expected=-1.0)


class TicTacToeNoToolRewardDeterminismTest(unittest.TestCase):
    """Tic-Tac-Toe XML no-tool reward: regex parsing + scoring must be stable."""

    def test_xml_reward_move_tag_is_deterministic(self):
        """A tagged winning move should always score 1.0 across repeated calls."""

        reward = _load_reward(EXAMPLES / "agentic" / "tictactoe", module_name="reward_no_tool")
        board = [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]]
        record = SimpleNamespace(
            source_record={"board": board},
            completion="<move>3</move>",
        )
        _assert_reward_stable(self, reward.reward_fn, record, expected=1.0)

    def test_xml_reward_missing_tag_is_deterministic(self):
        """A response without a move tag should always score -1.0."""

        reward = _load_reward(EXAMPLES / "agentic" / "tictactoe", module_name="reward_no_tool")
        board = [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]]
        record = SimpleNamespace(source_record={"board": board}, completion="3")
        _assert_reward_stable(self, reward.reward_fn, record, expected=-1.0)


@unittest.skipUnless(
    sys.version_info >= (3, 10),
    "codebreaker game.py uses zip(strict=True), requires Python 3.10+",
)
class CodebreakerRewardDeterminismTest(unittest.TestCase):
    """Codebreaker reward: bulls-and-cows scoring must be stable per episode."""

    def test_optimal_guess_is_deterministic(self):
        """Solving on the first guess should always score the same."""

        reward = _load_reward(EXAMPLES / "agentic" / "codebreaker")
        source = {"secret": "0123", "max_guesses": 6}
        record = SimpleNamespace(
            source_record=source,
            tool_calls=[{"name": "guess_code", "arguments": json.dumps({"code": "0123"})}],
        )
        _assert_reward_stable(self, reward.reward_fn, record, expected=1.0)

    def test_repeated_guess_is_deterministic(self):
        """Repeated guesses should always trigger the -0.5 duplicate penalty."""

        reward = _load_reward(EXAMPLES / "agentic" / "codebreaker")
        source = {"secret": "0123", "max_guesses": 6}
        record = SimpleNamespace(
            source_record=source,
            tool_calls=[
                {"name": "guess_code", "arguments": json.dumps({"code": "0145"})},
                {"name": "guess_code", "arguments": json.dumps({"code": "0145"})},
            ],
        )
        _assert_reward_stable(self, reward.reward_fn, record, expected=-0.5)


class CodingRewardDeterminismTest(unittest.TestCase):
    """Agentic coding reward: test-pass + submit ordering must be stable."""

    def test_solved_with_tests_is_deterministic(self):
        """A trajectory that passes tests and submits solved should score 1.0."""

        reward = _load_reward(EXAMPLES / "agentic" / "coding")
        record = SimpleNamespace(
            source_record={"test_commands": ["pytest tests/"]},
            tool_calls=[
                {"name": "apply_patch", "arguments": "{}"},
                {"name": "submit", "arguments": json.dumps({"status": "solved"})},
            ],
            tool_results=[
                {"content": json.dumps({"command": "pytest tests/", "returncode": 0})},
            ],
        )
        _assert_reward_stable(self, reward.reward_fn, record, expected=1.0)

    def test_solved_without_patch_is_deterministic(self):
        """Submitting solved without tests or patch should always score -0.5."""

        reward = _load_reward(EXAMPLES / "agentic" / "coding")
        # Empty test_commands: all_tests_passed=True, but `required_commands`
        # is falsy so the 0.5 branch is skipped; falls through to the -0.5
        # `submitted == "solved"` branch.
        record = SimpleNamespace(
            source_record={"test_commands": []},
            tool_calls=[
                {"name": "submit", "arguments": json.dumps({"status": "solved"})},
            ],
            tool_results=[],
        )
        _assert_reward_stable(self, reward.reward_fn, record, expected=-0.5)


class DuelGridRewardDeterminismTest(unittest.TestCase):
    """DuelGrid reward: step-turn scoring over a fixed state must be stable."""

    def test_legal_attack_is_deterministic(self):
        """A legal ATTACK on an adjacent opponent should score the same each time."""

        reward = _load_reward(EXAMPLES / "agentic" / "duelgrid")
        # Adjacent A and U on a 3x3 grid so ATTACK is legal.
        source = {
            "state": {
                "map": ["###", "#AU", "###"],
                "agent_hp": 10,
                "user_hp": 10,
                "agent_energy": 2,
                "user_energy": 2,
                "agent_max_energy": 3,
                "user_max_energy": 3,
                "turn": 0,
                "max_turns": 40,
            }
        }
        record = SimpleNamespace(
            source_record=source,
            completion="",
            tool_calls=[
                {"name": "choose_action", "arguments": {"actions": [{"action": "ATTACK", "direction": "RIGHT"}]}},
            ],
        )
        _assert_reward_stable(self, reward.reward_fn, record)

    def test_illegal_action_is_deterministic(self):
        """An illegal action should always produce the same -0.3-style penalty."""

        reward = _load_reward(EXAMPLES / "agentic" / "duelgrid")
        source = {
            "state": {
                "map": ["###", "#AU", "###"],
                "agent_hp": 10,
                "user_hp": 10,
                "agent_energy": 2,
                "user_energy": 2,
                "agent_max_energy": 3,
                "user_max_energy": 3,
                "turn": 0,
                "max_turns": 40,
            }
        }
        record = SimpleNamespace(
            source_record=source,
            completion="",
            tool_calls=[
                {"name": "choose_action", "arguments": {"actions": [{"action": "WAIT"}]}},
            ],
        )
        _assert_reward_stable(self, reward.reward_fn, record)


class ShoppingRewardDeterminismTest(unittest.TestCase):
    """Shopping reward: bundle scoring over a fixed catalog must be stable."""

    def test_no_submit_is_deterministic(self):
        """Missing submit_bundle should always score -1.0."""

        reward = _load_reward(EXAMPLES / "agentic" / "shopping")
        source = {
            "kit_name": "rain commute",
            "categories": ["jacket", "bottle"],
            "budget": 140,
            "required_features_by_category": {
                "jacket": ["waterproof", "packable"],
                "bottle": ["insulated", "leakproof"],
            },
        }
        record = SimpleNamespace(source_record=source, tool_calls=[])
        _assert_reward_stable(self, reward.reward_fn, record, expected=-1.0)

    def test_best_bundle_with_full_protocol_is_deterministic(self):
        """Submitting the best bundle with the canonical 4-tool protocol should score 1.0.

        ``game.best_bundle()`` internally uses ``max(scored, key=lambda x: x[0])``
        which returns the first maximum in Python, making the result deterministic
        even when multiple bundles share the same score.
        """

        reward = _load_reward(EXAMPLES / "agentic" / "shopping")
        game = _load_reward(EXAMPLES / "agentic" / "shopping", module_name="game")
        source = {
            "kit_name": "rain commute",
            "categories": ["jacket", "bottle"],
            "budget": 140,
            "required_features_by_category": {
                "jacket": ["waterproof", "packable"],
                "bottle": ["insulated", "leakproof"],
            },
        }
        best = game.best_bundle(source)
        record = SimpleNamespace(
            source_record=source,
            tool_calls=[
                {"name": "search_catalog", "arguments": json.dumps({"categories": best, "max_price": 140})},
                {"name": "inspect_items", "arguments": json.dumps({"item_ids": best})},
                {"name": "check_kit", "arguments": json.dumps({"item_ids": best})},
                {"name": "submit_bundle", "arguments": json.dumps({"item_ids": best})},
            ],
        )
        _assert_reward_stable(self, reward.reward_fn, record, expected=1.0)


@unittest.skipUnless(
    sys.version_info >= (3, 10),
    "areno.api uses dataclass(slots=True), requires Python 3.10+",
)
@unittest.skipUnless(
    sys.version_info >= (3, 10),
    "areno.api uses dataclass(slots=True), requires Python 3.10+",
)
class ComputeGroupAdvantagesDeterminismTest(unittest.TestCase):
    """Core helper: ``compute_group_advantages`` must be stable for the same rewards."""

    def test_advantages_are_deterministic(self):
        """The same reward list should produce the same advantage list each call."""

        rewards_mod = _load_rewards_module()
        compute_group_advantages = rewards_mod.compute_group_advantages

        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        outputs = [compute_group_advantages(rewards) for _ in range(N_CALLS)]
        first = outputs[0]
        for value in outputs[1:]:
            self.assertEqual(value, first)
        # Sanity: the first call already matches the documented normalisation.
        # float32 has ~7 significant digits; delta=1e-4 is safe across hardware.
        self.assertAlmostEqual(first[0], -1.4142135, delta=1e-4)

    def test_advantages_constant_rewards_are_deterministic(self):
        """Constant rewards should always produce all-zero advantages."""

        rewards_mod = _load_rewards_module()
        compute_group_advantages = rewards_mod.compute_group_advantages

        outputs = [compute_group_advantages([3.0, 3.0, 3.0]) for _ in range(N_CALLS)]
        first = outputs[0]
        for value in outputs[1:]:
            self.assertEqual(value, first)
        self.assertEqual(first, [0.0, 0.0, 0.0])


class MultiTurnCodingDeterminismTest(unittest.TestCase):
    """Multi-turn coding trajectory: many tool calls should still be stable."""

    def test_multi_turn_coding_is_deterministic(self):
        """A 10-turn coding trajectory should always produce the same reward."""

        reward = _load_reward(EXAMPLES / "agentic" / "coding")
        tool_calls = []
        tool_results = []
        for i in range(5):
            tool_calls.append({"name": "apply_patch", "arguments": "{}"})
            tool_results.append(
                {"content": json.dumps({"command": f"pytest tests/{i}", "returncode": 0})}
            )
        # Final submit
        tool_calls.append({"name": "submit", "arguments": json.dumps({"status": "solved"})})
        tool_results.append({"content": ""})

        record = SimpleNamespace(
            source_record={"test_commands": [f"pytest tests/{i}" for i in range(5)]},
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        _assert_reward_stable(self, reward.reward_fn, record, expected=1.0)


class EmptyRecordDeterminismTest(unittest.TestCase):
    """Boundary test: reward functions must be stable on minimal/empty records
    that still satisfy their internal key requirements."""

    def test_coding_reward_empty_record_is_deterministic(self):
        """An empty source_record with no tool_calls should always score -1.0."""
        reward = _load_reward(EXAMPLES / "agentic" / "coding")
        record = SimpleNamespace(source_record={}, tool_calls=[], tool_results=[])
        _assert_reward_stable(self, reward.reward_fn, record, expected=-1.0)

    def test_shopping_reward_empty_record_is_deterministic(self):
        """An empty source_record with no tool_calls should always score -1.0."""
        reward = _load_reward(EXAMPLES / "agentic" / "shopping")
        record = SimpleNamespace(source_record={}, tool_calls=[])
        _assert_reward_stable(self, reward.reward_fn, record, expected=-1.0)


class LargeScaleGeneratedDeterminismTest(unittest.TestCase):
    """Use dataset generators to create many test cases and verify each is deterministic.

    Each generator (shopping, tictactoe, codebreaker, duelgrid) produces N records
    with the same seed, ensuring reproducibility. For each generated record we call
    the reward function twice and confirm the outputs match.
    """

    LARGE_N = 100  # Number of generated test cases per category

    def test_shopping_generated_is_deterministic(self):
        """100 generated shopping tasks should all produce deterministic rewards."""

        # Import generator dynamically
        sys.path.insert(0, str(EXAMPLES / "agentic" / "shopping"))
        try:
            import dataset_generator
        finally:
            sys.path.remove(str(EXAMPLES / "agentic" / "shopping"))
            sys.modules.pop("dataset_generator", None)

        # Generate 100 shopping tasks with fixed seed
        records = dataset_generator.generate_records(self.LARGE_N, seed=2026)
        reward = _load_reward(EXAMPLES / "agentic" / "shopping")

        for rec in records:
            record = SimpleNamespace(source_record=rec, tool_calls=[])
            _assert_reward_stable(self, reward.reward_fn, record)

    def test_tictactoe_generated_is_deterministic(self):
        """100 generated Tic-Tac-Toe boards should all produce deterministic rewards."""

        # First import game to ensure constants are loaded
        sys.path.insert(0, str(EXAMPLES / "agentic" / "tictactoe"))
        try:
            import game as tictactoe_game
            import dataset_generator
        finally:
            sys.path.remove(str(EXAMPLES / "agentic" / "tictactoe"))
            sys.modules.pop("game", None)
            sys.modules.pop("dataset_generator", None)

        records = dataset_generator.generate_records(self.LARGE_N, seed=2026)
        reward = _load_reward(EXAMPLES / "agentic" / "tictactoe")

        for rec in records:
            record = SimpleNamespace(
                source_record={"board": rec["board"]},
                completion="<move>1</move>",
                tool_calls=[{"name": "choose_square", "arguments": {"square": 1}}],
            )
            # Just verify it's deterministic (score can vary by board)
            _assert_reward_stable(self, reward.reward_fn, record)

    @unittest.skipUnless(sys.version_info >= (3, 10), "codebreaker uses zip(strict=True)")
    def test_codebreaker_generated_is_deterministic(self):
        """100 generated codebreaker tasks should all produce deterministic rewards."""

        sys.path.insert(0, str(EXAMPLES / "agentic" / "codebreaker"))
        try:
            import game as cb_game
            import dataset_generator
        finally:
            sys.path.remove(str(EXAMPLES / "agentic" / "codebreaker"))
            sys.modules.pop("game", None)
            sys.modules.pop("dataset_generator", None)

        records = dataset_generator.generate_records(self.LARGE_N, seed=2026)
        reward = _load_reward(EXAMPLES / "agentic" / "codebreaker")

        for rec in records:
            record = SimpleNamespace(
                source_record={"secret": rec["secret"], "max_guesses": 6},
                tool_calls=[{"name": "guess_code", "arguments": json.dumps({"code": rec["secret"]})}],
            )
            _assert_reward_stable(self, reward.reward_fn, record, expected=1.0)

    def test_duelgrid_generated_is_deterministic(self):
        """100 generated duelgrid tasks should all produce deterministic rewards."""

        sys.path.insert(0, str(EXAMPLES / "agentic" / "duelgrid"))
        try:
            import game as dg_game
            import dataset_generator
        finally:
            sys.path.remove(str(EXAMPLES / "agentic" / "duelgrid"))
            sys.modules.pop("game", None)
            sys.modules.pop("dataset_generator", None)

        records = dataset_generator.generate_records(self.LARGE_N, seed=2026)
        reward = _load_reward(EXAMPLES / "agentic" / "duelgrid")

        for rec in records:
            # Wrap record in "state" key as expected by duelgrid reward
            state_dict = {
                "map": rec["map"],
                "agent_hp": rec["agent_hp"],
                "user_hp": rec["user_hp"],
                "agent_energy": rec["agent_energy"],
                "user_energy": rec["user_energy"],
                "agent_max_energy": rec["agent_max_energy"],
                "user_max_energy": rec["user_max_energy"],
                "turn": rec["turn"],
                "max_turns": rec["max_turns"],
            }
            record = SimpleNamespace(
                source_record={"state": state_dict},
                completion="",
                tool_calls=[{"name": "choose_action", "arguments": {"actions": [rec["best_action"]]}}],
            )
            _assert_reward_stable(self, reward.reward_fn, record)


if __name__ == "__main__":
    unittest.main()
