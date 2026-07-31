"""Evaluate player policies on Battleship fleets."""

from __future__ import annotations

# 导入标准库
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Protocol

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


# 玩家策略协议接口
class Player(Protocol):
    """Protocol for a player policy."""

    # 选择射击坐标的接口方法
    def choose_shot(self, state: game.GameState) -> str:
        """Choose a coordinate to fire at given the current state."""
        ...


# 随机玩家实现：基于随机选择的baseline
class RandomPlayer:
    """Random baseline player."""

    # 每次选择随机合法射击坐标
    def choose_shot(self, state: game.GameState) -> str:
        # 获取所有合法射击位置
        legal = game.legal_shots(state)
        # 理论上不应出现没有合法位置的情况
        if not legal:
            return "A1"  # Should not happen in practice
        # 从合法位置中随机选择一个
        row, col = random.choice(list(legal))
        # 转换为坐标字符串
        return game.format_coordinate(row, col)


# 确定性假玩家：用于测试，按固定顺序射击
class FakePlayer:
    """Deterministic fake player for testing: picks first legal shot each turn."""

    # 初始化，接受可选的自定义射击序列
    def __init__(self, sequence: list[tuple[int, int]] | None = None):
        self.sequence = sequence or [(0, 0), (0, 1), (1, 0), (1, 1)]  # Simple pattern
        # 当前序列索引
        self.idx = 0

    # 按预设序列选择射击坐标
    def choose_shot(self, state: game.GameState) -> str:
        # 获取合法射击位置
        legal = game.legal_shots(state)
        if not legal:
            return "A1"

        # 如果序列中还有未使用的坐标
        if self.idx < len(self.sequence):
            shot = self.sequence[self.idx]
            self.idx += 1
            # 检查该坐标是否仍然合法
            if shot in legal:
                return game.format_coordinate(shot[0], shot[1])

        # 回退到第一个合法位置
        # Fallback to first legal
        row, col = sorted(legal)[0]
        return game.format_coordinate(row, col)


class HeuristicPlayer:
    """Heuristic player using hunt/target strategy (same as web_ui.py)."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def choose_shot(self, state: game.GameState) -> str:
        legal = game.legal_shots(state)
        if not legal:
            return "A1"

        # Target phase: collect cells adjacent to unresolved hits.
        hit_cells = set()
        sunk_cells = set()
        for ship in state.ships:
            if ship.is_sunk:
                for cell in ship.cells:
                    sunk_cells.add(cell)
            else:
                for cell in ship.cells:
                    if cell in ship.hits:
                        hit_cells.add(cell)

        open_hits = hit_cells - sunk_cells
        candidates = []
        for hit in open_hits:
            r, c = hit
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n = (r + dr, c + dc)
                if n in legal:
                    candidates.append(n)

        if candidates:
            row, col = self.rng.choice(candidates)
            return game.format_coordinate(row, col)

        # Hunt phase: prefer checkerboard spread.
        spread = [cell for cell in legal if (cell[0] + cell[1]) % 2 == 0]
        pool = spread if spread else list(legal)
        row, col = self.rng.choice(pool)
        return game.format_coordinate(row, col)


# 评估玩家策略在给定舰队列表上的表现
def evaluate_player(player: Player, fleets: list[dict], max_turns: int | None = None) -> dict:
    """Evaluate a player policy against a list of fleet records."""
    # 默认使用游戏最大回合数
    if max_turns is None:
        max_turns = game.MAX_TURNS

    # 存储每局的结果
    results = []
    # 遍历每个舰队布局
    for fleet in fleets:
        # 初始化游戏状态
        state = game.init_state(fleet)
        # 记录无效射击次数
        invalid_count = 0

        # 游戏主循环：直到游戏结束或达到最大回合数
        while not game.is_terminal(state) and state.shots_used < max_turns:
            # 让玩家选择射击坐标
            coord = player.choose_shot(state)
            # 执行射击
            result = game.fire(state, coord)
            # 统计无效射击
            if result.get("status") == "invalid":
                invalid_count += 1

        # 从状态直接计算得分
        # Compute score directly from state
        total_hits = sum(len(s.hits) for s in state.ships)
        sunk_count = sum(1 for s in state.ships if s.is_sunk)

        # 记录该局的结果
        results.append({
            "fleet_id": fleet.get("id", "unknown"),
            "win": game.is_win(state),
            "completion": total_hits / game.TOTAL_SHIP_CELLS,
            "shots_used": state.shots_used,
            "hits": total_hits,
            "sunk_ships": sunk_count,
            "invalid_shots": invalid_count,
        })

    # 聚合统计
    # Aggregate
    wins = sum(1 for r in results if r["win"])
    total = len(results)
    completion_rates = [r["completion"] for r in results]
    shots_to_win = [r["shots_used"] for r in results if r["win"]]

    # 返回汇总结果
    return {
        "player": player.__class__.__name__,
        "total_fleets": total,
        "wins": wins,
        "win_rate": wins / total if total > 0 else 0.0,
        "completion_mean": sum(completion_rates) / total if total > 0 else 0.0,
        "completion_std": _std(completion_rates) if len(completion_rates) > 1 else 0.0,
        "shots_to_win_mean": sum(shots_to_win) / len(shots_to_win) if shots_to_win else max_turns,
        "results": results,
    }


# 计算样本标准差的内部函数
def _std(values: list[float]) -> float:
    """Compute sample standard deviation."""
    # 数据点少于2个时无法计算标准差
    if len(values) < 2:
        return 0.0
    # 计算均值
    mean = sum(values) / len(values)
    # 计算方差（使用 n-1 作为分母，即样本方差）
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    # 返回标准差（方差的平方根）
    return variance ** 0.5


# 从 JSONL 文件加载舰队记录
def load_fleets(path: str) -> list[dict]:
    """Load fleet records from JSONL."""
    # 展开路径为绝对路径
    fleets_path = Path(path).expanduser()
    # 如果是目录，则查找 fleets.jsonl 文件
    if fleets_path.is_dir():
        fleets_path = fleets_path / "fleets.jsonl"

    # 读取并解析 JSONL 文件
    fleets = []
    with fleets_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                fleets.append(json.loads(line))
    return fleets


# 命令行入口函数
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Evaluate player policies on Battleship fleets.")
    # 添加舰队路径必需参数
    parser.add_argument("--fleets", "-f", required=True, help="Path to fleets JSONL file or directory.")
    # 添加玩家类型选择参数
    parser.add_argument(
        "--player",
        "-p",
        choices=["random", "fake", "heuristic"],
        default="random",
        help="Player policy to evaluate.",
    )
    # 添加输出路径参数
    parser.add_argument("--output", "-o", default="-", help="Output JSON path, or '-' for stdout.")
    # 添加最大回合数参数
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum turns per game.")
    args = parser.parse_args()

    # 加载舰队数据
    # Load fleets
    fleets = load_fleets(args.fleets)
    # 如果没有加载到舰队，打印错误并退出
    if not fleets:
        print("No fleets found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(fleets)} fleets, evaluating {args.player} player...")

    # 根据参数创建对应的玩家
    # Create player
    if args.player == "random":
        player = RandomPlayer()
    elif args.player == "fake":
        player = FakePlayer()
    elif args.player == "heuristic":
        player = HeuristicPlayer(seed=42)
    else:
        raise ValueError(f"Unknown player: {args.player}")

    # 运行评估
    # Evaluate
    results = evaluate_player(player, fleets, args.max_turns)

    # 打印汇总结果
    # Print summary
    print("\n" + "=" * 60)
    print(f"Player: {results['player']}")
    print(f"Total fleets: {results['total_fleets']}")
    print(f"Wins: {results['wins']} ({results['win_rate']:.1%})")
    print(f"Completion rate: {results['completion_mean']:.3f} +/- {results['completion_std']:.3f}")
    print(f"Shots to win (mean): {results['shots_to_win_mean']:.1f}")
    print("=" * 60)

    # 输出 JSON 结果
    # Output JSON
    if args.output == "-":
        # 打印到标准输出
        print(json.dumps(results, indent=2))
    else:
        # 写入指定文件
        output_path = Path(args.output).expanduser()
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {output_path}")


# 如果直接运行此脚本，则调用 main 函数
if __name__ == "__main__":
    main()