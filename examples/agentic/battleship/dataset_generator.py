"""Generate Battleship fleets for the agentic example."""

from __future__ import annotations

# 导入标准库
import argparse
import json
import random
import sys
from pathlib import Path
from typing import TextIO

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

# 默认生成的舰队数量
DEFAULT_COUNT = 128
# 默认随机种子，确保可复现
DEFAULT_SEED = 2026


# 生成指定数量的可复现合法舰队布局
def generate_records(count: int = DEFAULT_COUNT, *, seed: int = DEFAULT_SEED) -> list[dict]:
    """Generate reproducible legal fleet placements."""
    # 创建随机数生成器，使用指定种子确保可复现性
    rng = random.Random(seed)
    # 存储生成的舰队记录
    records: list[dict] = []
    # 用于去重的已见集合，以船只位置元组为键
    seen: set[tuple[tuple[int, int], ...]] = set()
    # 尝试次数计数器
    attempts = 0

    # 循环生成直到达到目标数量
    while len(records) < count:
        attempts += 1
        # 如果尝试次数过多则放弃
        if attempts > count * 100:
            raise RuntimeError("could not generate enough unique Battleship fleets")

        # 使用递增的种子，基于基础种子加上记录数和尝试次数
        # 翻译英文注释: Use increasing seeds based on the base seed + index
        fleet_seed = seed + len(records) + attempts
        try:
            # 调用 game 模块放置舰队
            record = game.place_fleet(fleet_seed)
        except RuntimeError:
            # 放置失败则跳过
            continue

        # 根据船只位置去重
        # 翻译英文注释: Deduplicate by ship positions
        key = tuple(tuple(c) for c in record["ships"])
        if key in seen:
            continue
        seen.add(key)

        # 将生成的舰队记录添加到结果列表
        records.append({
            "id": f"generated-{len(records):05d}",
            "seed": fleet_seed,
            "ships": record["ships"],
            "ship_lengths": record["ship_lengths"],
            "grid_size": record["grid_size"],
        })

    return records


# 将记录列表写入 JSONL 格式文件
def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""
    # 遍历每条记录，以紧凑格式写入
    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


# 命令行入口函数
def main() -> None:
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Generate JSONL fleets for the Areno Battleship agentic example.")
    # 添加输出路径参数，默认为标准输出
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    # 添加生成数量参数
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of fleets to generate.")
    # 添加随机种子参数
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    # 验证数量参数必须为正数
    if args.count <= 0:
        raise ValueError("--count must be positive")

    # 生成指定数量的舰队记录
    records = generate_records(args.count, seed=args.seed)

    # 根据输出参数决定写入目标
    if args.output == "-":
        # 写入标准输出
        write_jsonl(records, sys.stdout)
    else:
        # 写入指定文件路径
        output_path = Path(args.output).expanduser()
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 以 UTF-8 编码打开文件并写入
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


# 如果直接运行此脚本，则调用 main 函数
if __name__ == "__main__":
    main()