"""Dataset loader for the Battleship agentic example."""

from __future__ import annotations

# 导入标准库
import json
import sys
from pathlib import Path

# 添加当前目录到 Python 路径，以便导入同级目录的模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


# Areno 训练数据加载入口函数
def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL fleets and convert them to Areno prompt records."""
    # 忽略未使用的 default_loader 参数
    del default_loader
    # 从指定路径加载舰队记录
    records = _load_records(dataset_path)
    # 将每条原始记录格式化为 Areno 需要的 prompt 记录
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


# 从文件或目录加载舰队记录的内部函数
def _load_records(dataset_path: str) -> list[dict]:
    # 将路径展开为绝对路径
    path = Path(dataset_path).expanduser()
    # 如果是目录，则默认查找 fleets.jsonl 文件
    if path.is_dir():
        path = path / "fleets.jsonl"
    # 如果文件不存在，则自动生成记录
    if not path.exists():
        return dataset_generator.generate_records()

    # 读取 JSONL 文件，每行解析为一个字典
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


# 将原始舰队记录格式化为 Areno 训练所需的字典结构
def _format_record(raw: dict, index: int) -> dict:
    # 验证舰队布局是否合法
    game.normalize_record(raw)
    # 初始化游戏状态
    state = game.init_state(raw)
    # 生成发送给 agent 的 prompt
    prompt = game.format_prompt(raw)

    # 构建返回的记录字典，包含 id、prompt、种子、舰队信息等
    return {
        "id": raw.get("id", f"fleet-{index:05d}"),
        "prompt": prompt,
        "seed": raw.get("seed", 0),
        "ships": raw.get("ships", []),
        "ship_lengths": raw.get("ship_lengths", list(game.SHIPS)),
        "grid_size": raw.get("grid_size", game.GRID),
    }