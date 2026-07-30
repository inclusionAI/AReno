"""Battleship game logic for agentic examples."""

# 导入标准库
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, List, Set, Tuple, Optional, Union

# 游戏常量定义
# Compact 8x8 board with fleet [4,3,2,2] = 11 cells
GRID = 8  # 棋盘边长（8x8）
SHIPS = (4, 3, 2, 2)  # 舰队各舰长度（4/3/2/2，共11格）
MAX_TURNS = 40  # 最大回合数（< 棋盘 64 格：让"穷举全盘"不可行，RL 才有提升空间）
TOTAL_SHIP_CELLS = sum(SHIPS)  # 船只总格子数（11）

# 坐标系统定义：A1..H8（字母 = 行 A-H，数字 = 列 1-8）
# Coordinate system: A1..H8 (letter = row A-H, number = column 1-8)
_COL_LETTERS = "ABCDEFGH"


# 船只数据类：包含船只的长度和位置，以及命中追踪
@dataclass
class Ship:
    """A single ship with its cells and hit tracking."""

    length: int  # 船只长度
    cells: list[tuple[int, int]] = field(default_factory=list)  # 船只占据的格子坐标 (row, col) 0-indexed
    hits: set[tuple[int, int]] = field(default_factory=set)  # 被命中的格子集合

    # 属性：判断船只是否被击沉
    @property
    def is_sunk(self) -> bool:
        return len(self.hits) == len(self.cells)


# 游戏状态数据类：单局游戏的状态
@dataclass
class GameState:
    """Mutable game state for a single episode."""

    ships: list[Ship] = field(default_factory=list)  # 船只列表
    shots_history: list[tuple[int, int]] = field(default_factory=list)  # 所有射击历史（按顺序）
    shots_used: int = 0  # 已使用射击次数
    grid_size: int = GRID  # 棋盘大小
    seed: int = 0  # 随机种子


# 解析坐标字符串为 (row, col) 形式（0-indexed）
def parse_coordinate(coord: str) -> Optional[Tuple[int, int]]:
    """Parse 'A1'..'H8' into (row, col) 0-indexed, or None if invalid."""
    # 去除空白并转为大写
    coord = coord.strip().upper()
    # 验证坐标格式
    if not coord or len(coord) < 2:
        return None
    # 获取字母部分（行）
    col_letter = coord[0]
    if col_letter not in _COL_LETTERS:
        return None
    # 解析数字部分（列）：从1-indexed转为0-indexed
    try:
        col = int(coord[1:]) - 1  # 1-indexed to 0-indexed
    except ValueError:
        return None
    # 获取行索引
    row = _COL_LETTERS.index(col_letter)
    # 验证范围
    if not (0 <= row < GRID and 0 <= col < GRID):
        return None
    return (row, col)


# 格式化坐标：将 (row, col) 0-indexed 转为字符串 'A1'..'H8'
def format_coordinate(row: int, col: int) -> str:
    """Format (row, col) 0-indexed into 'A1'..'H8'."""
    return f"{_COL_LETTERS[row]}{col + 1}"


def _get_neighbor_cells(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Get all cells adjacent to the given cells (including diagonals)."""
    neighbors = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue  # Skip the cell itself
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID and 0 <= nc < GRID:
                    neighbors.add((nr, nc))
    return neighbors


# 使用给定种子生成合法的舰队布局
def place_fleet(seed: int) -> dict:
    """Generate a seeded legal fleet placement. Returns dict with seed, ships, fleet_cells.

    Ships cannot overlap and cannot be adjacent (including diagonally),
    following standard Battleship game rules.
    """
    # 使用种子创建随机数生成器，确保可复现性
    rng = random.Random(seed)
    # 存储生成的船只
    ships: list[Ship] = []
    # 记录已占用的格子及其相邻格子（间隔缓冲区）
    occupied: set[tuple[int, int]] = set()
    # 记录禁止区域（occupied + 相邻格子）
    forbidden: set[tuple[int, int]] = set()

    # 依次放置每艘船
    for length in SHIPS:
        placed = False
        attempts = 0
        # 尝试放置船只，最多尝试5000次（增加限制避免无限循环）
        while not placed and attempts < 5000:
            attempts += 1
            # 随机选择水平或垂直放置
            horizontal = rng.choice([True, False])
            if horizontal:
                # 水平放置：计算最大列位置
                max_col = GRID - length
                row = rng.randint(0, GRID - 1)
                col = rng.randint(0, max_col)
                # 生成船只占据的格子列表
                cells = [(row, col + i) for i in range(length)]
            else:
                # 垂直放置：计算最大行位置
                max_row = GRID - length
                row = rng.randint(0, max_row)
                col = rng.randint(0, GRID - 1)
                cells = [(row + i, col) for i in range(length)]

            # 检查是否与已有船只冲突或相邻
            # 标准规则：船只不能重叠，也不能相邻（包括对角线）
            cells_set = set(cells)
            if not cells_set.intersection(forbidden):
                # 创建船只对象并添加到列表
                ship = Ship(length=length, cells=cells)
                ships.append(ship)
                # 更新已占用格子和禁止区域
                occupied.update(cells)
                forbidden.update(cells)
                forbidden.update(_get_neighbor_cells(cells))
                placed = True

        # 如果无法放置船只，抛出异常
        if not placed:
            raise RuntimeError(f"Failed to place ship of length {length} with seed {seed} after 5000 attempts")

    # 构建格子列表用于快速检查
    # Build fleet_cells list for easy checking
    fleet_cells = []
    for ship in ships:
        fleet_cells.extend(ship.cells)

    # 返回结果字典
    return {
        "seed": seed,
        "ships": [[c[0], c[1]] for ship in ships for c in ship.cells],  # flat list for serialization
        "ship_lengths": [ship.length for ship in ships],
        "grid_size": GRID,
    }


# 验证并规范化舰队记录
def normalize_record(raw: dict) -> dict:
    """Validate and normalize a fleet record."""
    # 获取船只格子列表和长度列表
    ships_flat = raw.get("ships", [])
    ship_lengths = raw.get("ship_lengths", list(SHIPS))

    # 验证格子数量是否匹配
    if len(ships_flat) != sum(ship_lengths):
        raise ValueError(f"Ship cell count mismatch: expected {sum(ship_lengths)}, got {len(ships_flat)}")

    # 检查所有格子是否在范围内且不重叠
    # Check all cells in range and non-overlapping
    occupied: set[tuple[int, int]] = set()
    for cell in ships_flat:
        r, c = cell[0], cell[1]
        # 验证格子在棋盘范围内
        if not (0 <= r < GRID and 0 <= c < GRID):
            raise ValueError(f"Cell out of range: {r},{c}")
        # 验证格子不重叠
        if (r, c) in occupied:
            raise ValueError(f"Overlapping cells: {r},{c}")
        occupied.add((r, c))

    return raw


# 从舰队记录初始化游戏状态
def init_state(record: dict) -> 'GameState':
    """Initialize a game state from a fleet record."""
    # 创建新的游戏状态
    state = GameState()
    # 设置种子
    state.seed = record.get("seed", 0)

    # 获取船只相关数据
    ships_flat = record.get("ships", [])
    ship_lengths = record.get("ship_lengths", list(SHIPS))

    # 根据长度列表重建船只
    idx = 0
    for length in ship_lengths:
        # 提取该船的所有格子坐标
        cells = [(ships_flat[i][0], ships_flat[i][1]) for i in range(idx, idx + length)]
        # 创建船只并添加到状态
        state.ships.append(Ship(length=length, cells=cells))
        # 更新索引
        idx += length

    # 验证总格子数
    # Verify total
    total_cells = sum(len(s.cells) for s in state.ships)
    assert total_cells == TOTAL_SHIP_CELLS, f"Expected {TOTAL_SHIP_CELLS} cells, got {total_cells}"

    return state


# 获取所有合法射击位置（尚未射击的格子）
def legal_shots(state: GameState) -> set[tuple[int, int]]:
    """Return all cells not yet shot at."""
    # 将射击历史转换为集合
    shot_set = set(state.shots_history)
    # 返回所有格子减去已射击的格子
    return {(r, c) for r in range(state.grid_size) for c in range(state.grid_size)} - shot_set


# 执行一次射击动作
def fire(state: 'GameState', coord: Union[str, Tuple[int, int]]) -> dict:
    """Execute a fire action. Returns dict with status and metadata.

    coord can be a string like "A1" or a tuple (row, col).
    Returns:
        - {"status": "invalid", "reason": "...", "shots_used": n}
        - {"status": "miss", "shots_used": n, "hit_cells": 0, "sunk_ships": n, "remaining": n}
        - {"status": "hit", "shots_used": n, "hit_cells": x, "sunk_ships": n, "remaining": n}
        - {"status": "sunk", "shots_used": n, "hit_cells": x, "sunk_ships": n, "remaining": n, "ship_sunk": length}
    """
    # 解析坐标（支持字符串如 "A1" 或元组 (row, col)）
    # Parse coordinate
    if isinstance(coord, str):
        parsed = parse_coordinate(coord)
    else:
        parsed = coord
        # 验证元组格式
        # Validate tuple format
        if not (isinstance(parsed, tuple) and len(parsed) == 2):
            parsed = None

    # 处理无效坐标
    if parsed is None:
        state.shots_used += 1
        return {"status": "invalid", "reason": "invalid coordinate", "shots_used": state.shots_used}

    # 分离出行列
    row, col = parsed

    # 检查是否在棋盘范围内
    # Check bounds
    if not (0 <= row < GRID and 0 <= col < GRID):
        state.shots_used += 1
        return {"status": "invalid", "reason": "out of range", "shots_used": state.shots_used}

    # 检查是否已射击过该位置
    # Check repeated shot
    if (row, col) in state.shots_history:
        state.shots_used += 1
        return {"status": "invalid", "reason": "already shot", "shots_used": state.shots_used}

    # 记录射击
    # Record the shot
    state.shots_history.append((row, col))
    state.shots_used += 1

    # 检查是否命中船只
    # Check for hit
    hit_ship_idx = None
    for idx, ship in enumerate(state.ships):
        if (row, col) in ship.cells:
            hit_ship_idx = idx
            # 记录命中
            ship.hits.add((row, col))
            break

    # 计算统计信息
    sunk_count = sum(1 for s in state.ships if s.is_sunk)
    remaining = sum(1 for s in state.ships if not s.is_sunk)
    total_hits = sum(len(s.hits) for s in state.ships)

    # 根据命中情况返回不同结果
    if hit_ship_idx is None:
        return {"status": "miss", "shots_used": state.shots_used, "hit_cells": total_hits, "sunk_ships": sunk_count, "remaining": remaining}
    elif state.ships[hit_ship_idx].is_sunk:
        return {"status": "sunk", "shots_used": state.shots_used, "hit_cells": total_hits, "sunk_ships": sunk_count, "remaining": remaining, "ship_sunk": state.ships[hit_ship_idx].length}
    else:
        return {"status": "hit", "shots_used": state.shots_used, "hit_cells": total_hits, "sunk_ships": sunk_count, "remaining": remaining}


# 检查是否获胜（所有船只都被击沉）
def is_win(state: GameState) -> bool:
    """Return True if all ships are sunk."""
    return all(ship.is_sunk for ship in state.ships)


# 检查游戏是否结束（获胜或达到回合上限）
def is_terminal(state: GameState) -> bool:
    """Return True if game is over (win or turn cap reached)."""
    return is_win(state) or state.shots_used >= MAX_TURNS


# 为 agent 构建系统提示和用户提示
def format_prompt(record: Optional[dict] = None) -> str:
    """Build the system+user prompt for the agent."""
    # 系统提示：游戏规则说明
    sys_prompt = (
        "You are playing Battleship. Use the fire(coordinate) tool to sink all ships.\n\n"
        "Rules:\n"
        "- The grid is 8x8, columns numbered 1-8, rows labeled A-H.\n"
        "- Coordinates are given as letter+number, e.g., A1 (top-left), H8 (bottom-right).\n"
        "- The fire tool returns:\n"
        "  - 'miss': no ship at that coordinate.\n"
        "  - 'hit': a ship was hit but not sunk.\n"
        "  - 'sunk': the last remaining cell of a ship was hit.\n"
        "- Do not fire at the same coordinate twice.\n"
        "- Do not fire outside the A1-H8 range.\n"
        "- Win by sinking all ships in as few shots as possible.\n\n"
        "Board (your view - unknown cells are '.'):\n"
    )

    # 如果有记录，渲染棋盘
    # If we have a state, render the board
    if record is not None:
        state = init_state(record)
        board_repr = board_text(state)
        user_prompt = board_repr + "\n\nFire your first shot:"
    else:
        # 构建空棋盘
        user_prompt = "Here's the empty board:\n" + ("    1 2 3 4 5 6 7 8\n" + "\n".join(f"{_COL_LETTERS[i]} . . . . . . . ." for i in range(GRID))) + "\n\nFire your first shot:"

    return f"{sys_prompt}{user_prompt}"


# 渲染棋盘文本（agent 视角：只显示命中/未命中/未知，永不显示未射击的船只位置）
def board_text(state: GameState) -> str:
    """Render the board as the agent sees it (only miss/hit/unknown, never unshot ship cells)."""
    # 构建命中格子集合
    # Build hit set
    hit_cells = set()
    sunk_ships_count = 0
    for ship in state.ships:
        if ship.is_sunk:
            sunk_ships_count += 1
        for cell in ship.cells:
            if cell in ship.hits:
                hit_cells.add(cell)

    # 构建未命中格子集合
    # Build miss set
    miss_cells = set(state.shots_history) - hit_cells

    # 构建棋盘显示行
    lines = ["    1 2 3 4 5 6 7 8"]
    for row in range(GRID):
        row_cells = []
        for col in range(GRID):
            # 根据格子状态显示不同字符
            if (row, col) in hit_cells:
                row_cells.append("X")  # 命中
            elif (row, col) in miss_cells:
                row_cells.append("o")  # 未命中
            else:
                row_cells.append(".")  # 未知
        lines.append(f"{_COL_LETTERS[row]} {' '.join(row_cells)}")

    # 构建状态信息
    sunk_info = f"Sunk ships: {sunk_ships_count}/{len(state.ships)}"
    shots_info = f"Shots used: {state.shots_used}/{MAX_TURNS}"
    remaining = sum(1 for s in state.ships if not s.is_sunk)
    remaining_info = f"Ships remaining: {remaining}"

    return "\n".join(lines) + f"\n\n{sunk_info} | {shots_info} | {remaining_info}"


# 对完整回合进行评分（用于 reward_fn 和评测框架）
def score_episode(state: GameState, tool_calls: list[dict]) -> dict:
    """Score a complete episode from tool_calls. Used by reward_fn and eval harness."""
    # 创建新的游戏状态用于重放工具调用
    # Replay the tool calls into a fresh state
    test_state = GameState()
    test_state.seed = state.seed
    # 复制船只结构
    # Copy ship structure
    for ship in state.ships:
        test_state.ships.append(Ship(length=ship.length, cells=list(ship.cells)))

    # 初始化统计变量
    invalid_shots = 0

    # 重放所有工具调用
    for call in tool_calls:
        # 获取工具名称
        name = call.get("name") if isinstance(call, dict) else None
        if name != "fire":
            continue
        # 获取工具参数
        args = call.get("arguments", {})
        # 如果是字符串，解析为字典
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        # 获取坐标参数
        coord = args.get("coordinate")

        # 执行射击并更新统计
        result = fire(test_state, coord)
        if result["status"] == "invalid":
            invalid_shots += 1

    # 计算最终统计
    # 直接按船只是否被击沉来计数，避免以船长作为集合键时
    # 将多艘同长度船只误判为一艘（舰队 [4,3,2,2] 中有两艘长 2 的船）。
    total_hits = sum(len(s.hits) for s in test_state.ships)
    sunk_total = sum(1 for s in test_state.ships if s.is_sunk)
    win = is_win(test_state)
    completion = total_hits / TOTAL_SHIP_CELLS if TOTAL_SHIP_CELLS > 0 else 0.0

    # 返回评分结果
    return {
        "win": win,
        "completion": completion,
        "hits": total_hits,
        "sunk_ships": sunk_total,
        "shots_used": test_state.shots_used,
        "invalid_shots": invalid_shots,
    }