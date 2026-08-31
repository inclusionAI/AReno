"""Generate reproducible terminal-hacking puzzles."""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

WORD_BANK = {
    5: "ALERT AMBER BLADE BRICK CABLE CACHE CHAIR CLEAN CLOUD CROWN DEPOT DREAM DRIFT EAGER FIELD FLAME FRAME GHOST GLASS GRAIN GRANT GREEN HEART INDEX LASER LAYER LIGHT METAL NIGHT NORTH OCEAN PANEL PAPER PHASE PLANT POWER RADIO RIVER ROBOT ROUGH ROUTE SOLAR SOUTH STEAM STONE STORM TABLE TRACE TRACK TRAIN VAULT WATER WHEEL WORLD".split(),
    6: "ACCESS ACTIVE AGENTS BINARY BREACH BRIDGE BROKEN BUFFER CAMERA CIPHER COLUMN COMBAT COREUP CRATER DEVICE ENGINE FILTER FUSION GARAGE GARDEN HIDDEN JACKET MEMORY ORANGE PACKET PLANET REMOTE RESCUE ROUTER SCREEN SECRET SENSOR SHIELD SIGNAL SOCKET STATIC SYSTEM TARGET TUNNEL VECTOR VERIFY WINDOW".split(),
    7: "ARCHIVE BARRIER BATTERY CABINET CAPTURE CENTRAL CHANNEL CIRCUIT COMMAND CONTROL CORRECT DEFENSE DISPLAY FACTORY GATEWAY HAZARDS KEYCARD MACHINE MONITOR NETWORK NUCLEAR OFFLINE OPERATE REACTOR RECOVER SHELTER STORAGE UNKNOWN VINTAGE WARNING".split(),
}
JUNK = "!@#$%^&*_-+=;:,.?/\\|~`'\""
BRACKETS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
MIN_CANDIDATES = 16
MAX_CANDIDATES = 20


def _make_one(index_seed: tuple[int, int]) -> dict:
    index, seed = index_seed
    rng = random.Random(seed)
    word_length = rng.choice(sorted(WORD_BANK))
    count = rng.randint(MIN_CANDIDATES, MAX_CANDIDATES)
    candidates = rng.sample(WORD_BANK[word_length], count)
    password = rng.choice(candidates)
    probes = []
    duds = [word for word in candidates if word != password]
    for probe_index, (opening, closing) in enumerate(rng.sample(BRACKETS, 3)):
        effect = "replenish" if probe_index == 0 else "remove_dud"
        token = opening + "".join(rng.choice(JUNK) for _ in range(rng.randint(1, 3))) + closing
        probes.append(
            {
                "id": f"P{probe_index}",
                "token": token,
                "effect": effect,
                "target": duds[probe_index % len(duds)] if effect == "remove_dud" else None,
            }
        )
    rng.shuffle(probes)
    tokens = [(word, f"guess:{word}") for word in candidates] + [
        (probe["token"], f"probe:{probe['id']}") for probe in probes
    ]
    rows = _build_dump(rng, tokens)
    return {
        "id": f"terminal-hacking-{index + 1:06d}",
        "seed": seed,
        "password": password,
        "candidates": candidates,
        "attempts": 3,
        "probes": probes,
        "dump_rows": rows,
    }


def _build_dump(rng: random.Random, tokens: list[tuple[str, str]]) -> list[dict]:
    row_count = max(18, (len(tokens) + 1) // 2)
    slots = [(row, side) for row in range(row_count) for side in ("left", "right")]
    rng.shuffle(slots)
    placed: dict[tuple[int, str], tuple[str, str]] = dict(zip(slots, tokens, strict=False))
    base = rng.randrange(0xA000, 0xE000, 0x100)
    rows = []
    for row_index in range(row_count):
        row: dict[str, object] = {}
        for side_index, side in enumerate(("left", "right")):
            address = base + (row_index * 2 + side_index) * 12
            text = "".join(rng.choice(JUNK) for _ in range(12))
            segments: list[dict[str, str]] = [{"text": text, "action": ""}]
            token_data = placed.get((row_index, side))
            if token_data is not None:
                token, action = token_data
                offset = rng.randint(0, 12 - len(token))
                text = text[:offset] + token + text[offset + len(token) :]
                segments = []
                if offset:
                    segments.append({"text": text[:offset], "action": ""})
                segments.append({"text": token, "action": action})
                if offset + len(token) < len(text):
                    segments.append({"text": text[offset + len(token) :], "action": ""})
            row[f"{side}_address"] = f"0x{address:04X}"
            row[side] = text
            row[f"{side}_segments"] = segments
        rows.append(row)
    return rows


def _fingerprint(record: dict) -> str:
    payload = {key: value for key, value in record.items() if key not in {"id", "seed"}}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def generate_records(count: int = 8192, *, seed: int = 2026, workers: int = 20) -> list[dict]:
    """Generate independent deterministic puzzles, optionally in worker processes."""

    master = random.Random(seed)
    seeds: list[int] = []
    seed_set: set[int] = set()
    while len(seeds) < count:
        candidate_seed = master.getrandbits(63)
        if candidate_seed not in seed_set:
            seed_set.add(candidate_seed)
            seeds.append(candidate_seed)
    jobs = list(enumerate(seeds))
    if workers <= 1 or count <= 1:
        records = [_make_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, count)) as pool:
            records = list(pool.map(_make_one, jobs))
    seen: set[str] = set()
    for index, record in enumerate(records):
        fingerprint = _fingerprint(record)
        retry = 0
        while fingerprint in seen:
            retry += 1
            record = _make_one((index, seeds[index] ^ (retry * 0x9E3779B97F4A7C15)))
            fingerprint = _fingerprint(record)
        record["id"] = f"terminal-hacking-{index + 1:06d}"
        records[index] = record
        seen.add(fingerprint)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    records = generate_records(args.count, seed=args.seed, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
