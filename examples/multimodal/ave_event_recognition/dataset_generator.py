"""Download AVE from Hugging Face and generate event-recognition JSONL."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

from common import EventAnnotation, read_annotations, relative_path
from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "UnFaZeD07/AVE-Dataset"
SPLIT_FILES = {"train": "trainSet.txt", "val": "valSet.txt", "test": "testSet.txt"}


def download_ave(dataset_root: str | Path, *, repo_id: str = DEFAULT_REPO_ID) -> Path:
    """Download the original AVE files and extract the media archive."""

    root = Path(dataset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for filename in ("Annotations.txt", *SPLIT_FILES.values()):
        source = Path(hf_hub_download(repo_id, filename, repo_type="dataset"))
        shutil.copy2(source, root / filename)

    archive = Path(hf_hub_download(repo_id, "videos.zip", repo_type="dataset"))
    videos = root / "videos"
    if not videos.exists():
        videos.mkdir(parents=True)
        with zipfile.ZipFile(archive) as handle:
            _safe_extract(handle, videos)
    return root


def generate_manifest(
    dataset_root: str | Path,
    output: str | Path,
    *,
    split: str = "train",
    seed: int = 42,
) -> list[dict]:
    """Generate one record per annotated event in the requested official split."""

    root = Path(dataset_root).expanduser().resolve()
    annotations = read_annotations(root / "Annotations.txt", has_header=True)
    split_rows = read_annotations(root / SPLIT_FILES[split], has_header=False)
    annotations_by_key = {annotation.key: annotation for annotation in annotations}
    selected = [annotations_by_key.get(row.key, row) for row in split_rows]
    annotations_by_video: dict[str, list[EventAnnotation]] = defaultdict(list)
    for annotation in selected:
        annotations_by_video[annotation.video_id].append(annotation)

    records: list[dict] = []
    for video_id, video_annotations in annotations_by_video.items():
        video = _find_video(root / "videos", video_id)
        audio = _ensure_audio(root / "audio", video, video_id)
        intervals = sorted({(row.start_seconds, row.end_seconds) for row in video_annotations})
        for start, end in intervals:
            overlapping = [
                row for row in video_annotations if max(start, row.start_seconds) < min(end, row.end_seconds)
            ]
            labels = sorted({row.event_class for row in overlapping})
            qualities = sorted({row.quality for row in overlapping})
            records.append(_record(video_id, labels, qualities, start, end, root, video, audio, split))

    random.Random(seed).shuffle(records)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def _record(
    video_id: str,
    event_classes: list[str],
    qualities: list[str],
    start_seconds: float,
    end_seconds: float,
    root: Path,
    video: Path,
    audio: Path,
    split: str,
) -> dict:
    return {
        "id": f"{video_id}:events:{start_seconds:g}-{end_seconds:g}",
        "dataset_root": str(root),
        "split": split,
        "video_id": video_id,
        "video_path": relative_path(video, root),
        "audio_path": relative_path(audio, root),
        "event_classes": event_classes,
        "qualities": qualities,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
    }


def _find_video(videos_root: Path, video_id: str) -> Path:
    matches = [path for path in videos_root.rglob(f"{video_id}.*") if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"no extracted AVE video for {video_id}")
    return sorted(matches)[0].resolve()


def _ensure_audio(audio_root: Path, video: Path, video_id: str) -> Path:
    audio_root.mkdir(parents=True, exist_ok=True)
    output = (audio_root / f"{video_id}.wav").resolve()
    if output.is_file():
        return output
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(output),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to extract AVE audio tracks") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to extract audio from {video}") from exc
    return output


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"unsafe path in AVE archive: {member.filename}")
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if not args.skip_download:
        download_ave(args.dataset_root, repo_id=args.repo_id)
    records = generate_manifest(args.dataset_root, args.output, split=args.split, seed=args.seed)
    print(f"wrote {len(records)} {args.split} events to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
