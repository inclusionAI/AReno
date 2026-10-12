"""Send a local WAV/MP3 file to an AReno OpenAI-compatible server."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path


def encode_input_audio(audio_path: str | Path) -> dict:
    """Build the OpenAI ``input_audio`` content part for a local file."""

    path = Path(audio_path)
    audio_format = path.suffix.lower().removeprefix(".")
    if audio_format not in {"wav", "mp3"}:
        raise ValueError(f"input_audio requires a .wav or .mp3 file, got {path.name!r}")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": payload, "format": audio_format}}


def build_messages(audio_path: str | Path, prompt: str) -> list[dict]:
    """Create one audio-and-text user message."""

    return [
        {
            "role": "user",
            "content": [encode_input_audio(audio_path), {"type": "text", "text": prompt}],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Local WAV or MP3 file")
    parser.add_argument("--prompt", default="Transcribe and summarize this audio.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI Python client to run this example") from exc

    client = OpenAI(base_url=args.base_url, api_key="unused")
    response = client.chat.completions.create(
        model=args.model,
        messages=build_messages(args.audio, args.prompt),
        max_tokens=args.max_tokens,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
