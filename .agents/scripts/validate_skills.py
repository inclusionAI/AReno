#!/usr/bin/env python3
"""Validate repository-local AReno skills and their executable scripts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

FRONTMATTER = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)
LINK = re.compile(r"\[[^]]+\]\((?P<target>[^)#]+)(?:#[^)]+)?\)")
NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def metadata(text: str) -> dict[str, str]:
    match = FRONTMATTER.match(text)
    if not match:
        return {}
    result = {}
    for line in match.group("body").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            result[key.strip()] = value.strip()
    return result


def validate_skill(skill: Path, run_help: bool) -> list[str]:
    errors = []
    skill_file = skill / "SKILL.md"
    if not skill_file.is_file():
        return [f"{skill}: missing SKILL.md"]
    text = skill_file.read_text(encoding="utf-8")
    meta = metadata(text)
    if set(meta) != {"name", "description"}:
        errors.append(f"{skill_file}: frontmatter must contain only name and description")
    if meta.get("name") != skill.name or not NAME.fullmatch(meta.get("name", "")):
        errors.append(f"{skill_file}: name must equal directory and use lowercase hyphen-case")
    if len(meta.get("description", "")) < 40:
        errors.append(f"{skill_file}: description is too short to trigger reliably")
    for match in LINK.finditer(text):
        target_text = match.group("target")
        if target_text.startswith(("http://", "https://", "mailto:")):
            continue
        target = skill / target_text
        if not target.exists():
            errors.append(f"{skill_file}: missing linked file {target_text}")
    interface = skill / "agents" / "openai.yaml"
    if not interface.is_file():
        errors.append(f"{skill}: missing agents/openai.yaml")
    else:
        interface_text = interface.read_text(encoding="utf-8")
        for key in ("display_name:", "short_description:", "default_prompt:"):
            if key not in interface_text:
                errors.append(f"{interface}: missing {key[:-1]}")
    for script in sorted((skill / "scripts").glob("*.py")) if (skill / "scripts").exists() else []:
        try:
            compile(script.read_text(encoding="utf-8"), str(script), "exec")
        except Exception as exc:
            errors.append(f"{script}: failed to compile: {exc}")
            continue
        if run_help:
            try:
                process = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                if process.returncode:
                    errors.append(f"{script}: --help exited {process.returncode}: {process.stderr.strip()}")
            except subprocess.TimeoutExpired:
                errors.append(f"{script}: --help timed out after 20 seconds")
            except Exception as exc:
                errors.append(f"{script}: failed to run --help: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".agents/skills"))
    parser.add_argument("--skip-script-help", action="store_true")
    args = parser.parse_args()
    skills = sorted(path for path in args.root.iterdir() if path.is_dir()) if args.root.is_dir() else []
    errors = [error for skill in skills for error in validate_skill(skill, not args.skip_script_help)]
    result = {
        "ok": bool(skills) and not errors,
        "root": str(args.root),
        "skill_count": len(skills),
        "script_count": sum(len(list((skill / "scripts").glob("*.py"))) for skill in skills),
        "errors": errors or ([] if skills else ["no skills found"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
