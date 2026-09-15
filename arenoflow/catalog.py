"""Discover CLI and registered model metadata without importing GPU code."""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from pathlib import Path

from arenoflow.algorithm_policy import annotate_options
from arenoflow.model_presets import CHECKPOINTS

ROOT = Path(__file__).resolve().parents[1]


def literal(node, constants=None):
    if isinstance(node, ast.Name) and constants and node.id in constants:
        return constants[node.id]
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def cli_schema(kind: str, root: Path = ROOT) -> list[dict]:
    tree = ast.parse((root / "areno" / "cli" / f"{kind}.py").read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            constants[node.target.id] = literal(node.value, constants)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = literal(node.value, constants)
    groups = {name: group for group, names in constants.get("TRAIN_OPTION_GROUPS", ()) for name in names}
    command = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == f"{kind}_command")
    options = []
    for dec in command.decorator_list:
        if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute) or dec.func.attr != "option":
            continue
        declarations = [literal(arg) for arg in dec.args]
        flags = [v for v in declarations if isinstance(v, str) and v.startswith("--")]
        explicit = next((v for v in declarations if isinstance(v, str) and not v.startswith("-")), None)
        primary = flags[0].split("/")[0]
        name = explicit or primary[2:].replace("-", "_")
        kw = {k.arg: k.value for k in dec.keywords}
        default = literal(kw.get("default"), constants)
        typ = kw.get("type")
        choices = None
        type_name = typ.id if isinstance(typ, ast.Name) else None
        if isinstance(typ, ast.Call) and isinstance(typ.func, ast.Attribute) and typ.func.attr == "Choice":
            choices = literal(typ.args[0])
        paired = "/" in flags[0]
        is_flag = paired or literal(kw.get("is_flag")) is True
        if is_flag:
            type_name = "bool"
            if "default" not in kw:
                default = False
        if type_name is None:
            type_name = "float" if isinstance(default, float) else "int" if type(default) is int else "str"
        options.append(
            dict(
                name=name,
                flag=primary,
                flags=flags,
                negative=flags[0].split("/")[1] if paired else None,
                type=type_name,
                choices=choices,
                default=default,
                required=bool(literal(kw.get("required"))),
                multiple=bool(literal(kw.get("multiple"))),
                help=literal(kw.get("help")) or "",
                group=groups.get(name, "General"),
            )
        )
    return annotate_options(options, root / "areno/cli/train.py") if kind == "train" else options


def catalog(root: Path = ROOT) -> dict:
    registry = ast.parse((root / "areno/models/__init__.py").read_text())
    registered = {
        n.args[0].func.id
        for n in ast.walk(registry)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "register_adapter"
        and n.args
        and isinstance(n.args[0], ast.Call)
        and isinstance(n.args[0].func, ast.Name)
    }
    models = []
    for file in sorted((root / "areno/models").glob("*/model.py")):
        for node in ast.parse(file.read_text()).body:
            if not isinstance(node, ast.ClassDef) or node.name not in registered:
                continue
            for item in node.body:
                if isinstance(item, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "name" for t in item.targets
                ):
                    models.append(
                        dict(
                            id=literal(item.value),
                            family=file.parent.name,
                            adapter=node.name,
                            source=str(file.relative_to(root)),
                        )
                    )
    models = [model for model in models if model["id"] != "bailing_moe_linear_v2"]
    for model in models:
        model["checkpoint"] = CHECKPOINTS.get(model["id"], "")
    algorithms = []
    for node in ast.walk(ast.parse((root / "areno/api/algorithms.py").read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "AlgorithmSpec":
            fields = {kw.arg: literal(kw.value) for kw in node.keywords}
            if isinstance(fields.get("name"), str):
                algorithms.append(dict(id=fields["name"], rollout=fields.get("requires_rollout", False)))
    # Checkpoint suggestions are examples in this checkout, never a second model registry.
    references = set()
    for file in [root / "README.md", *sorted((root / "docs").rglob("*.rst"))]:
        references.update(
            re.findall(r"(?<![\w/])(?:Qwen|google|meta-llama|allenai|openbmb|inclusionAI)/[\w.\-]+", file.read_text())
        )
    train, serve = cli_schema("train", root), cli_schema("serve", root)
    for field in train + serve:
        if field["name"] == "model_hub":
            field.update(
                default="hf",
                choices=["hf"],
                help="Model repositories use Hugging Face. Dataset sources are configured separately.",
            )
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "source checkout"
    fingerprint = hashlib.sha256((root / "areno/cli/train.py").read_bytes()).hexdigest()[:12]
    workflow = (root / ".github/workflows/create-docker-image.yml").read_text()
    image_name = re.search(r'image_name:.*?default: "([^"]+)"', workflow, re.S).group(1)
    return dict(
        models=models,
        checkpoints=sorted(references | {model["checkpoint"] for model in models if model["checkpoint"]}),
        algorithms=algorithms,
        train=train,
        serve=serve,
        revision=revision,
        schema_version=fingerprint,
        image=f"ghcr.io/inclusionai/{image_name}:latest",
        presets=presets(train),
    )


def presets(schema: list[dict]) -> dict:
    base = dict(
        world_size=1,
        tp_size=1,
        batch_size=4,
        mini_bs=1,
        max_steps=100,
        epochs=1,
        max_prompt_tokens=1024,
        max_new_tokens=1024,
        activation_checkpointing=True,
        save_interval=25,
        model_hub="hf",
        attn_backend="native",
    )
    names = {p["name"] for p in schema}
    result = {}
    for algo in ("sft", "dpo", "gspo", "grpo", "ppo"):
        config = {**base, "algo": algo, "lr": 2e-5 if algo == "sft" else 1e-6}
        if algo in ("gspo", "grpo", "ppo"):
            config.update(n_samples=4, reward_fn_path="examples/math/math_verify_reward.py")
        result[algo] = {key: value for key, value in config.items() if key in names}
    return result


def arguments(kind: str, values: dict, schema: list[dict]) -> list[str]:
    if not isinstance(values, dict):
        raise ValueError("Parameters must be an object")
    by_name = {p["name"]: p for p in schema}
    unknown = set(values) - by_name.keys()
    if unknown:
        raise ValueError(f"Unknown {kind} parameters: {', '.join(sorted(unknown))}")
    args = []
    for name, value in values.items():
        if value is None or value == "":
            continue
        item = by_name[name]
        if item["type"] == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            if value:
                args.append(item["flag"])
            elif item["negative"]:
                args.append(item["negative"])
            elif item["default"] is True:
                raise ValueError(f"{name} cannot be disabled through this CLI")
            continue
        entries = value if item["multiple"] and isinstance(value, list) else [value]
        for entry in entries:
            if isinstance(entry, (dict, list, bool)):
                raise ValueError(f"Invalid value for {name}")
            if item["type"] in ("int", "float"):
                import math

                try:
                    number = float(entry)
                    if not math.isfinite(number) or (item["type"] == "int" and not number.is_integer()):
                        raise ValueError()
                except (ValueError, TypeError):
                    raise ValueError(f"{name} must be a finite {item['type']}") from None
                entry = int(number) if item["type"] == "int" else number
            if item["choices"] and entry not in item["choices"]:
                raise ValueError(f"{name} must be one of {item['choices']}")
            args.extend([item["flag"], str(entry)])
    return args
