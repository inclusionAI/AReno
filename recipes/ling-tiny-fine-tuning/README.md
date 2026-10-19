# Ling Tiny payment SFT notebooks

[Mac notebook](Ling-Tiny-SFT-Mac.ipynb) ·
[Google Colab notebook](Ling-Tiny-SFT-Colab.ipynb) ·
[ModelScope dataset](https://modelscope.cn/datasets/AsystemAReno/Ling-Tiny-SFT) ·
[Hugging Face dataset](https://huggingface.co/datasets/AsystemAReno/Ling-Tiny-SFT)

Two platform-specific notebooks cover installation, published data loading, baseline
validation, attention LoRA SFT, adapter export, fresh-process reload, and validation
comparison. The Mac notebook uses MLX; the Google Colab notebook uses CUDA and starts
with **To Run this** and **News**, following the tutorial organization of the
[Unsloth example](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-%2820B%29-Fine-tuning.ipynb).
Both share the same dataset, experiment defaults, and AReno training/scoring helpers.

**Status:** implementation ready for review. Notebook execution, CPU tests, and
real training on Mac/Colab/Kaggle have not yet been run for this implementation.
Ling MLX support is experimental. No platform capacity or score improvement is claimed.

## Mac / SSH

From this branch's checkout, with Python 3.10+ and Git available:

```bash
bash recipes/ling-tiny-fine-tuning/run.sh
```

The command creates `recipes/ling-tiny-fine-tuning/.venv`, installs dependencies,
registers a kernel inside that environment, and executes `Ling-Tiny-SFT-Mac.ipynb`
without a browser. The first run downloads dependencies and model weights. To reuse a checkpoint:

```bash
bash recipes/ling-tiny-fine-tuning/run.sh --model-path /path/to/Ling-3.0-tiny
```

Optional arguments: `--data-hub huggingface` (default: `modelscope`) and
`--output-dir /path/to/new-run-directory`. Set `PYTHON=python3.12` to choose the
interpreter when creating the environment. No credentials or machine endpoints
are embedded in the recipe.

An executed notebook is saved under `runs/notebook-*/`, after every completed cell
and on failure. Model artifacts and metrics are stored separately under
`runs/ling-tiny-*/` or the requested output directory. Cell failure stops execution
and returns a nonzero exit status. Re-running starts a new experiment; this launcher
does not resume optimizer state from a failed run.

## Colab / Kaggle

After the recipe branch is published to GitHub, use
[Open in Colab](https://colab.research.google.com/github/inclusionAI/AReno/blob/chore/ling-tiny-sft-recipe/recipes/ling-tiny-fine-tuning/Ling-Tiny-SFT-Colab.ipynb).
This link is a publication target, not a confirmed live deployment.

For Kaggle, download `Ling-Tiny-SFT-Colab.ipynb` from the published branch and import that
notebook into a new Notebook. Enable Internet and an NVIDIA GPU. Both platforms
execute the installation cell, which retrieves the AReno source ref configured in
the notebook. Set `ARENO_REF` to the matching published commit when releasing it.
The notebook has standard Python 3 metadata and does not require Colab-specific APIs.

Use a fresh hosted runtime with CUDA PyTorch >= 2.6 and the CUDA toolkit (`nvcc`).
The installer builds AReno's native extension and installs dependencies in the
hosted Python environment. The local launcher selects the Mac notebook;
use the Colab notebook for Linux CUDA environments.

Opening the file and completing training are separate requirements: the unquantized
base weights plus training state must fit on one accelerator. A free T4 or other
small GPU is not a supported capacity claim. Multi-GPU tensor parallelism and QLoRA
are outside this first recipe. The notebook reports hardware and checks obvious
weight-memory failures; passing that check does not prove training will fit.

## Experiment contract

- Fixed published splits: 1,400 train / 100 validation. No generation or re-splitting.
- Both dataset mirrors are pinned to their uploaded commits in the notebook.
- The base model defaults to a pinned ModelScope revision; a local path is explicit.
- One full training epoch, batch size 4, microbatch 1, rank 8, alpha 16, learning rate 1e-4.
- The same 100 validation prompts are evaluated before and after training, with greedy decoding.
- Direct JSON answers; full typed comparison, including errors and array order.
- Token-budget failures stop the experiment. No silent filtering or truncation.
- Final PEFT adapter is reloaded in a fresh worker for the second evaluation.
- This validation comparison is not an independent held-out test result.

`config.json` and `environment.txt` in each run record the source commit, source dirty
state, model/data references, parameters, and installed package versions. Source
publication and dependency verification are still required before treating this
as a released reproducible recipe.

## Files used at runtime

| File | Purpose |
| --- | --- |
| `Ling-Tiny-SFT-Mac.ipynb` | Apple Silicon MLX tutorial and headless experiment |
| `Ling-Tiny-SFT-Colab.ipynb` | Google Colab CUDA tutorial, also importable into Kaggle |
| `run.sh` | Mac isolated environment and headless entry point |
| `install.py` | Shared installation for local and hosted environments |
| `execute.py` | Mac notebook kernel execution and partial-output preservation |
| `recipe.py` | Token checks, strict scoring, isolated AReno train/eval workers |

The notebook does not call `dataset_generator.py`, `prepare_mlx_data.py`, or
`prepare_data.py`, and does not load the repository's historical `data/` directory.
The payment rules are already included in the published prompts, and targets are
already prepared. The migrated service and generator files remain available for
review; they are not runtime dependencies of this notebook.

## Review and validation

After code review, the next validation step is notebook structure/syntax, the
CPU scoring tests in `tests/test_ling_recipe_cpu.py`, and a small headless execution
check. Real model loading, training, adapter reload, and hosted execution require
separate validation. Existing migrated payment tests have not been run in this step.

## Historical migration record

The notes below describe the earlier migration. Paths refer to the old
`recipes/payment` location, before this directory was renamed. Legacy imports in
those offline files still need review before those historical entry points are used.
They do not describe the new notebook entry point.

## 来源与文件对应

来源仓库：`mlx-finetune-microsystem`。
来源提交：`280a5d921b8679dffd1b1941ce6d6632bfe9d84c`。
迁入时来源工作区干净。

| 原仓库文件 | AReno 中的位置 | 本次调整 |
| --- | --- | --- |
| `src/payment_service/__init__.py` | `recipes/payment/payment_service/__init__.py` | 原样迁入 |
| `src/payment_service/models.py` | `recipes/payment/payment_service/models.py` | 原样迁入 |
| `src/payment_service/promotion_engine.py` | `recipes/payment/payment_service/promotion_engine.py` | 原样迁入 |
| `src/payment_service/service.py` | `recipes/payment/payment_service/service.py` | 原样迁入 |
| `src/payment_service/prompt_templates.py` | `recipes/payment/payment_service/prompt_templates.py` | 原样迁入 |
| `scripts/dataset_generator.py` | `recipes/payment/dataset_generator.py` | 更新包导入；默认数据目录改为本 recipe 下的 `data/` |
| `scripts/prepare_mlx_data.py` | `recipes/payment/prepare_mlx_data.py` | 更新包导入和默认数据目录；移除原仓库的 `sys.path` 注入 |
| `scripts/payment_reward.py` | `recipes/payment/payment_reward.py` | 原样迁入，保留历史评分语义 |
| `tests/test_payment_system_and_verifier.py` | `tests/test_payment_recipe_cpu.py` | 更新导入和测试文件名，保留原有四个测试 |

另外新增本说明及 `recipes/payment/__init__.py`。保留原有模块划分和函数名，
方便与来源代码对照；没有合并重写为新的 `payment_task.py`。
首次迁入没有包含数据集、模型权重、历史报告、训练配置或其他实验脚本。

导入以 AReno 仓库根目录为起点，使用 `recipes.payment` 包路径。
生成器与数据准备脚本后续采用 `python -m recipes.payment.<模块名>` 形式调用；
没有执行这两个历史入口，也没有验证其运行结果。

## 保留的历史行为及待处理项

以下问题在本次迁入中保留，待人工确认迁入内容后，再按确认的范围验证和修正。

- `dataset_generator.py` 的请求指纹包含订单 ID，不能据此保证业务输入跨集合无重复。
  场景标签代表生成意图，是否实际触发对应业务边界需要检查。
- `prompt_templates.py` 的叠加优惠、辅助渠道选择及余额不足描述与服务实现存在差异。
  提示词和生成的推理文本尚未重新审核；原文中的正确性保证不代表本 recipe 的验证结果。
- `payment_reward.py` 是历史五维部分评分器，尚非完整 schema 或严格业务结果评分器。
  顶层订单字段、错误响应及数值类型等校验仍需补齐。
  原有 `reward_fn(record)` 及其 AReno 兼容性描述一并保留，尚未接入或验证当前 AReno reward 接口。
- `prepare_mlx_data.py` 保留历史 MLX/QLoRA 命名、默认模型位置
  `~/models/Ling-3.0-tiny-4bit`、长度阈值及尾部划分验证集的行为。
  它还保留了长度过滤异常后继续执行的逻辑。
  这些默认值不代表新 recipe 的模型或训练配置，后续需接入显式模型参数与数据检查。
- 原生成器的 RL 数据分支保留以便审阅原始迁入差异；本 recipe 计划使用 SFT。
- 原有四个测试仅作为迁入的测试材料，本次没有运行，也不足以替代后续的严格验证。

后续验证应先检查迁入前后的行为一致性，再为数据隔离、提示词及严格评分的修改
建立相应测试。最终训练入口及平台可用性另行实施和验证。
