# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

openpi (Physical Intelligence) is an open-source repo for robotics vision-language-action (VLA) models: π₀ (flow-based), π₀-FAST (autoregressive, FAST action tokenizer), and π₀.₅ (flow-matching head only, in this repo). Both JAX (primary) and PyTorch (newer, fewer features) implementations exist side by side.

## Setup

```bash
git submodule update --init --recursive   # required, repo uses submodules
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` is required because LeRobot is pulled as a dependency. Python 3.11+, uv-managed. Uses `uv` workspace with `packages/*` (currently `packages/openpi-client`).

For PyTorch support, after `uv sync`, patch the vendored transformers library (needed for AdaRMS support, activation precision control, and static KV cache):

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

This mutates the uv cache and can leak into other projects using transformers from the same cache — `uv cache clean transformers` to undo.

## Common commands

```bash
# Lint / format (also runs in pre-commit and CI)
uv run ruff check .
uv run ruff format .
pre-commit install && pre-commit run --all-files

# Tests (CI runs: uv run pytest --strict-markers -m "not manual")
uv run pytest --strict-markers -m "not manual"
uv run pytest src/openpi/models/pi0_test.py            # single file
uv run pytest src/openpi/models/pi0_test.py::test_name # single test
# testpaths are: src, scripts, packages

# Compute norm stats (required before training a new config)
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# Train (JAX)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<run_name> --overwrite

# Train (PyTorch)
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<n> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Serve a trained policy over websocket
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_dir>

# Convert a JAX checkpoint to PyTorch
uv run examples/convert_jax_model_to_pytorch.py --checkpoint_dir <jax_ckpt> --config_name <config_name> --output_path <out>
```

Checkpoints download from `gs://openpi-assets` and cache in `~/.cache/openpi` (override with `OPENPI_DATA_HOME`). Local training checkpoints go to `./checkpoints`.

## Architecture

### Data flow: raw data → model

There are two parallel pipelines (train time and inference time) that must produce consistent tensors, both defined through the same `DataConfig`/transform mechanism:

1. **Repack transforms** — adapt a dataset-specific raw format (e.g. LeRobot dataset columns) into a common intermediate dict shape.
2. **Data transforms** — robot/dataset-specific transforms (e.g. `AlohaInputs`, `LiberoInputs`, `DroidInputs` in `src/openpi/policies/*_policy.py`) that map into the shape the model transforms expect. Applied before normalization.
3. **Normalization** — z-score or quantile normalization using precomputed `norm_stats` (see `openpi.shared.normalize`, `scripts/compute_norm_stats.py`).
4. **Model transforms** (`ModelTransformFactory` in `src/openpi/training/config.py`) — model-type-specific (PI0 / PI05 / PI0_FAST) transforms: image resizing, prompt injection/tokenization, state/action padding, FAST tokenization. Produces `openpi.models.model.Observation`/`Actions`.

Each policy module under `src/openpi/policies/` defines both `*Inputs` (raw → model format) and `*Outputs` (model format → environment/robot format) transform classes, used identically at train and inference time — this symmetry is the reason to look at the same file for both directions when adding a new robot.

### Configuration system (`src/openpi/training/config.py`)

Everything is driven by a registry of `TrainConfig` dataclasses in `_CONFIGS` (looked up by name via `_CONFIGS_DICT` / `get_config(name)`). A `TrainConfig` bundles:
- `model`: a `_model.BaseModelConfig` (`Pi0Config` etc.) — architecture/model-type/action horizon/action dim.
- `data`: a `DataConfigFactory` subclass (`FakeDataConfig`, `SimpleDataConfig`, `LeRobotAlohaDataConfig`, `LeRobotLiberoDataConfig`, DROID RLDS config, etc.) that builds the `DataConfig` (repack/data/model transforms + norm stats loading) for that dataset/robot.
- `weight_loader`: how to initialize weights (from scratch, from a base checkpoint, etc. — see `weight_loaders.py`).
- Optimizer/training hyperparameters (see `optimizer.py`).

To add support for a new robot/dataset: add an `Inputs`/`Outputs` pair in `src/openpi/policies/`, a `DataConfigFactory` (or reuse `SimpleDataConfig`), and register a new `TrainConfig` entry in `_CONFIGS`.

### Models (`src/openpi/models/`)

- `model.py` defines the shared `Observation`/`Actions` data structures (dict-of-arrays with fixed `IMAGE_KEYS`, `IMAGE_RESOLUTION`), `ModelType` enum (PI0, PI0_FAST, PI05), and abstract model interface — generic over JAX arrays / PyTorch tensors / numpy arrays via `ArrayT`.
- `pi0.py` / `pi0_config.py`: π₀/π₀.₅ flow-matching model (JAX/Flax nnx).
- `pi0_fast.py`: autoregressive FAST-tokenizer model.
- `gemma.py` / `gemma_fast.py`, `siglip.py`, `vit.py`: backbone components (Gemma language model, SigLIP vision encoder).
- `tokenizer.py`: `PaligemmaTokenizer` and `FASTTokenizer`.
- `lora.py`: LoRA fine-tuning support (JAX only; not yet supported in PyTorch path).
- `src/openpi/models_pytorch/`: PyTorch reimplementation (`pi0_pytorch.py`) plus `transformers_replace/` — files that get copied over the installed `transformers` package to patch it (see Setup).

Not yet supported in the PyTorch path: π₀-FAST, mixed precision training, FSDP, LoRA, EMA weights.

### Training / serving entry points

- `scripts/train.py` / `scripts/train_pytorch.py`: training loops (JAX vs PyTorch), read a `TrainConfig` by name.
- `scripts/compute_norm_stats.py`: must be run once per new data config before training (produces the norm stats consumed by `DataConfigFactory._load_norm_stats`).
- `scripts/serve_policy.py`: spins up a websocket policy server (`src/openpi/serving/websocket_policy_server.py`) around a `policy_config.create_trained_policy(...)`-built policy; this is how remote/robot-side inference clients talk to the model (see `docs/remote_inference.md`).
- `src/openpi/policies/policy.py` / `policy_config.py`: the runtime `Policy` object wrapping a model + its input/output transforms for `policy.infer(example)`.
- `packages/openpi-client`: standalone lightweight client package (websocket client, image tools) meant to be installed on the robot side without the full JAX/PyTorch training stack.

### GPU memory / precision notes worth knowing before touching training code

- JAX training defaults to mixed precision (float32 weights/grads, bfloat16 activations); set `dtype=float32` in config for full float32.
- PyTorch training is full bfloat16 (default) or full float32 via `pytorch_training_precision`; no mixed precision yet.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` and `--fsdp-devices <n>` (JAX only) are the standard levers for GPU memory pressure.

## Linting

Ruff, line-length 120, target py311, extensive rule set (see `pyproject.toml`). isort is `force-single-line` with sorted imports within sections. `third_party/`, `docker/`, and `src/openpi/models_pytorch/transformers_replace/` are excluded from ruff.
