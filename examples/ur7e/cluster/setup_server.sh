#!/usr/bin/env bash
# One-time setup of openpi on a GPU server reached over SSH, starting from an empty folder.
#
#   bash setup_server.sh <work_dir> <hf-user>/<dataset-name>
#   e.g. bash setup_server.sh ~/openpi_work myuser/ur7e_real_robot_new_size
#
# Then start training with:  bash <work_dir>/openpi/examples/ur7e/cluster/train.sh

WORK=work
HF_DATASET=Nicholas-Nik/Teleop
OTHER_DATASET=Nicholas-Nik/handover_dagger

REPO_URL=https://github.com/Niiikkkk/openpi.git
BRANCH=main_subtask
REPO_ID=ur7e_real_robot_new_size        # must match repo_id in the TrainConfig
OTHER_REPO_ID=ur7e_real_robot_new_size_handover_dagger
CONFIG=ur7e_pi05_real_from_droid_handover_finetune
BASE_CKPT=Nicholas-Nik/ckpt
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment (export HF_TOKEN=hf_...)}"

mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)

# Environment file reused by train.sh: keeps every cache inside $WORK.
cat > "$WORK/env.sh" <<EOF
export HF_LEROBOT_HOME=$WORK/lerobot
EOF
source "$WORK/env.sh"

# 1. uv
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 2. Code
cd "$WORK"
if [ ! -d openpi ]; then
    git clone --branch "$BRANCH" "$REPO_URL" openpi
fi
cd openpi
git submodule update --init --recursive

# 3. Python environment
uv sync
uv pip install -e .

# 4. Dataset (folder name must equal the config's repo_id)
uv run hf auth login --token "$HF_TOKEN"
uv run hf download "$HF_DATASET" --repo-type dataset --local-dir "$HF_LEROBOT_HOME/$REPO_ID"
uv run hf download "$OTHER_DATASET" --repo-type dataset --local-dir "$HF_LEROBOT_HOME/$OTHER_REPO_ID"


# 6. Norm stats (skipped if they were committed to the repo)
uv run scripts/compute_norm_stats.py --config-name "$CONFIG"

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "$CONFIG" --exp-name="$CONFIG" --overwrite

# echo
# echo "Setup done. Start training with:"
# echo "  bash $WORK/openpi/examples/ur7e/cluster/train.sh"
