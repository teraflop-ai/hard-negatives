#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --job-name=hn-search
#SBATCH --array=0-31
#SBATCH --output=/e/project1/reformo/enrico/hard-negatives/logs/%x_%A_%a.out

set -euo pipefail

PROJECT_DIR=/e/project1/reformo/enrico/hard-negatives
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"

exec "$PROJECT_DIR/.venv/bin/accelerate" launch --multi_gpu --num_processes "$SLURM_GPUS_ON_NODE" --main_process_port $((20000 + SLURM_ARRAY_TASK_ID)) main.py search "$@"