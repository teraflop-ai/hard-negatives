#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --job-name=hn-embed
#SBATCH --array=0-31
#SBATCH --output=/e/project1/reformo/enrico/hard-negatives/logs/%x_%A_%a.out

set -euo pipefail

PROJECT_DIR=/e/project1/reformo/enrico/hard-negatives
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"

exec "$PROJECT_DIR/.venv/bin/python" -u main.py embed "$@"