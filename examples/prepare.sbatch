#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00
#SBATCH --job-name=hn-prepare
#SBATCH --output=/e/project1/reformo/enrico/hard-negatives/logs/%x_%j.out

set -euo pipefail

PROJECT_DIR=/e/project1/reformo/enrico/hard-negatives
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"

exec "$PROJECT_DIR/.venv/bin/python" -u main.py prepare "$@"