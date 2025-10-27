#!/bin/bash
#SBATCH --account=def-six
#SBATCH --job-name=DiGress_Inference
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=10:00:00
#SBATCH --output=outputs/slurm-logs/%j/log.out
#SBATCH --error=outputs/slurm-logs/%j/log.err
#SBATCH --mail-user=er.liang@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load StdEnv/2023
module load python/3.11.5
module load cuda/12.2

set -euo pipefail

# change to repo directory (fall back to hard-coded path if SLURM_SUBMIT_DIR not set)
REPO_DIR="${SLURM_SUBMIT_DIR:-/home/e1444/repos/mammograph}"
cd "$REPO_DIR"

# activate virtualenv inside the repository (ensure ENV exists)
if [ -f "ENV/bin/activate" ]; then
	# shellcheck disable=SC1091
	source ENV/bin/activate
else
	echo "Warning: virtualenv ENV not found at $REPO_DIR/ENV. Continuing without activating virtualenv."
fi

export WANDB_MODE=offline

# Training command: adjust paths below as needed
python -m src.train_hydra --config-name=resnet18