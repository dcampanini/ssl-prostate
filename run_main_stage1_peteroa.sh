#!/bin/bash
#SBATCH --job-name stage1        # Custom name
#SBATCH -t 3-00:00:00
#SBATCH --cpus-per-task 4                    # Request 2 cores
#SBATCH --mem=32G                                # Indicate required memory
#SBATCH --gpus=1                                # Ask for 1 GPU
module load conda
conda activate /home/dicampanini/miniconda3/envs/env_sslprostate
#python main.py configs/unetr/unetr_prostate_mae_peteroa.yaml
python main.py configs/unetr/unetr_prostate_denoise.yaml