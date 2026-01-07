#!/bin/bash
#SBATCH --job-name inference-ssl-prostate          # Custom name
#SBATCH -t 3-00:00:00
#SBATCH --cpus-per-task 4                    # Request 2 cores
#SBATCH --mem=10G                                # Indicate required memory
#SBATCH --gpus=1                                # Ask for 1 GPU
module load conda
conda activate /home/dicampanini/miniconda3/envs/env_sslprostate
# inference
python main.py configs/unetr/unetr_prostate_p158f0_inference_peteroa.yaml
