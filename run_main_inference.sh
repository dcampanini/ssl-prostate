#!/bin/bash
#SBATCH --job-name inference-ssl-prostate          # Custom name
#SBATCH -t 3-00:00:00
#SBATCH -p batch                                # Choose partition (interactive or batch)
#SBATCH -q batch                                # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 4                    # Request 2 cores
#SBATCH --mem=10G                                # Indicate required memory
#SBATCH --gpus=1                                # Ask for 1 GPU
#SBATCH --nodelist ih-condor
module load conda
conda activate /mnt/researchers/pablo-estevez/datasets/dcampanini/envs/env_sslprostate3
# inference
python main.py configs/unetr/unetr_prostate_p158f0_inference.yaml
