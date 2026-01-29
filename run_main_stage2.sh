#!/bin/bash
#SBATCH --job-name stage2-ssl         # Custom name
#SBATCH -t 3-00:00:00
#SBATCH -p batch                                # Choose partition (interactive or batch)
#SBATCH -q batch                                # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 8                    # Request 2 cores
#SBATCH --mem=24G                                # Indicate required memory
#SBATCH --gpus=1                                # Ask for 1 GPU
#SBATCH --nodelist ih-condor
module load conda
conda activate /mnt/researchers/pablo-estevez/datasets/dcampanini/envs/env_sslprostate3
exec -a stage2 python main.py configs/unet/unet_prostate_picai_f0_server79.yaml
#exec -a stage2  python main.py configs/unet/unet_prostate_picai_f0.yaml
#exec -a stage2  python main.py configs/unetr/unetr_stage2_picai_f0.yaml
exec -a stage2  python main.py configs/unetr/unetr_stage2_picai_f0_server79.yaml
#exec -a stage2_only  python main.py configs/unetr/unetr_stage2_only_picai_f0.yaml

