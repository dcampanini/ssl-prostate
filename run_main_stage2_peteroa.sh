#!/bin/bash
#SBATCH --job-name stage2        # Custom name
#SBATCH -t 3-00:00:00
#SBATCH --cpus-per-task 8                   # Request 2 cores
#SBATCH --mem=64G                                # Indicate required memory
#SBATCH --gpus=1                                # Ask for 1 GPU
module load conda
conda activate /home/dicampanini/miniconda3/envs/env_sslprostate
#exec -a stage2_25  python main.py configs/unet/unet_stage2_picai_f0_stage1_25_peteroa.yaml #&
#exec -a stage2_50  python main.py configs/unet/unet_stage2_picai_f0_stage1_50_peteroa.yaml
# wait