#!/bin/bash
#SBATCH --job-name stage3-prostate          # Custom name
#SBATCH -t 3-00:00:00
#SBATCH -p batch                                # Choose partition (interactive or batch)
#SBATCH -q batch                                # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 8                    # Request 2 cores
#SBATCH --mem=64G                                # Indicate required memory
#SBATCH --gpus=1                                # Ask for 1 GPU
#SBATCH --nodelist ih-loica
module load conda
conda activate /mnt/researchers/pablo-estevez/datasets/dcampanini/envs/env_sslprostate3
# exec -a f0 python main.py configs/unetr/unetr_prostate_p158f0.yaml &
# exec -a f1 python main.py configs/unetr/unetr_prostate_p158f1.yaml &
# exec -a f2 python main.py configs/unetr/unetr_prostate_p158f2.yaml &
# exec -a f3 python main.py configs/unetr/unetr_prostate_p158f3.yaml &
# exec -a f4 python main.py configs/unetr/unetr_prostate_p158f4.yaml &
# wait


exec -a f0 python main.py configs/unet/unetr_prostate_p158f0.yaml &
exec -a f1 python main.py configs/unet/unetr_prostate_p158f1.yaml &
exec -a f2 python main.py configs/unet/unetr_prostate_p158f2.yaml &
exec -a f3 python main.py configs/unet/unetr_prostate_p158f3.yaml &
exec -a f4 python main.py configs/unet/unetr_prostate_p158f4.yaml &
wait

