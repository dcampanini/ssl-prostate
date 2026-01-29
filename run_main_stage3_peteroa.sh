#!/bin/bash
#SBATCH --job-name stage3-prostate          # Custom name
#SBATCH -t 3-00:00:00
#SBATCH --cpus-per-task 16                    # Request 2 cores
#SBATCH --mem=64G                                # Indicate required memory
#SBATCH --gpus=1                                # Ask for 1 GPU
#CUDA_VISIBLE_DEVICES=0
module load conda
conda activate /home/dicampanini/miniconda3/envs/env_sslprostate
#python main.py configs/unetr/unetr_prostate_p158f0.yaml #&
# python main.py configs/unetr/unetr_prostate_p158f1.yaml &
# python main.py configs/unetr/unetr_prostate_p158f2.yaml &
# python main.py configs/unetr/unetr_prostate_p158f3.yaml &
# python main.py configs/unetr/unetr_prostate_p158f4.yaml &
# wait
exec -a f0 python main.py configs/unetr/unetr_prostate_uc_f0.yaml &
exec -a f1 python main.py configs/unetr/unetr_prostate_uc_f1.yaml &
exec -a f2 python main.py configs/unetr/unetr_prostate_uc_f2.yaml &
exec -a f3 python main.py configs/unetr/unetr_prostate_uc_f3.yaml &
exec -a f4 python main.py configs/unetr/unetr_prostate_uc_f4.yaml &
wait
