#!/bin/bash
#SBATCH --job-name inference-ssl-prostate          # Custom name
#SBATCH -t 3-00:00:00
#SBATCH --cpus-per-task 4                    # Request 2 cores
#SBATCH --mem=10G                                # Indicate required memory
#SBATCH --gpus=1                                # Ask for 1 GPU
CUDA_VISIBLE_DEVICES=5
module load conda
conda activate /home/dicampanini/miniconda3/envs/env_sslprostate
# inference
#exec -a inference python main.py configs/unetr/unetr_prostate_p158f0_inference_peteroa.yaml
exec -a inference python main.py configs/unetr/unetr_prostate_uc_f0_inference_peteroa.yaml
#exec -a inference python main_test.py configs/unetr/unetr_prostate_p158f0_inference_peteroa_all.yaml

#exec -a inference python main_test.py configs/unet/unet_prostate_uc_inference_peteroa_all.yaml 

