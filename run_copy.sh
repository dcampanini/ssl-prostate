#!/bin/bash
#SBATCH --job-name ssl-prostate          # Custom name
#SBATCH -t 3-00:00:00
#SBATCH -p batch                                # Choose partition (interactive or batch)
#SBATCH -q batch                                # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 16                    # Request 2 cores
#SBATCH --mem=30G                                # Indicate required memory
#SBATCH --gpus=0                               # Ask for 1 GPU
#SBATCH --nodelist ih-loica
scp -r /mnt/researchers/denis-parra/datasets/jfacuse_workdir/new_models_diego/stage1_unetr_dae /mnt/researchers/denis-parra/datasets/ssl_prostate_models_diego/stage1