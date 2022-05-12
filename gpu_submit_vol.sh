#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=2-00
#SBATCH --job-name=crystal_gnn
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/pstjohn/gpu.%j.out

source ~/.bashrc
module load cudnn/8.1.1/cuda-11.2
module load gcc
conda activate rlmol

srun python train_model_vol.py
