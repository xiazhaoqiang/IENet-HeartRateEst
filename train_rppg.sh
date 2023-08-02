#!/bin/bash
#SBATCH --job-name=lili
#SBATCH --account=Project_2001654
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100:1,nvme:180

$SCRATCH

#mem-per-cpu=4G

module load pytorch

srun nvidia-smi

python train_rppg.py