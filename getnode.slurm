#!/bin/bash
#SBATCH --job-name=capstone_cv
#SBATCH --open-mode=append
#SBATCH --output=/scratch/alc9635/cds-bootcamp/homework/cv/logs/capstone_cv.out
#SBATCH --error=/scratch/alc9635/cds-bootcamp/homework/cv/logs/capstone_cv.err
#SBATCH --export=ALL
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -c 8
sleep 3h
