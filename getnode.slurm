#!/bin/bash
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/%u/projects/NYU-Zillow-Capstone-2022-Team-A/logs/%x.out
#SBATCH --error=/scratch/%u/projects/NYU-Zillow-Capstone-2022-Team-A/logs/%x.err
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=256G 
#SBATCH -c 8

sleep 4h
