#!/bin/bash
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/%u/projects/NYU-Zillow-Capstone-2022-Team-A/logs/%x.out
#SBATCH --error=/scratch/%u/projects/NYU-Zillow-Capstone-2022-Team-A/logs/%x.err
#SBATCH --export=ALL
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G 
#SBATCH -c 8

sleep 3h
