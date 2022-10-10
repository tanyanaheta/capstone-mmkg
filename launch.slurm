#!/bin/bash
#SBATCH --job-name=zillow_MMKG
#SBATCH --open-mode=append
#SBATCH --output=/home/%u/scratch/projects/NYU-Zillow-Capstone-2022-Team-A/logs/%x_%j.out
#SBATCH --error=/home/%u/scratch/projects/NYU-Zillow-Capstone-2022-Team-A/logs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8

rsync --info=progress2 /home/dsr331/scratch/public/coco.sqsh $TMPDIR
DATA_DIRECTORY=$TMPDIR INSTANCE_NAME=zillow_clip ./scripts/start_singularity_instance.sh
singularity run --containall --pwd=$PWD instance://zillow_clip conda run -p /ext3/conda/zillow_MMKG --no-capture-output python clip_embed.py