#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --output=/scratch/%u/projects/NYU-Zillow-Capstone-2022-Team-A/logs/%x_%j.out
#SBATCH --error=/scratch/%u/projects/NYU-Zillow-Capstone-2022-Team-A/logs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8

# rsync --info=progress2 /scratch/$USER/data/zillow.sqsh $TMPDIR
DATA_DIRECTORY=$TMPDIR INSTANCE_NAME=$INSTANCE_NAME scripts/start_singularity_instance.sh
singularity run --containall --pwd=$PWD instance://$INSTANCE_NAME conda run -p /ext3/conda/zillow_MMKG --no-capture-output python train_graphsage.py
