#!/bin/bash
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/%u/projects/NYU-Zillow-Capstone-2022-Team-A/logs/%x.out
#SBATCH --error=/scratch/%u/projects/NYU-Zillow-Capstone-2022-Team-A/logs/%x.err
#SBATCH --export=ALL
#SBATCH --time=02:30:00
#SBATCH -c 2

rsync --info=progress2 /scratch/$USER/data/zillow.sqsh $TMPDIR
DATA_DIRECTORY=$TMPDIR INSTANCE_NAME=$INSTANCE_NAME  ./start_singularity_instance.sh
sleep 3h
