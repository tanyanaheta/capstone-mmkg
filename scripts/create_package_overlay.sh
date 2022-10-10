#! /bin/bash

set -e

# Set this to the directory containing empty overlay images
# Note: on GCP the overlay directory does not exist
OVERLAY_DIRECTORY=/scratch/work/public/overlay-fs-ext3/
if [[ ! -d $OVERLAY_DIRECTORY ]]; then
OVERLAY_DIRECTORY=/scratch/wz2247/singularity/overlays/
fi

IMAGE_DIRECTORY=/scratch/wz2247/singularity/images/

# Set this to the overlay to use for additional packages.
ADDITIONAL_PACKAGES_OVERLAY=overlay-10GB-400K.ext3

# We will install our own packages in an additional overlay
# So that we can easily reinstall packages as needed without
# having to clone the base environment again.
echo "Extracting additional package overlay"
mkdir -p overlays
cp $OVERLAY_DIRECTORY/$ADDITIONAL_PACKAGES_OVERLAY.gz ./overlays/
gunzip ./overlays/$ADDITIONAL_PACKAGES_OVERLAY.gz
mv ./overlays/$ADDITIONAL_PACKAGES_OVERLAY ./overlays/overlay-packages.ext3


# We now execute the commands to install the packages that we need.
echo "Installing additional packages"
singularity exec --containall --no-home -B $HOME/.ssh \
    --overlay overlays/overlay-packages.ext3 \
    --overlay overlays/overlay-base.ext3:ro \
    $IMAGE_DIRECTORY/pytorch_22.08-py3.sif /bin/bash << 'EOF'
source ~/.bashrc
conda activate /ext3/conda/zillow_MMKG
conda install -y pytest
conda install -c conda-forge -y hydra-core omegaconf
conda install -c pytorch -y torchvision cudatoolkit=11.0
pip install ftfy regex tqdm pytorch-lightning pycocotools
pip install git+https://github.com/openai/CLIP.git
EOF
