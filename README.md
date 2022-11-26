# NYU-Zillow-Capstone-2022-Team-A

## Setup and Environment Overview 

### Getting the Data 
Due to their large sizes, datasets are not included in this repo and should be downloaded locally. Instructions for each dataset are provided below:

1. Zillow Dataset:
Zillow data can be downloaded from this [private Google Drive location](https://drive.google.com/drive/u/0/folders/1lRgFdKi_74Q3a3qLudOrpc6Nd60vGcCZ). The Google Drive folder also contains a high-level data dictionary.


2. MSCOCO
To download zip files of the validation set of images (5000 images total) and annotations respectively, you can run the following commands in your project directory:

```
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
```

### Cluster Environment Setup 
Be sure you've run the following steps in order to work in our singularity instance (here we assume we are working with Zillow data, not MS COCO).

#### A) Cluster setup:
1. SSH into a Greene compute node or GCP burst node and ensure you have the contents of this repository available in your scratch/[your_username]/ directory. ***Confirm you are in the root level of NYU-Zillow-Capstone-2022-Team-A before proceeding with the rest of the instructions***.

#### B) Data configuration:
2. Upload the file contents of the Zillow Dataset to a folder called "zillow_data" (create this folder for yourself) at the root level of NYU-Zillow-Capstone-2022-Team-A.
3. Run the following bash command: `mksquashfs zillow_data zillow.sqsh; mkdir -p /scratch/$USER/data_zillow; mv zillow.sqsh scratch/$USER/data_zillow`

#### C) Overlays and Singularity Startup
4. Run `source layer_setup.sh` to automatically create overlays and launch Singularity instance.
5. Run `singularity instance list` and ensure that an instance named "mycontainer" is running.

#### D) Access Singularity instance
6. SSH directly into the Singularity container (via VSCode or Terminal)
7. Run `source singularity_setup.sh` to initialize the conda environment in the running Singularity instance.

From this point, you should have all the dependencies needed to complete development work. If not, please make a note of additional dependencies to add to our scripts/create_package_overlay.sh file.

## Data Organization 
After following instructions as laid out above, your scratch/[username] directory should be structured as shown below (unimpacted project files omitted):

```
[scratch/your_username]
│
└───tmp **(CREATED)**
│
└───NYU-Zillow-Capstone-2022-Team-A
    │   layer_setup.sh
    │   overlay-base.ext3 **(CREATED)**
    │   overlay-packages.ext3 **(CREATED)**
    │   overlay-temp.ext3 **(CREATED)**
    |   start_singularity_instance.sh
    │   singularity_setup.sh
    │   launch.slurm
    │   getnode.slurm
    │   ...
    │
    └───scripts
    │   │   create_base_overlay.sh
    │   │   create_package_overlay.sh
    │       
    └───zillow_data **(CREATED)**
    │   │   image_embed.joblib **(UPLOADED)**
    │   │   keyword_embed.joblib **(UPLOADED)**
    │   │   NYU_photoboard_file.csv **(UPLOADED)**
    │   │   scene_embed.joblib **(UPLOADED)**
    │   
    └───src
        |
        └───datamodules
                      |
                      └───build_graph
                      |
                      └───cnnx_experiment
                      |
                      └─── ...
    │   
    └───notebooks
    │   │   ...  
    │   
    └───graph
        │           
        |
        └───coco_graph_csv
        |
        |
        |
        └───zillow_graph_csv_images_975
        |
        |
        |
        └───zillow_verified_graph_csv_images_975
    │
    └───data
        |
        |
        └───coco_data
        |
        |
        |
        └───zillow_data
        |
        |
        |
        └───zillow_verified_data   
    |
    |
    └───conf
    
```
## Components 

### Exploratory EDA 
Some exploration of the MS COCO dataset, available in the following Jupyter Notebook 
```
coco_explore.ipynb
```

### Graph Generation 
Generating a connected graph using the MS COCO images and image tags. Work is done in a Jupyter Notebook, calling custom functions available in two python files. The Jupyter Notebook writes a graph to the graph_csv directory. 
```
embeddings_and_graph_generation.ipynb
mscoco.py
clip_embed.py
```

### GraphSAGE Integration 
Implementing Embedding Inference using the GraphSAGE architecture. Work is done in a Jupyter Notebook, calling custom functions from one python file. The Jupyter Notebook takes in the graph from the graph_csv directory generated in the Graph Generation component. 
```
DGL_to_SAGE.ipynb
graph_training.py
```

### Link Prediction + Reconnection Experimentation 

```
src/datamodules/cnnx_experiment.py
notebooks/validation_exp_all.ipynb
```
