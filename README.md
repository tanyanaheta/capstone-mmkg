# Multimodal Graph Induction: Project Respository

This is the project repository for Team A within the NYU-Zillow Capstone collaboration. The code in this repository was produced by Adi Srikanth, Andre Chen, David Roth, and Tanya Naheta. This project was originally built on proprietary information provided by Zillow Group. As such, all data from Zillow has been omitted from this public repository, and we provide results on a similarly formatted open-source multimodal dataset ([MSCOCO](https://cocodataset.org/#download)) instead. 

This project was sanctioned and facilitated by the NYU Center for Data Science with mentorship from the Zillow Applied Science team.

# Executive Summary
 
This project applied GraphSAGE, an inductive graph neural network, to multimodal representation learning using real estate listing data from Zillow. We constructed a multi-modal knowledge graph (MMKG) with CLIP-initialized image and text embeddings, and trained a two-layer GraphSAGE model with mean aggregation and contrastive loss. To simulate inference on unseen data, we tested three node reconnection strategies. Link prediction tasks on MS-COCO and Zillow datasets evaluated performance. Results showed improved recall and embedding quality on human-labeled data, highlighting the potential of GNNs for multimodal search, keyword attribution, and recommendation systems in production settings.

# Setup and Environment Overview 

<ins>Table of Contents</ins>

| Section of Project                | Relevant Files                                               |
|-----------------------------------|--------------------------------------------------------------|
| Configuration                     | `conf/config.yaml`                                           |
| Data Processing                   | `src/datamodules/clip_embed.py`                              |
| Graph Generation - Initialization | `src/datamodules/build_graph.py`                             |
| Graph Generation - Training       | `train_graphsage.py`                                         |
| Graph Generation - Validation     | `notebooks/validation_exp_all.ipynb`                         |
| Link Prediction                   | `notebooks/validation_exp_all.ipynb`                         |
| Graph Objects                     | `graph/*`                                                    |
| Stored Data                       | `data/*`                                                     |

## Packages 

A yml file containing the conda environment used for this project can be found in the `conf` directory. 

## Data Access

Due to their large sizes, datasets are not included in this repo and should be downloaded locally. Instructions for each dataset are provided below:

### Zillow Development Dataset
Zillow development data can be downloaded from this [private Google Drive location](https://drive.google.com/drive/u/0/folders/1lRgFdKi_74Q3a3qLudOrpc6Nd60vGcCZ). 

The Google Drive folder also contains a high-level data dictionary.

### Zillow Test Dataset
Zillow test data can be downloaded from this [private Google Drive location](https://drive.google.com/drive/u/0/folders/17qjTIMBwEmwWSAXvXUAIwLj0C8CfU_zf).

### MSCOCO
To download zip files of the validation set of images (5000 images total) and annotations respectively, you can run the following commands in your project directory:

```
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
```

## Data configuration:

1. Upload the file contents of the Zillow Dataset to a folder called "zillow_data" (create this folder for yourself) at the root level of `NYU-Zillow-Capstone-2022-Team-A`.
3. Run the following bash command: `mksquashfs zillow_data zillow.sqsh; mkdir -p /scratch/$USER/data_zillow; mv zillow.sqsh scratch/$USER/data_zillow`

## Directory Organization 
After following instructions as laid out above, directory should be structured as shown below (unimpacted project files omitted):

```
[parent or home directory]
│
└───tmp **(CREATED)**
│
└───NYU-Zillow-Capstone-2022-Team-A
    |   baseline.py
    |   train_graphsage.py
    |   .gitignore
    |   README
    │   ...
    │    ..
    |
    └───NYU_HPC
    |   |   launch_job.sh
    |   |   launch.slurm
    |   |   getnode.slurn
    |   └─── scripts
    |   |   │   layer_setup.sh
    |   |   │   overlay-base.ext3 **(CREATED)**
    |   |   │   overlay-packages.ext3 **(CREATED)**
    |   |   │   overlay-temp.ext3 **(CREATED)**
    |   |   |   start_singularity_instance.sh
    |   |   │   singularity_setup.sh
    |   |   │   launch.slurm
    |   |   │   getnode.slurm
    |   |   |   ..
    │   └─── overlays
    |        | ...
    |        | ..    
    |
    └─── data
    |   └───zillow_data **(CREATED)**
    │       │   image_embed.joblib **(UPLOADED)**
    │       │   keyword_embed.joblib **(UPLOADED)**
    │       │   NYU_photoboard_file.csv **(UPLOADED)**
    │       │   scene_embed.joblib **(UPLOADED)**
    │       |   ..
    |   └───zillow_verified_data **(CREATED)**
    │       │   image_embed.joblib **(UPLOADED)**
    │       │   keyword_embed.joblib **(UPLOADED)**
    │       │   NYU_photoboard_file.csv **(UPLOADED)**
    │       │   scene_embed.joblib **(UPLOADED)**
    |   └───coco_data **(CREATED)**
    |       |   ...
    |       |   ..
    |
    └───src
    |   |
    |   └───datamodules
    |    |  |
    |    |  |   build_graph.py
    |    |  |   cnnx_experiment.py
    |    |  |   graph_utils.py
    |    |  |   clip_embed.py
    |    |  |   ..   
    |    |  └─── ...
    |    |        
    |    └───model
    |    |   | SAGE.py 
    |    |   | ..
    |    |
    └───notebooks
    │   │   validation_exp_all.ipynb
    │   |   ...
    |   |   ..
    └───graph
    |   │           
    |   └───coco_graph_csv
    |   |   | *_edges.csv
    |   |   | *_nodes.csv
    |   |   | meta.yaml
    |   |   | ..
    |   |
    |   └───zillow_graph_csv_images_975
    |   |   | *_edges.csv
    |   |   | *_nodes.csv
    |   |   | meta.yaml
    |   |   | .. 
    |   |
    |   └───zillow_verified_graph_csv_images_975
    |   |   | *_edges.csv
    |   |   | *_nodes.csv
    |   |   | meta.yaml   
    |   |   | ..
    |
    └───conf
    |   | config.yaml
    |   | ..
    |
    └───previous_work
    |   | ... 
    |   | ..
    |
    exprmt_metrics
        | ...
        | ..
```

# Project Components 

We break our project components down using the following sections (listed in order): 

- Data processing (COCO only)
- Graph Generation (sub parts: Graph Initialization, Graph Training, and Graph Validation)
- Link Prediction 

## Data Processing 

If using MS COCO data, the script `src/datamodules/clip_embed.py` will produce the necessary CLIP embeddings for MS COCO. No additional parameters are neccesary. 

## Graph Generation 

We take in images and keywords from either the COCO or Zillow Dataset. Using these, we initialize a graph using the DGL library. Next, we train the graph using GraphSAGE in order to update node embeddings. We elaborate below: 

### Graph Initialization 

The script `src/datamodules/build_graph.py` initializes the graph. This script does not take command line arguments because of the use of the Hydra Main Wrapper. Instead, arguments passed to the Hydra Main Wrapper can be directly modified using the `main_wrapper()` function call within `src/datamodules/build_graph.py`. The arguments for the function call are: 
- org (str): default="zillow" | also accepts "zillow_verified" for human verified labels and "coco" for MS COCO data
- scenes (bool): default=True | whether or not to include scene connections 
- new_edge_mode (str): default=None | accepts "images" or "keywords" in order to enable new edge generation within one type of node during graph building
- sim_threshold (float): default=None | cosine similarity threshold for creating edges between nodes if new_edge_mode != None (values used to produce our results are included in commented out lines in the bottom of the script)
- new_edges_batch_size (int): default=500 | batch size for adding new edges (useful on local since process of adding new edges is memory intensive) 

Running this script to completion produces an initialized graph to the graph directory specified in the `conf/config.yaml`. 

### Graph Training 

_Known Bugs: the validation() and baseline() methods are deprecated and therefore commented out in train_graphsage.py. This code can be run in the Jupyter Notebook `notebooks/validation_exp_all.ipynb` if need be. However, running the python file below will still complete training._

The script `train_graphsage.py` trains an initialized graph using GraphSAGE. The config file `conf/config.yaml` defines the dataset/graph to be trained. The class `SAGELightning` defines the parameters for the GNN used in training. 

Notably, while the conf/config.yaml file defines various paths for different graphs, the org (required) and graph type (bipartite or non-bipartite, optional) must be specified. An example is shown below (parameters only need to be specified in the train_wrapper() function call) in the case where we want to use a non-bipartite graph that has already been created using build_graph.py, where new edges are created between nodes of the same modality if the cosine similarity of their embeddings is > 0.975. 

```
if __name__ == "__main__":
    train_wrapper(org='zillow', pre_connect_threshold=0.975)
```

If we want to use ae bipartite graph, we can omit the `pre_connect_threshold` parameter.

Running this script to completion trains the graph passed in. A saved file of the trained graph is stored as saved_model_[org_name + other configs].pt

### Graph Validation 

Graph Validation can be done in the Jupyter notebook `validation_exp_all.ipynb`. 

## Link Prediction

### Modularization Attempt 

The script `src/datamodules/cnnx_experiment.py` is an attempt to modularize a portion of the `notebooks/validation_exp_all.ipynb`. Development was halted in favor of the notebook (for rapid development). However, if one choose to modularize the notebook, much of the code from the aforementioned python file can be reused. 

### Link Prediction Experiments

The link prediction experiment is handled in `validation_exp_all.ipynb`. Notably, this notebook contains code to run our three variants of link prediction. The variants are defined by the method with which validation nodes are reconnected to the full graph to conduct full-graph link prediction. These methods are (1) reconnection via cosine similarity, (2) reconnection via scene connection, and (3) reconnection via self-loop (or self connect). 

The notebook is quite extensive, therefore, documentation for the notebook is done in-notebook. 

### Top K Experiment 

We attempted to improve Link Prediction by limiting each node to K connections during link prediction. This did not seem to improve link prediction metrics, but a variant of this method could. Code for this is available at the very end of the aforementioned Jupyter Notebook. 

## Saved Graphs 

A saved version of a trained GraphSAGE graph is stored as `model_saved.py`

## Known Bugs 

- `train_graphsage.py`: Methods validation() and baseline() have errors stemming from outdated dataloader
- `src/datamodules/cnnx_experiment.py`: trainer.fit() throwing CUDA error
