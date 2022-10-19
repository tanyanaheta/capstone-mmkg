# NYU-Zillow-Capstone-2022-Team-A

## Setup and Environment Overview 

### Data Configuration (MSCOCO):
Due to its large size, MS COCO data is not included in this repository should be downloaded locally. To download zip files of the validation set of images  (5000 images total) and annotations respectively, you can run the following commands in your project directory:

```
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
```

### Environment Setup 
There are scripts available to automate both the creation of a singularity instance with the properly configured overlays AND the activation of a conda environment once on the singularity that installs dgl. The scripts are listed below (you may have to run chmod +x script_name to execute the script).

```
layer_setup.sh
singularity_setup.sh
```

### Data Organization 
To work with current data processing modules (and in alignment with MS COCO's recommended directory structure), MS COCO data should be saved into the "coco" folder in your project directory and organized relative to other files in the project directory as shown below.

```
NYU-Zillow-Capstone-2022-Team-A
│   clip_embed.py
│   ...
│
└───src
│   └───datamodules
│       │   mscoco.py
│       
└───conf
│   │   config.yaml
│       
└───coco
│   └───annotations
│   │   │   captions_train2017.json
│   │   │   captions_val2017.json
│   │   │   instances_train2017.json
│   │   │   instances_val2017.json
│   │   │   ...
│   │
│   └───images
│       │   000000000139.jpg
│       │   000000000285.jpg
│       │   ...
│   ...    

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