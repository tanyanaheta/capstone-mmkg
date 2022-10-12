# NYU-Zillow-Capstone-2022-Team-A
NYU-Zillow-Capstone-2022-Team-A

### Data Configuration (MSCOCO):
Due to its large size, MS COCO data is not included in this repository should be downloaded locally. To download zip files of the validation set of images  (5000 images total) and annotations respectively, you can run the following commands in your project directory:

```
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
```

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
