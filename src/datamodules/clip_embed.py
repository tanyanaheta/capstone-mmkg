import os

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from mscoco import MSCOCODataset
import pandas as pd
import pyrootutils
from tqdm import tqdm
from PIL import Image


def get_img_id(filename):
    return int(filename.split('.')[0])


def add_scene_data(img_txt_data_file='data/coco_data/image_text_data.csv', scene_data_file='coco_scene_data_nyu.csv'):
    img_txt_data = pd.read_csv(img_txt_data_file)
    scene_data = pd.read_csv(scene_data_file)

    scene_data_sorted = scene_data.sort_values('image_path', ascending=True).reset_index(drop=True)
    scene_data_sorted['coco_img_id'] = scene_data_sorted['image_path'].apply(lambda x: get_img_id(x))
    scene_data_sorted_ids = scene_data_sorted[['coco_img_id', 'scene_hash']]

    img_txt_data_scenes = img_txt_data.merge(scene_data_sorted_ids, how='inner', on='coco_img_id')
    img_txt_data_scenes.to_csv('image_text_data.csv', index=False)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('#'*40)
    print('device:', device)
    print('#'*40)
    tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip.model_id)
    processor = CLIPProcessor.from_pretrained(cfg.clip.model_id)
    model = CLIPModel.from_pretrained(cfg.clip.model_id).to(device)

    ds = MSCOCODataset(
        root=os.path.join(cfg.coco.root, cfg.coco.partition),
        annFile=os.path.join(cfg.coco.root, "annotations", f"instances_{cfg.coco.partition}.json")
    )

    ann_items = ds.coco.loadCats(ds.coco.getCatIds())
    tags = [item['name'] for item in ann_items]
    tag_ids = [item['id'] for item in ann_items]

    images_tensor = None
    tags_tensor = None

    with torch.no_grad():
        for i in tqdm(range(0, len(ds.ids), cfg.coco.batch_size), desc='encoding image batches'):
            batch_ids = ds.ids[i:i+cfg.coco.batch_size]
            batch_imgs = [Image.open('../../coco/images/'+ds.coco.loadImgs(id)[0]["file_name"]).convert('RGB') for id in batch_ids]
            batch = processor(
                        text=None,
                        images=batch_imgs,
                        return_tensors='pt',
                        padding=True
                    )['pixel_values'].to(device)
            batch_emb = model.get_image_features(pixel_values=batch)
            batch_emb = batch_emb.squeeze(0)
            
            if images_tensor is None:
                images_tensor = batch_emb
            else:
                images_tensor = torch.cat((images_tensor, batch_emb), dim=0)

        for tag in tqdm(tags, desc='encoding tags'):
            inputs = tokenizer(tag, return_tensors="pt").to(device)
            tag_emb = model.get_text_features(**inputs)

            if tags_tensor is None:
                tags_tensor = tag_emb
            else:
                tags_tensor = torch.cat((tags_tensor, tag_emb), dim=0)  
    
    torch.save(images_tensor, cfg.data.mscoco.image_embeds)
    torch.save(tags_tensor, cfg.data.mscoco.keyword_embeds)

    ds.get_img_text_data()
    ds.img_text_data.to_csv(cfg.data.mscoco.connections, index=False)
    add_scene_data()



if __name__ == "__main__":
    root_path = pyrootutils.find_root(search_from=__file__, indicator=".git")
    print("Set WD location to", root_path)
    pyrootutils.set_root(
        path=root_path,
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=True,
    )

    main()
