from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO
import torchvision.io as io
from PIL import Image
import os.path
import pandas as pd
import numpy as np
import pyrootutils


root_path = pyrootutils.find_root(search_from=__file__, indicator=".git")
print("Set WD location to", root_path)
pyrootutils.set_root(
    path=root_path,
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

class MSCOCODataset(VisionDataset):
    """
    A torch.utils.data.Dataset wrapper for MSCOCO (2017)
    """
    def __init__(
        self, 
        root, 
        annFile, 
        transform=None, 
        target_transform=None,
        transforms=None):
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def get_img_tags(self, img_id: int) -> list:
        '''
        Helper function that takes in an image ID from the COCO class instance and returns all 
        associated object tags identified in the corresponding image.
        '''

        annotations = self.coco.imgToAnns[img_id]
        category_ids = list(set([annotations[i]['category_id'] for i in range(len(annotations))]))
        category_names = [item['name'] for item in self.coco.loadCats(category_ids)]
        return [category_ids, category_names]
    
    def get_img_text_data(self):
        '''
        Builds a reference table using dictionary that relates image IDs in the COCO dataset with 
        their local URL (coco/images/{image_filename}.jpg) and corresponding text data (tags / captions).
        '''
        img_text_data = pd.DataFrame()
        all_tag_ids = [item['id'] for item in self.coco.loadCats(self.coco.getCatIds())]

        for i, id in enumerate(self.ids):
            tag_ids, tag_names = self.get_img_tags(id)
            tag_id_idxs = np.array([all_tag_ids.index(tag_id) for tag_id in tag_ids]) # Get row position of tag_ids in tag embeddings matrix
            tag_id_idxs_offset = tag_id_idxs + len(self.ids) # these are our new tag ids: they encode the position of the tag's embedding in embedding matrix, offset by number of image embeddings (since we will concatenate them)
            row = [{'img_id': i, 'coco_img_id': id, 'tag_ids': tag_id_idxs_offset.tolist(), 'tag_names': tag_names}]
            img_text_data = pd.concat([img_text_data, pd.DataFrame(row)])
        
        self.img_text_data = img_text_data
    
    def _load_image(self, id):
        path = 'images/' + self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image
    
    def __len__(self):
        return len(self.ids)

