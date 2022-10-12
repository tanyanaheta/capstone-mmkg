from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO
import torchvision.io as io
from PIL import Image
import os.path


def get_img_tags(img_id: int, coco) -> list:
    '''
    Helper function that takes in an image ID from the COCO class instance and returns all 
    associated object tags identified in the corresponding image.
    '''

    annotations = coco.imgToAnns[img_id]
    category_ids = [annotations[i]['category_id'] for i in range(len(annotations))]
    category_names = [item['name'] for item in coco.loadCats(category_ids)]
    return [list(set(category_ids)), list(set(category_names))]


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

    def get_img_text_data(self):
        '''
        Builds a reference table using dictionary that relates image IDs in the COCO dataset with 
        their local URL (coco/images/{image_filename}.jpg) and corresponding text data (tags / captions).
        '''
        img_text_data = {}
        for i, id in enumerate(self.ids):
            img_text_data[id] = {'img_embed_row': i}

            tag_ids, tag_names = get_img_tags(id, self.coco)
            img_text_data[id]['tag_ids'] = tag_ids
            img_text_data[id]['tag_embed_rows'] = [x-1 for x in tag_ids]
            img_text_data[id]['tag_names'] = tag_names
        
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

