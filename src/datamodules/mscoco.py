from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO
from PIL import Image
import os.path

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

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
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

