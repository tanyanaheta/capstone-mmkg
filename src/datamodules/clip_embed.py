import os
import torch
import clip
import hydra
# from omegaconf import DictConfig, OmegaConf
from src.datamodules.mscoco import MSCOCODataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(cfg.clip.vision_model, device=device)


    print(os.listdir())

    ds = MSCOCODataset(
        root=os.path.join(cfg.coco.root, cfg.coco.partition),
        annFile=os.path.join(cfg.coco.root, "annotations", f"instances_{cfg.coco.partition}.json"),
        transform=preprocess
    )

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.coco.batch_size,
        num_workers=cfg.coco.num_workers
    )

    with torch.no_grad():
        for imgs in dataloader:
            embeds = model.encode_image(imgs)
            print(embeds)



if __name__ == "__main__":
    main()
