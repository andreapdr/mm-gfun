import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import lightning as L
import torch

from src.data.gfun_datamodule import MMgFunDataModule
from src.models.gfun import GFunMM
from src.models.text_model import AutoTextClassificationLightningModule
from src.models.vision_model import AutoImageClassificationLightningModule

from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path="../configs/", config_name="devel-gfun.yaml")
def main(cfg: DictConfig):
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    print(datamodule)

    text_model = AutoTextClassificationLightningModule.load_from_checkpoint("logs/train-textual/runs/2023-12-28_10-35-01/csv/version_0/checkpoints/epoch=9-step=31250.ckpt",)
    vision_model = AutoImageClassificationLightningModule.load_from_checkpoint("logs/train-vision/runs/2023-12-19_15-42-59/csv/version_0/checkpoints/epoch=8-step=112500.ckpt")
    # text_model = None
    # vision_model = None

    train_dataloader = datamodule.train_dataloader()

    gfun = GFunMM(
        text_model=text_model,
        vision_model=vision_model,
        metaclassifier="metaclassifier",
        classification_type="single-label",
    )

    gfun.first_forward(dataloader=train_dataloader, cache=True, debug=False)


if __name__ == "__main__":
    main()