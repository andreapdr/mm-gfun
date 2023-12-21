import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import lightning as L
import logging

from omegaconf import DictConfig

from utils.instantiators import init_callbacks, init_loggers


log = logging.getLogger(name=__name__)

@hydra.main(version_base="1.3", config_path="../configs/", config_name="train.yaml")
def main(cfg: DictConfig):

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    log.info(datamodule)

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    log.info(f"Instantiating model <{cfg.models._target_}>")
    model = hydra.utils.instantiate(cfg.models, num_labels=datamodule.num_labels)

    log.info(f"Instantiating callbacks...")
    callbacks = init_callbacks(cfg.get("callbacks"))
    log.info(f"Instantiating loggers...")
    loggers = init_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    log.info(f"Starting training...")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # TODO test phase

if __name__ == "__main__":
    main()