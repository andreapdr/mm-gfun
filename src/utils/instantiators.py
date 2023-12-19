from typing import List

import logging
import hydra

from omegaconf import DictConfig
from lightning import Callback
from lightning.pytorch.loggers import Logger

log = logging.getLogger(name=__name__)

def init_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.info("No callback configs found! Skipping..")
        return callbacks
    
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def init_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []

    if not logger_cfg:
        log.info("No logger configs found! Skipping..")
        return loggers 

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
    
    return loggers