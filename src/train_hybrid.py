import hydra
import os
import torch
import rootutils
import lightning as L
from typing import List
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import OmegaConf

# import data_config as data_config
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.htr_datamodule import HTRDataModule
from src.models.hybrid_module import HybridModule

torch.autograd.set_detect_anomaly(True) 

print(f'Importing modules...')

from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)

from src.utils.instantiators import instantiate_data_configs, instantiate_tokenizers

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:

    # Set all seeds for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Init wandb logger and project
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Instantiating datamodule
    log.info("Instantiating DataModule...")
    data_configs = instantiate_data_configs(cfg.get("data"))
    log.info(f'TRAIN, VAL, TEST DATA_CONFIGS INSTANTIATED: {data_configs}')
    log.info(f'TRAIN: {data_configs["train_config"]}')
    log.info(f'VAL: {data_configs["val_config"]}')
    log.info(f'TEST: {data_configs["test_config"]}')

    # Update wandb logger with data config
    logger[0].experiment.config.update(
        OmegaConf.to_object(cfg.get("data").get("train_config"))
    )

    # Instantiating tokenizer
    tokenizer = instantiate_tokenizers(cfg.get("tokenizer"))

    print(f'TOKENIZER: {tokenizer}')
    

    # Init data module
    log.info("Instantiating DataModule...")
    datamodule: LightningDataModule = HTRDataModule(
        train_config=data_configs["train_config"],
        val_config=data_configs["val_config"],
        test_config=data_configs["test_config"],
        tokenizer=tokenizer,
        seed=cfg.get("seed"),
    )
    log.info(f'DATAMODULE INSTANTIATED: {datamodule}')

    # Setup data module
    log.info("Setting up DataModule TRAIN AND VAL...")
    datamodule.setup(stage="fit")

    log.info("Setting up DataModule TEST...")
    datamodule.setup(stage="test")

    print(f'Instantiating model...')

    # Manually instantiate model components to avoid Hydra recursively instantiating data configs
    net = hydra.utils.instantiate(cfg.model.net)
    optimizer_partial = hydra.utils.instantiate(cfg.model.optimizer)
    scheduler_partial = hydra.utils.instantiate(cfg.model.scheduler)
    # Instantiate logger mapping from logger config
    logger_map = {}
    for name, lg_conf in cfg.logger.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger_map[name] = hydra.utils.instantiate(lg_conf)
    # Create the model directly
    model = HybridModule(
        net=net,
        optimizer=optimizer_partial,
        scheduler=scheduler_partial,
        compile=cfg.model.compile,
        _logger=logger_map,
        datasets=data_configs,
        tokenizer=tokenizer,
        log_val_metrics=cfg.model.log_val_metrics,
    )
    print(f'MODEL INSTANTIATED: {model}')

    # Update wandb logger with model config
    logger[0].experiment.config.update(
        OmegaConf.to_object(cfg.model)
    )

    # Predict on test set
    log.info("Predicting on test set...")
    trainer_cfg = cfg.get("trainer")
    trainer: Trainer = hydra.utils.instantiate(
        trainer_cfg, logger=logger, callbacks=instantiate_callbacks(cfg.get("callbacks"))
    )

    # Load from a pretrained_checkpoint
    ckpt_path = cfg.callbacks.model_checkpoint_base.dirpath + cfg.get("pretrained_checkpoint") + '.ckpt' if cfg.get("pretrained_checkpoint") else None
    
    # if ckpt_path exists, load the model from the checkpoint
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f'CHECKPOINT PATH EXISTS: {ckpt_path}')
        print(f'MODEL WILL BE LOADED FROM CHECKPOINT: {model}')
    else:
        print(f'CHECKPOINT PATH DOES NOT EXIST: {ckpt_path}')
        print(f'MODEL WILL BE TRAINED FROM SCRATCH: {model}')
        ckpt_path = None
      
    model = HybridModule.load_from_checkpoint(ckpt_path, net=model.net, datasets=cfg.get("data"), tokenizer=tokenizer) if ckpt_path is not None else model
    # model = torch.compile(model)

    if cfg.train is True:
        print(f'TRAINING MODEL: {model}')
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
        trainer.test(model, datamodule.test_dataloader())
    else:
        print(f'MODEL WILL NOT BE TRAINED: {model}. Only testing will be performed.')
        trainer.validate(model, datamodule.val_dataloader())
        trainer.test(model, datamodule.test_dataloader())

@hydra.main(version_base="1.3", config_path="../configs", config_name="train_htr.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    print(f'Main for training HTR models for HTR!')
    extras(cfg)
    # train the model
    _ = train(cfg)

    return None


if __name__ == "__main__":
    main()