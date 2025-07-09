import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import logging
import argparse
from omegaconf import DictConfig, OmegaConf
from models.runner_module import TrainingModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from download_cifar100 import load_cifar100
from sklearn.model_selection import train_test_split
from data.data_module import ImageClassificationDataModule


def main(cfg: DictConfig) -> None:
    """
    Main training entry point.
    Loads config, prepares data, initializes model and trainer, and starts training.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration
    logger.info("Configuration loaded.")

    # Load and split CIFAR-100 data
    paths, labels = load_cifar100("./dataset/cifar100/train")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=cfg.train.seed, shuffle=True
    )
    logger.info(f"Sample train paths: {train_paths[:5]}")
    logger.info(f"Sample train labels: {train_labels[:5]}")
    logger.info(f"Sample val paths: {val_paths[:5]}")
    logger.info(f"Sample val labels: {val_labels[:5]}")

    # Initialize model and data module
    model = TrainingModule(cfg)
    data_module = ImageClassificationDataModule(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        **cfg.data,
    )

    # Trainer setup
    progress_bar = TQDMProgressBar(refresh_rate=1)
    trainer = pl.Trainer(
        accelerator="cuda",
        devices="auto",
        strategy="ddp",
        log_every_n_steps=cfg.train.log_every_n_steps,
        max_epochs=cfg.train.epochs,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        val_check_interval=min(cfg.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(cfg.train.validation_interval), 1),
        callbacks=[progress_bar],
    )

    # Start training
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            "Loaded config is not a DictConfig. Please check your config file."
        )
    main(cfg)
