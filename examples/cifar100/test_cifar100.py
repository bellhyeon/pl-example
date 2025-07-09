import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from data.dataset import ImageClassificationDataset
from models.runner_module import TrainingModule
from download_cifar100 import load_cifar100


def setup_logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


# @torch.inference_mode()
def main(cfg: DictConfig) -> None:
    """
    Main testing entry point.
    Loads config, prepares data, initializes model and runs inference.
    """
    logger = setup_logger()

    # Load model from checkpoint
    model = TrainingModule.load_from_checkpoint(cfg.checkpoint_path, cfg=cfg)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Model loaded and set to eval mode.")

    # Load test data
    test_paths, test_labels = load_cifar100("./dataset/cifar100/test")
    logger.info(f"Sample test paths: {test_paths[:5]}")
    logger.info(f"Sample test labels: {test_labels[:5]}")

    test_dataset = ImageClassificationDataset(
        test_paths, None, use_augmentation=False, img_size=224
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    preds = []

    # Inference loop
    with torch.no_grad():
        for batch in test_loader:
            batch["input"] = batch["input"].to(device)
            pred = model.inference(batch)
            preds.extend(torch.argmax(pred, dim=1).cpu().numpy())
    preds = np.array(preds)
    logger.info(f"Predictions: {preds[:10]}")

    # Accuracy calculation
    acc = np.mean(preds == np.array(test_labels))
    logger.info(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load("configs/base.yaml")
    args = OmegaConf.create(vars(args))

    cfg.merge_with(args)
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            "Loaded config is not a DictConfig. Please check your config file."
        )
    main(cfg)
