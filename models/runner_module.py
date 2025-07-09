from pytorch_lightning import LightningModule
from typing import Dict, Any
from timm.optim._optim_factory import create_optimizer_v2
from models.model.modeling import Model
from utils.loss_factory import create_loss
from utils.metrics import accuracy
from utils.scheduler_factory import create_scheduler
from omegaconf import DictConfig


class TrainingModule(LightningModule):
    """
    PyTorch LightningModule for training and validation.
    Handles model, loss, optimizer, and scheduler setup.
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Configuration object (OmegaConf DictConfig or similar).
        """
        super().__init__()
        self.cfg = cfg

        # Model and loss function setup
        self.model = Model.get_model(**cfg.model)
        self.loss_func = create_loss(**cfg.loss)

    def forward(self, item: Dict[str, Any]):
        """
        Forward pass for a batch.
        Args:
            item: Dict with "input" and "target" tensors.
        Returns:
            Tuple of (loss, accuracy)
        """
        img = item["input"]
        label = item["target"]
        logits = self.model(img)
        loss = self.loss_func(logits, label)
        acc = accuracy(logits, label)
        return loss, acc

    def inference(self, item: Dict[str, Any]):
        """
        Inference step for one batch.
        """
        img = item["input"]
        logits = self.model(img)
        return logits

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Training step for one batch.
        """
        loss, acc = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, sync_dist=True)
        self.log("step", self.global_step, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Validation step for one batch.
        """
        loss, acc = self.forward(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("step", self.global_step, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """
        Set up optimizer and scheduler.
        """
        optimizer = create_optimizer_v2(self, **self.cfg.optim)
        scheduler = create_scheduler(optimizer, **self.cfg.scheduler)
        return [optimizer], [scheduler]
