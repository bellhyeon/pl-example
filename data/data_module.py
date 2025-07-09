from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import List

from data.dataset import ImageClassificationDataset


class ImageClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        train_paths: List[str],
        train_labels: List[int],
        val_paths: List[str],
        val_labels: List[int],
        **kwargs
    ):
        """
        Initialize the data module.

        Args:
            train_paths (List[str]): List of paths to the training images.
            train_labels (List[int]): List of labels for the training images.
            val_paths (List[str]): List of paths to the validation images.
            val_labels (List[int]): List of labels for the validation images.
        """
        super().__init__()
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.val_paths = val_paths
        self.val_labels = val_labels
        self.train_img_size = kwargs.get("train_img_size", 224)
        self.val_img_size = kwargs.get("val_img_size", 224)
        self.batch_size = kwargs.get("batch_size", 32)
        self.num_workers = kwargs.get("num_workers", 15)

    def setup(self, stage: str | None = None):
        self.train_dataset = ImageClassificationDataset(
            img_paths=self.train_paths,
            labels=self.train_labels,
            use_augmentation=True,
            img_size=self.train_img_size,
        )
        self.val_dataset = ImageClassificationDataset(
            img_paths=self.val_paths,
            labels=self.val_labels,
            use_augmentation=False,
            img_size=self.val_img_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

