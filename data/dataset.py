import torch
import cv2
from torch.utils.data import Dataset
from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Optional
from dataclasses import dataclass
from utils.constant import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class ImageClassificationDataset(Dataset):
    img_paths: List
    labels: Optional[List]
    use_augmentation: bool
    img_size: int

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict:
        img = cv2.imread(self.img_paths[idx])
        if img is None:
            raise ValueError(f"Failed to load image: {self.img_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = self._get_transforms()
        img = transform(image=img)["image"]

        if self.labels is not None:
            label = self.labels[idx]
            return {"input": img, "target": torch.tensor(label, dtype=torch.long)}
        else:
            return {"input": img}

    def _get_transforms(self):
        if self.use_augmentation:
            train_transforms = [
                Resize(self.img_size, self.img_size),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
            return Compose(train_transforms)
        else:
            test_transforms = [
                Resize(self.img_size, self.img_size),
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
            return Compose(test_transforms)
