from abc import ABCMeta, abstractmethod
from torch import nn
from lightning import LightningDataModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Runner(metaclass=ABCMeta):
    def __init__(
        self,
        model: nn.Module,
        data_module: LightningDataModule,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: str,
    ):
        self.model = model
        self.data_module = data_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    @abstractmethod
    def forward(self, item):
        pass
