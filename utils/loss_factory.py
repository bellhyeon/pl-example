from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass
from typing import List
import torch.nn as nn
import inspect

from timm.loss.asymmetric_loss import (
    AsymmetricLossMultiLabel,
    AsymmetricLossSingleLabel,
)
from timm.loss.binary_cross_entropy import BinaryCrossEntropy
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.loss.jsd import JsdCrossEntropy


@dataclass(frozen=True)
class LossInfo:
    name: str
    loss_class: Callable[..., nn.Module]
    args: Optional[Dict[str, Any]] = None


class LossRegistry:
    def __init__(self):
        self._losses: Dict[str, LossInfo] = {}

    def register(self, info: LossInfo) -> None:
        name = info.name.lower()
        self._losses[name] = info

    def list_losses(self) -> List[str]:
        names = list(sorted(self._losses.keys()))
        return names

    def get_loss_info(self, name: str) -> LossInfo:
        key = name.lower()
        if key not in self._losses:
            raise ValueError(f"Loss '{name}' not found in registry")
        return self._losses[key]

    def _filter_kwargs(
        self, loss_class: Callable[..., nn.Module], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        sig = inspect.signature(loss_class.__init__)
        valid_args = sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        return filtered_kwargs

    def create(self, **kwargs) -> nn.Module:
        name = kwargs.pop("name")
        info = self.get_loss_info(name)
        filtered_args = self._filter_kwargs(info.loss_class, kwargs)
        return info.loss_class(**filtered_args)


def _register_timm_losses(registry: LossRegistry):
    timm_losses = [
        LossInfo(
            name="asymmetric_loss_multi_label", loss_class=AsymmetricLossMultiLabel
        ),
        LossInfo(
            name="asymmetric_loss_single_label", loss_class=AsymmetricLossSingleLabel
        ),
        LossInfo(name="binary_cross_entropy", loss_class=BinaryCrossEntropy),
        LossInfo(
            name="label_smoothing_cross_entropy", loss_class=LabelSmoothingCrossEntropy
        ),
        LossInfo(name="soft_target_cross_entropy", loss_class=SoftTargetCrossEntropy),
        LossInfo(name="jsd_cross_entropy", loss_class=JsdCrossEntropy),
    ]

    for loss in timm_losses:
        registry.register(loss)


def _register_pytorch_losses(registry: LossRegistry):
    pytorch_losses = [
        LossInfo("l1", loss_class=nn.L1Loss),
        LossInfo("mse", loss_class=nn.MSELoss),
        LossInfo("cross_entropy", loss_class=nn.CrossEntropyLoss),
        LossInfo("ctc", loss_class=nn.CTCLoss),
        LossInfo("nll", loss_class=nn.NLLLoss),
        LossInfo("poisson_nll", loss_class=nn.PoissonNLLLoss),
        LossInfo("gaussian_nll", loss_class=nn.GaussianNLLLoss),
        LossInfo("kl_div", loss_class=nn.KLDivLoss),
        LossInfo("bce", loss_class=nn.BCELoss),
        LossInfo("bce_with_logits", loss_class=nn.BCEWithLogitsLoss),
        LossInfo("margin_ranking", loss_class=nn.MarginRankingLoss),
        LossInfo("hinge_embedding", loss_class=nn.HingeEmbeddingLoss),
        LossInfo("multi_margin", loss_class=nn.MultiMarginLoss),
        LossInfo("huber", loss_class=nn.HuberLoss),
        LossInfo("smooth_l1", loss_class=nn.SmoothL1Loss),
        LossInfo("soft_margin", loss_class=nn.SoftMarginLoss),
        LossInfo("multi_label_soft_margin", loss_class=nn.MultiLabelSoftMarginLoss),
        LossInfo("cosine_embedding", loss_class=nn.CosineEmbeddingLoss),
        LossInfo("triplet_margin", loss_class=nn.TripletMarginLoss),
        LossInfo(
            "triplet_margin_with_distance", loss_class=nn.TripletMarginWithDistanceLoss
        ),
    ]
    for loss in pytorch_losses:
        registry.register(loss)


def _register_custom_losses(registry: LossRegistry):
    pass


def _register_default_loss_functions() -> None:
    """Register all default loss functions to the global registry."""
    _register_timm_losses(default_registry)
    _register_pytorch_losses(default_registry)
    _register_custom_losses(default_registry)


default_registry = LossRegistry()
_register_default_loss_functions()


def create_loss(**kwargs) -> nn.Module:
    return default_registry.create(**kwargs)


def list_losses() -> List[str]:
    return default_registry.list_losses()
