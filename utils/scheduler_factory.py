import inspect
from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    LambdaLR,
    MultiplicativeLR,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    PolynomialLR,
    CosineAnnealingLR,
    ChainedScheduler,
    SequentialLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
)


@dataclass(frozen=True)
class SchedulerInfo:
    name: str
    optimizer: Optimizer
    scheduler_class: Callable[..., LRScheduler]
    args: Optional[Dict[str, Any]] = None


class SchedulerRegistry:
    def __init__(self):
        self._schedulers: Dict[str, SchedulerInfo] = {}

    def register(self, info: SchedulerInfo) -> None:
        name = info.name.lower()
        self._schedulers[name] = info

    def list_schedulers(self) -> List[str]:
        names = list(sorted(self._schedulers.keys()))
        return names

    def get_scheduler_info(self, name: str) -> SchedulerInfo:
        key = name.lower()
        if key not in self._schedulers:
            raise ValueError(f"Scheduler '{name}' not found in registry")
        return self._schedulers[key]

    def _filter_kwargs(
        self, scheduler_class: Callable[..., LRScheduler], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        sig = inspect.signature(scheduler_class.__init__)
        valid_args = sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        return filtered_kwargs

    def create(self, optimizer: Optimizer, **kwargs) -> LRScheduler:
        name = kwargs.pop("name")
        info = self.get_scheduler_info(name)
        filtered_args = self._filter_kwargs(info.scheduler_class, kwargs)
        return info.scheduler_class(optimizer, **filtered_args)


def _register_schedulers(registry: SchedulerRegistry, optimizer: Optimizer):
    pytorch_schedulers = [
        SchedulerInfo(name="lambda", scheduler_class=LambdaLR, optimizer=optimizer),
        SchedulerInfo(
            name="multiplicative", scheduler_class=MultiplicativeLR, optimizer=optimizer
        ),
        SchedulerInfo(name="step", scheduler_class=StepLR, optimizer=optimizer),
        SchedulerInfo(
            name="multi_step", scheduler_class=MultiStepLR, optimizer=optimizer
        ),
        SchedulerInfo(name="constant", scheduler_class=ConstantLR, optimizer=optimizer),
        SchedulerInfo(name="linear", scheduler_class=LinearLR, optimizer=optimizer),
        SchedulerInfo(
            name="exponential", scheduler_class=ExponentialLR, optimizer=optimizer
        ),
        SchedulerInfo(
            name="polynomial", scheduler_class=PolynomialLR, optimizer=optimizer
        ),
        SchedulerInfo(
            name="cosine_annealing",
            scheduler_class=CosineAnnealingLR,
            optimizer=optimizer,
        ),
        SchedulerInfo(
            name="chained", scheduler_class=ChainedScheduler, optimizer=optimizer
        ),
        SchedulerInfo(
            name="sequential", scheduler_class=SequentialLR, optimizer=optimizer
        ),
        SchedulerInfo(
            name="reduce_on_plateau",
            scheduler_class=ReduceLROnPlateau,
            optimizer=optimizer,
        ),
        SchedulerInfo(name="cyclic", scheduler_class=CyclicLR, optimizer=optimizer),
        SchedulerInfo(
            name="one_cycle", scheduler_class=OneCycleLR, optimizer=optimizer
        ),
        SchedulerInfo(
            name="cosine_annealing_warm_restarts",
            scheduler_class=CosineAnnealingWarmRestarts,
            optimizer=optimizer,
        ),
    ]
    for scheduler in pytorch_schedulers:
        registry.register(scheduler)


def _register_custom_schedulers(registry: SchedulerRegistry):
    pass


def _register_default_schedulers(
    registry: SchedulerRegistry, optimizer: Optimizer
) -> None:
    """Register all default schedulers to the global registry."""
    _register_schedulers(registry, optimizer)
    _register_custom_schedulers(registry)


def create_scheduler(
    optimizer: Optimizer,
    **kwargs,
) -> LRScheduler:
    default_registry = SchedulerRegistry()
    _register_default_schedulers(default_registry, optimizer)

    return default_registry.create(optimizer, **kwargs)


def list_schedulers(registry: SchedulerRegistry) -> List[str]:
    return registry.list_schedulers()
