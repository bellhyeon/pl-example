import torch


def accuracy(prediction: torch.Tensor, label: torch.Tensor) -> float:
    pred_class = torch.argmax(prediction, dim=-1)
    correct = (pred_class == label).float().mean()
    return correct.item()
