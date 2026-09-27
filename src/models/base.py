from abc import ABC, abstractmethod
import torch

class ModelWrapper(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self, loss_fn, logits, *args, **kwargs):
        pass

    @abstractmethod
    def inference(self, *args, **kwargs):
        pass