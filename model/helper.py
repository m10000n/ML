from torch import Tensor
from torch.nn import Module


class AddResidual(Module):
    @staticmethod
    def forward(x: Tensor, residual: Tensor) -> Tensor:
        return x + residual
