from math import ceil, log

from torch import Tensor, arange, cos, empty, exp, sin
from torch.nn import Module


# done
class PosEncode(Module):
    """
    Implements sinusoidal positional encoding.

    This module injects position information into input embeddings using fixed sine and cosine functions, ensuring that each position in the sequence is
    uniquely represented. Even indices of the embedding use sine functions, while odd indices use cosine functions. This enables Transformers to capture both
    absolute and relative positions without learnable parameters. The encoding is added directly to the input embeddings.

    Attributes:
        d_model (int): Dimensionality of the token embeddings.
        seq_limit (int): Maximum sequence length supported.
        stable (bool): Whether to use numerically stable exponentiation.

    Note:
        The computation of the sinusoidal functions is determined by `stable`:
        - `True`: Uses `exp(dimensions * (-log(10000) / d_model))`, which avoids large exponentiation errors and ensures numerical stability.
        - `False`: Uses `1 / pow(10000, dimensions / d_model)`, which follows the original formulation from "Attention Is All You Need" (Vaswani et al.,
        2017) but can introduce floating-point precision issues.
    """

    def __init__(self, d_model: int = 512, seq_limit: int = 512, stable: bool = True):
        """
        Initializes the PosEncode class.

        Args:
            d_model (int, optional): Dimensionality of the token embeddings. Default: 512.
            seq_limit(int, optional): Maximum sequence length supported. Default: 512.
            stable (bool, optional): Whether to use numerically stable exponentiation: Default: True.
        """
        super().__init__()
        self.seq_limit = seq_limit
        self.d_model = d_model
        self.stable = stable

        positions = arange(seq_limit).unsqueeze(1)
        dimensions = arange(0, int(ceil(d_model / 2)))
        if stable:
            angle = exp(dimensions * (-log(10000) / d_model))
        else:
            angle = 1 / pow(10000, dimensions / d_model)

        pos_encoding = empty((seq_limit, d_model))
        pos_encoding[:, 0::2] = sin(positions * angle)
        pos_encoding[:, 1::2] = cos(positions * angle)

        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PosEncode class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, `d_model`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, `d_model`), representing the input embeddings with added positional encodings.

        Precondition:
            - seq_len <= `seq_limit`
        """
        seq_len = x.shape[1]

        return x + self.pos_encoding[:, :seq_len, :].to(x.device)
