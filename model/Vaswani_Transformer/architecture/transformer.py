from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type

from torch import Tensor
from torch.nn import Dropout, Linear, ModuleList

from helper.class_ import Model_, ModelConfig_
from helper.validator import validator
from model.Vaswani_Transformer.architecture.decoder_block import DecoderBlock
from model.Vaswani_Transformer.architecture.embed import Embed
from model.Vaswani_Transformer.architecture.encoder_block import EncoderBlock
from model.Vaswani_Transformer.architecture.pos_encode import PosEncode


class Transformer(Model_):
    """
    Deep learning model implementing the encoder-decoder Transformer for sequence-to-sequence tasks.

    This model embeds both source and target token sequences into learnable representations, adds sinusoidal positional
    encodings, and applies dropout. The source embeddings are processed through a stack of Transformer encoder blocks,
    each consisting of multi-head self-attention and a position-wise feed-forward network. The decoder similarly embeds
    the target sequence and, at each position, performs masked self-attention, encoder-decoder attention over the
    encoder outputs, and a feed-forward network. The final decoder outputs are projected via a linear layer (tied to the
    input embeddings). This implementation follows the original Transformer design from “Attention Is All You Need”
    (Vaswani et al., 2017).

    Attributes:
        config (TransformerConfig): Configuration of the model.
    """

    config: TransformerConfig

    def __init__(
        self,
        config: TransformerConfig,
    ):
        """
        Initializes the Transformer class.

        Args:
            config (TransformerConfig): Configuration of the model.
        """

        super().__init__(config)

        # embedding
        self._embed = Embed(vocab_size=config.vocab_size, d_model=config.d_model)
        self._pos_encode = PosEncode(d_model=config.d_model, seq_limit=config.seq_limit)
        self._drop_out = Dropout(p=config.p_drop)

        # encoder
        self._encoder = ModuleList(
            [
                EncoderBlock(d_model=config.d_model, n_heads=config.n_heads, dff=config.dff, p_drop=config.p_drop)
                for _ in range(config.n_layers)
            ]
        )

        # decoder
        self._decoder = ModuleList(
            [
                DecoderBlock(d_model=config.d_model, n_heads=config.n_heads, dff=config.dff, p_drop=config.p_drop)
                for _ in range(config.n_layers)
            ]
        )
        self._linear = Linear(in_features=config.d_model, out_features=config.vocab_size)
        self._linear.weight = self._embed.weight

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor | None = None, tgt_mask: Tensor | None = None
    ) -> Tensor:
        """
        Forward pass of the Transformer class.

        Args:
            src (Tensor): Source token sequence of shape (batch_size, src_seq_len).
            tgt (Tensor): Target token sequence of shape (batch_size, tgt_seq_len).
            src_mask (Tensor | None): Source mask of shape (batch_size, 1, 1, src_seq_len).
            tgt_mask (Tensor | None): Target mask of shape (batch_size, 1, tgt_seq_len, tgt_seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, tgt_seq_len, vocab_size).

        Precondition:
            - src_seq_len must be <= `seq_limit`
            - tgt_seq_len must be <= `seq_limit`
        """
        # encoder
        x = self._embed(src)
        x = self._pos_encode(x)
        x = self._drop_out(x)
        for layer in self._encoder:
            x = layer(x, mask=src_mask)
        encoder_out = x

        # decoder
        y = self._embed(tgt)
        y = self._pos_encode(y)
        y = self._drop_out(y)
        for layer in self._decoder:
            y = layer(y, encoder_out=encoder_out, src_mask=src_mask, tgt_mask=tgt_mask)
        y = self._linear(y)
        return y


@dataclass(init=False)
class TransformerConfig(ModelConfig_):
    """
    Configuration class for the Transformer model.

    Attributes:
        description: (str): Description of the model.
        input_shape: (tuple[int, int]): Shape of the input tensor.
        vocab_size: (int): Size of the vocabulary.
        d_model: (int): Dimensionality of the model.
        seq_limit: (int): Maximum sequence length.
        n_heads: (int): Number of attention heads.
        n_layers: (int): Number of encoder and decoder layers.
        dff: (int): Dimensionality of the feed-forward network.
        p_drop: (float): Dropout probability.
        class_: (Type[Transformer]): Class of the model.
    """

    class_: ClassVar[Type[Transformer]] = Transformer

    input_shape: tuple[int, int]
    vocab_size: int
    d_model: int
    seq_limit: int
    n_heads: int
    n_layers: int
    dff: int
    p_drop: float

    @validator.constraints("vocab_size", "x > 0")
    @validator.constraints("d_model", "x > 0")
    @validator.constraints("seq_limit", "x > 0")
    @validator.constraints("n_heads", "x > 0")
    @validator.constraints("n_layers", "x > 0")
    @validator.constraints("dff", "x > 0")
    @validator.constraints("p_drop", "x >= 0 and x <= 1")
    def __init__(
        self,
        description: str,
        input_shape: tuple[int, int] | list[int],
        vocab_size: int,
        d_model: int = 512,
        seq_limit: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dff: int = 2048,
        p_drop: float = 0.1,
    ):
        """
        Initializes the TransformerConfig class.

        Args:
            description: (str): Description of the model.
            input_shape (tuple[int, int] | list[int]):
                Shape of the input tensor.
                Must be a tuple or list of 2 positive integers.
            vocab_size (int): Size of the vocabulary. Must be a positive integer.
            d_model (int): Dimensionality of the model. Must be a positive integer. Default: 512.
            seq_limit (int): Maximum sequence length. Must be a positive integer. Default: 512.
            n_heads (int): Number of attention heads. Must be a positive integer. Default: 8.
            n_layers (int): Number of encoder and decoder layers. Must be a positive integer. Default: 6.
            dff (int): Dimensionality of the feed-forward network. Must be a positive integer. Default: 2048.
            p_drop (float): Dropout probability. Must be a float in range [0, 1]. Default: 0.1.
        """
        super().__init__(description=description, input_shape=input_shape)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_limit = seq_limit
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dff = dff
        self.p_drop = p_drop
