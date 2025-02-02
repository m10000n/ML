from kagglehub.models_helpers import delete_model
from torch import Tensor, int64
from torch.nn import Dropout, Module, ModuleList, Linear, Softmax
from torchinfo import summary

from model.Vaswani_Transformer.Embed import Embed
from model.Vaswani_Transformer.EncoderBlock import EncoderBlock
from model.Vaswani_Transformer.PosEncode import PosEncode
from model.Vaswani_Transformer.DecoderBlock import DecoderBlock


# don't forget about the max sequence length
class Transformer(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        seq_limit: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dff: int = 2048,
        p_drop: float = 0.1,
    ):
        super().__init__()

        self._embed = Embed(vocab_size=vocab_size, d_model=d_model)
        self._pos_encode = PosEncode(d_model=d_model, seq_limit=seq_limit)
        self._drop_out = Dropout(p=p_drop)
        self._encoder = ModuleList(
            [
                EncoderBlock(d_model=d_model, n_heads=n_heads, dff=dff, p_drop=p_drop)
                for _ in range(n_layers)
            ])
        self._decoder = ModuleList([
            DecoderBlock(d_model=d_model, n_heads=n_heads, dff=dff, p_drop=p_drop)
        ])
        self._linear = Linear(in_features=d_model, out_features=vocab_size)
        self._linear.weight = self._embed.weight
        self._softmax = Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        x = self._embed(x)
        x = self._pos_encode(x)
        x = self._drop_out(x)
        for layer in self._encoder:
            x = layer(x, mask=None)
        encoder_out = x
        #
        for layer in self._decoder:
            x = layer(x, encoder_out=encoder_out, src_mask=None, tgt_mask=None)
        x = self._linear(x)
        return self._softmax(x)


if __name__ == "__main__":
    summary(
        Transformer(100),
        input_size=(10, 4),
        dtypes=[int64],
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "trainable",
        ),
    )
