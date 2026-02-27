from typing import Sequence

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table
from fvcore.nn.jit_handles import get_shape
from torch._C import Value

from helper.helper_ import round_to_str
from helper.print import print_color

##### config start #####
_MAX_DEPTH: int = 10
# fmt: off
_IGNORE_OPS = {
    # --- elementwise math (incl. in-place) ---
    "aten::add","aten::add_","aten::rsub","aten::sub","aten::sub_",
    "aten::mul","aten::mul_","aten::div","aten::div_","aten::neg","aten::abs",
    "aten::pow","aten::clamp","aten::clamp_min","aten::clamp_max",
    "aten::maximum","aten::minimum","aten::where","aten::masked_fill","aten::masked_fill_",
    "aten::logical_and","aten::logical_or","aten::logical_xor",
    "aten::bitwise_and","aten::bitwise_or","aten::bitwise_xor",
    "aten::round","aten::floor","aten::ceil","aten::trunc","aten::sign",
    "aten::exp","aten::log","aten::sqrt","aten::rsqrt","aten::reciprocal",
    "aten::addcmul","aten::addcdiv",

    # --- activations ---
    "aten::relu","aten::relu_","aten::leaky_relu","aten::leaky_relu_",
    "aten::hardtanh","aten::hardtanh_","aten::gelu","aten::silu","aten::sigmoid",
    "aten::tanh","aten::hardsigmoid","aten::hardswish","aten::elu","aten::selu",
    "aten::softplus","aten::softsign",

    # --- softmax / log-softmax ---
    "aten::softmax","aten::_softmax","aten::log_softmax","aten::_log_softmax",

    # --- normalization (elementwise) ---
    "aten::layer_norm","aten::native_layer_norm",
    "aten::group_norm","aten::instance_norm",
    "aten::batch_norm","aten::native_batch_norm",
    "aten::_native_batch_norm_legit","aten::_native_batch_norm_legit_no_training",
    "aten::_cudnn_batch_norm",

    # --- dropout / identity / copies ---
    "aten::dropout","aten::feature_dropout","aten::alpha_dropout","aten::feature_alpha_dropout","aten::dropout_",
    "aten::noop","aten::detach","aten::clone",

    # --- padding ---
    "aten::pad","aten::constant_pad_nd",
    "aten::reflection_pad1d","aten::reflection_pad2d",
    "aten::replication_pad1d","aten::replication_pad2d","aten::replication_pad3d",

    # --- reshape / view / indexing / layout ---
    "aten::reshape","aten::view","aten::flatten","aten::squeeze","aten::unsqueeze",
    "aten::permute","aten::transpose","aten::contiguous","aten::copy_",
    "aten::expand","aten::expand_as","aten::repeat","aten::tile",
    "aten::narrow","aten::slice","aten::select",
    "aten::gather","aten::index","aten::index_put","aten::index_put_","aten::index_select","aten::take",
    "aten::cat","aten::stack","aten::split","aten::split_with_sizes","aten::chunk","aten::unbind",
    "aten::unflatten",

    # --- type / device casts ---
    "aten::to","aten::type_as",

    # --- pooling ---
    "aten::avg_pool1d","aten::avg_pool2d","aten::avg_pool3d",
    "aten::adaptive_avg_pool1d","aten::adaptive_avg_pool2d","aten::adaptive_avg_pool3d",
    "aten::max_pool1d","aten::max_pool2d","aten::max_pool3d",
    "aten::adaptive_max_pool1d","aten::adaptive_max_pool2d","aten::adaptive_max_pool3d",

    # --- upsampling / interpolate ---
    "aten::upsample_nearest1d","aten::upsample_nearest2d","aten::upsample_nearest3d",
    "aten::upsample_linear1d","aten::upsample_bilinear2d","aten::upsample_bicubic2d",
    "aten::upsample_trilinear3d","aten::_upsample_bilinear2d_aa",
    "aten::interpolate",

    # --- common reductions ---
    "aten::mean","aten::sum","aten::amax","aten::amin",
    "aten::argmax","aten::argmin","aten::var","aten::std","aten::norm",
}
# fmt: on
##### config end #####

SDPA_OPS = [
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_attention_math",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_flash_attention",
]


def analyze_flops(model: nn.Module, input_shape: tuple[int, ...], batch_size: int, max_depth: int = _MAX_DEPTH) -> None:
    x = torch.randn(batch_size, *input_shape, device="cpu")

    model.eval()
    with torch.no_grad():
        model(x)

    fca = FlopCountAnalysis(model=model, inputs=(x,))

    for op in SDPA_OPS:
        fca.set_op_handle(**{op: _sdpa_flop_jit})

    for op in _IGNORE_OPS:
        fca.set_op_handle(**{op: _zero_flop_jit})

    print(flop_count_table(flops=fca, max_depth=max_depth, show_param_shapes=False))

    unsupported = fca.unsupported_ops()
    if unsupported:
        print_color(text=f"Unsupported ops found (counted as 0):", color_="yellow")
        print(f"    {", ".join([f"{op} - {count}" for op, count in unsupported.items()])}")

    total_gflop = (2 * int(fca.total())) / 1e9
    print(f"\n{round_to_str(x=total_gflop, digits=3)} GFLOP (per forward pass, assuming 1 MAC = 2 FLOPs)")


def _sdpa_flop_jit(inputs: Sequence[Value], outputs: Sequence[Value]) -> int:
    q_shape = get_shape(inputs[0])
    k_shape = get_shape(inputs[1])
    v_shape = get_shape(inputs[2])

    if not q_shape or len(q_shape) < 3:
        raise RuntimeError(f"[SDPA] unknown/short Q shape: {q_shape}")
    if not k_shape or len(k_shape) < 3:
        raise RuntimeError(f"[SDPA] unknown/short K shape: {k_shape}")
    if not v_shape or len(v_shape) < 3:
        raise RuntimeError(f"[SDPA] unknown/short V shape: {v_shape}")

    batch_like = _prod(q_shape[:-3]) if len(q_shape) > 3 else 1
    H = int(q_shape[-3])
    S_q = int(q_shape[-2])
    Hd = int(q_shape[-1])
    S_k = int(k_shape[-2])

    if int(k_shape[-1]) != Hd or int(v_shape[-1]) != Hd:
        raise RuntimeError(f"[SDPA] head-dim mismatch: Q{q_shape}, K{k_shape}, V{v_shape}")
    if int(v_shape[-2]) != S_k:
        raise RuntimeError(f"[SDPA] K/V seq mismatch: K{k_shape}, V{v_shape}")

    return 2 * batch_like * H * S_q * S_k * Hd


def _zero_flop_jit(inputs: Sequence[Value], outputs: Sequence[Value]) -> int:
    return 0


def _prod(seq: Sequence[int]) -> int:
    p = 1
    for s in seq:
        p *= int(s)
    return p
