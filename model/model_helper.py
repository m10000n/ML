import torch
import torch.nn as nn
import torch.nn.functional as F


# utility layers
class Add(nn.Module):
    """
    Adds two tensors element-wise.
    """

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Add class.

        Args:
            x (torch.Tensor): First tensor.
            y (torch.Tensor): Second tensor.

        Returns:
            The sum of the two tensors.
        """
        return x + y


class Concat(nn.Module):
    """
    Concatenates a list of tensors along a specified dimension.

    Attributes:
        dim (int): Dimension along which to concatenate the tensors.
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize the Concat class.

        Args:
            dim (int): Dimension along which to concatenate the tensors.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Concat class.

        Args:
            x (list[torch.Tensor]): List of tensors to concatenate.

        Returns:
            The concatenated tensor.
        """
        return torch.cat(x, dim=self.dim)


class Unsqueeze(nn.Module):
    """
    Unsqueezes a tensor along a specified dimension.

    Attributes:
        dim (int): Dimension along which to unsqueeze the tensor.
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize the Unsqueeze class.

        Args:
            dim (int): Dimension along which to unsqueeze the tensor.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Unsqueeze class.

        Args:
            x (torch.Tensor): Tensor to unsqueeze.

        Returns:
            The unsqueezed tensor.
        """
        return x.unsqueeze(self.dim)


class Pad(nn.Module):
    """
    Zero-pads a tensor along the specified dimensions.

    Wrapper around `pad` for composition within `nn.Module` graphs. See `model.model_helper.pad` for details.
    """

    def __init__(self, padding: list[int | tuple[int, int] | None]) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pad(x, self.padding)


class PadToFactors(nn.Module):
    """
    Zero-pads a tensor so specified dimensions are multiples of given factors.

    Wrapper around `pad_to_factors` for composition within `nn.Module` graphs. See `model.model_helper.pad_to_factors`
    for details.
    """

    def __init__(self, patch_factors: list[int | None]) -> None:
        super().__init__()
        self.patch_factors = patch_factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pad_to_factors(x, self.patch_factors)


class CropToFactors(nn.Module):
    """
    Crops a tensor so specified dimensions are multiples of given factors.

    Wrapper around `crop_to_factors` for composition within `nn.Module` graphs. See`model.model_helper.crop_to_factors`
    for details.
    """

    def __init__(self, factors: list[int | None]) -> None:
        super().__init__()
        self.factors = factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return crop_to_factors(x, self.factors)


class CenterCrop(nn.Module):
    """
    Crops a tensor to a specified target size.

    Wrapper around `center_crop` for composition within `nn.Module` graphs. See `model.model_helper.center_crop` for
    details.
    """

    def __init__(self, target_size: list[int | None]) -> None:
        super().__init__()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return center_crop(x, self.target_size)


class Subsample(nn.Module):
    """
    Subsamples a tensor by the specified per-dimension strides.

    Wrapper around `subsample` for composition within `nn.Module` graphs. See `model.model_helper.subsample` for
    details.
    """

    def __init__(self, strides: list[int | None]) -> None:
        super().__init__()
        self.strides = strides

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return subsample(x, self.strides)


class Interpolate3d(nn.Module):
    """
    Resizes a 5D tensor to a fixed 3D spatial size using trilinear interpolation.

    Wrapper around `torch.nn.functional.interpolate` (mode="trilinear") for composition within `nn.Module` graphs.
    """

    def __init__(self, size: tuple[int, int, int]) -> None:
        super().__init__()

        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=self.size, mode="trilinear", align_corners=False)


# utility functions
def get_output_shape(
    input_shape: tuple[int, ...],
    module: nn.Module | list[nn.Module],
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.get_default_dtype(),
) -> list[int]:
    """
    Returns the output shape (excluding the batch dimension) after applying one ore more modules.

    Runs a no-grad forward pass with a dummy input of shape (`batch_size`, *`input_shape`). If a list of modules is
    provided, the modules are applied sequentially.

    Args:
        input_shape (tuple[int, ...]): Shape of the input tensor. All entries must be positive.
        module (nn.Module | list[nn.Module]): Module or list of modules to apply.
        batch_size (int): Batch size of the input tensor. Defaults to 1. Must be positive.
        device (torch.device): Device on which to create the dummy input tensor. Defaults to CPU.
        dtype (torch.dtype): Data type of the dummy input tensor. Defaults to PyTorch's default dtype.

    Returns:
        A list of ints representing the output shape (excluding the batch dimension) after applying the module(s).
    """
    assert all(dim > 0 for dim in input_shape), "All entries of `input_shape` must be positive."
    assert batch_size > 0, "`batch_size` must be positive."

    x = torch.randn(batch_size, *input_shape, device=device, dtype=dtype)
    module_ = module if isinstance(module, list) else [module]

    with torch.no_grad():
        for m in module_:
            x = m.forward(x)

    return list(x.shape[1:])


def pad(x: torch.Tensor, paddings: list[int | tuple[int, int] | None]) -> torch.Tensor:
    """
    Zero-pads a tensor along the specified dimensions.

    The length of `paddings` must be equal to the number of dimensions of `x`. Each entry in `paddings` corresponds to
    the same-index dimension of `x`. Depending on the type of an entry of `paddings`, padding is applied as follows:
        - int p: p zeros will be added to start and end.
        - tuple (p1, p2): p1 zeros will be added to the start and p2 zeros will be added to the end.
        - None: no padding will be applied.

    Args:
        x (torch.Tensor): Tensor to pad.
        paddings (list[int | tuple[int, int] | None]): Padding to apply to each dimension of `x`. All ints must be
            non-negative.

    Returns:
        The padded tensor.

    Notes:
        Internally torch.nn.functional.pad is used.
    """
    assert x.ndim == len(paddings), f"The number of dimensions of `x` and the length of `paddings` must be equal."

    paddings_: list[int] = []

    for padding in paddings:
        if isinstance(padding, tuple):
            dim_padding = (padding[1], padding[0])
        elif isinstance(padding, int):
            dim_padding = (padding, padding)
        elif padding is None:
            dim_padding = (0, 0)
        else:
            raise ValueError(f"`paddings` must be a list of ints, tuples, or None.")

        paddings_.extend(dim_padding)

    assert all(padding_ >= 0 for padding_ in paddings_), f"All ints in `paddings` must be non-negative."

    return F.pad(x, paddings_[::-1])


def pad_to_factors(x: torch.Tensor, factors: list[int | None]) -> torch.Tensor:
    """
    Zero-pads a tensor so that each specified dimension is a multiple of the corresponding factor.

    The length of `factors` must be equal to the number of dimensions of `x`. Each entry in `factors` corresponds to
    the same-index dimension of `x`. If an entry of `factors` is None, no padding will be applied to the corresponding
    dimension. Padding will be distributed symmetrically to the start and end of the dimension.

    Args:
        x (torch.Tensor): Tensor to pad to factors.
        factors (list[int | None]): Factors to pad `x` to. All ints must be positive.

    Returns:
        The padded tensor.

    Notes:
        Internally torch.nn.functional.pad is used.d
    """
    assert x.ndim == len(factors), f"The number of dimensions of `x` and the length of `factors` must be equal."
    assert all(factor > 0 for factor in factors if factor is not None), f"All ints in `factors` must be positive."

    padding = []
    for dim_size, factor in zip(x.shape, factors):
        if not factor:
            padding.extend([0, 0])
        else:
            pad_x = -dim_size % factor
            padding.extend([pad_x - pad_x // 2, pad_x // 2])

    x = F.pad(x, padding[::-1])

    assert all(
        not factor or dim_size % factor == 0 for dim_size, factor in zip(x.shape, factors)
    ), f"The padding failed. Padded dimensions: {tuple(x.shape)}. Factors: {(*factors,)}."

    return x


def crop_to_factors(x: torch.Tensor, factors: list[int | None]) -> torch.Tensor:
    """
    Crops a tensor so that each specified dimension is a multiple of the corresponding factor.

    The length of `factors` must be equal to the number of dimensions of `x`. Each entry in `factors` corresponds to
    the same-index dimension of `x` and must not exceed the corresponding dimension. If an entry of `factors` is None,
    no cropping will be applied. Cropping will be applied symmetrically to the start and end of the dimension.

    Args:
        x (torch.Tensor): Tensor to pad to factors.
        factors (list[int | None]): Factors to pad `x` to. All ints must be positive.

    Returns:
        The cropped tensor.
    """
    assert x.ndim == len(factors), f"The number of dimensions of `x` and the length of `factors` must be equal."
    assert all(
        dim_size >= factor for dim_size, factor in zip(x.shape, factors) if factor is not None
    ), f"All dimensions of `x` must be greater than or equal to the corresponding factor."

    slices = []
    for dim_size, factor in zip(x.shape, factors):
        if factor is None:
            slices.append(slice(None))
        else:
            drop_x = dim_size % factor
            drop_front = drop_x // 2
            drop_back = drop_x - drop_front
            slices.append(slice(drop_front, dim_size - drop_back if drop_back > 0 else None))

    x_cropped = x[tuple(slices)]

    assert all(
        not factor or dim_size % factor == 0 for dim_size, factor in zip(x_cropped.shape, factors)
    ), f"The cropping failed. Cropped dimensions: {tuple(x_cropped.shape)}. Factors: {(*factors,)}."

    return x_cropped


def center_crop(x: torch.Tensor, target_size: list[int | None]) -> torch.Tensor:  #
    """
    Crops a tensor to a specified size.

    The length of `target_size` must be equal to the number of dimensions of `x`. Each entry in `target_size`
    corresponds to the same-index dimension of `x` and must not exceed the corresponding dimension. If an entry of
    `target_size` is None, no cropping will be applied. Cropping will be applied symmetrically to the start and end of
    the dimension.

    Args:
        x (torch.Tensor): Tensor to crop to target size.
        target_size (list[int | None]): Target size to crop `x` to. All ints must be positive.

    Returns:
        The cropped tensor.
    """
    assert x.ndim == len(target_size), f"The number of dimensions of `x` and the length of `target_size` must be equal."
    assert all(
        target_size_ > 0 for target_size_ in target_size if target_size_ is not None
    ), "All target sizes must be positive."

    slices = []

    for dim_size, target_size_ in zip(x.shape, target_size):
        if target_size_ is None:
            slices.append(slice(None))
        else:
            assert target_size_ > 0, ""
            assert (
                target_size_ <= dim_size
            ), f"Target size `{target_size_}` for dimension `{dim_size}` cannot exceed dimension size `{dim_size}`."

            start = (dim_size - target_size_) // 2
            end = start + target_size_
            slices.append(slice(start, end))

    x_cropped = x[tuple(slices)]

    assert all(
        x_size == target_size_ for x_size, target_size_ in zip(x_cropped.shape, target_size) if target_size_ is not None
    ), f"The cropping failed. Cropped dimensions: {tuple(x_cropped.shape)}. Target size: {(*target_size,)}."

    return x_cropped


def subsample(x: torch.Tensor, strides: list[int | None]) -> torch.Tensor:
    """
    Subsamples a tensor by a specified stride.

    The length of `strides` must be equal to the number of dimensions of `x`. Each entry in `strides` corresponds to
    the same-index dimension of `x`. If an entry of `strides` is None, no subsampling will be applied and the
    corresponding dimenions will not be reduced.

    Args:
        x (torch.Tensor): Tensor to subsample.
        strides (list[int | None]): Strides to subsample `x` by. All ints must be positive.

    Returns:
        The subsampled tensor.
    """
    assert x.ndim == len(strides), f"The number of dimensions of `x` and the length of `strides` must be equal."
    assert all(stride > 0 for stride in strides if stride is not None), f"All ints in `strides` must be positive."

    slices = [slice(None) if (stride is None or stride == 1) else slice(None, None, stride) for stride in strides]

    return x[tuple(slices)]


def assert_no_nan(tensor: torch.Tensor, tensor_name: str | None = None) -> None:
    """
    Asserts that a tensor does not contain any NaN values.

    Args:
        tensor (torch.Tensor): Tensor to check for NaN values.
        tensor_name (str | None): Name of the tensor. Defaults to None.

    Returns:
        None.
    """
    assert not torch.isnan(tensor).any(), (
        f"The tensor `{tensor_name}` contains NaN." if tensor_name else "Tensor contains NaN."
    )
