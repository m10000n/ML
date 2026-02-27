from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Type, cast

import einops.layers.torch as einops
import torch
import torch.nn as nn

from helper.class_ import Module_, ModuleConfig_
from helper.validator import validator
from model.model_helper import CropToFactors, PadToFactors


# abstract class
class PatchEmbed4d_(Module_):
    """
    Abstract base class for learned patch embedding modules that map fMRI data to a sequence of embeddings.
    Subclasses must implement the embedding logic.

    Attributes:
        config (PatchEmbed4dConfig_): Configuration of the module.
    """

    config: PatchEmbed4dConfig_

    @abstractmethod
    def __init__(self, config: PatchEmbed4dConfig_) -> None:
        if not config.is_initialized(mode="use"):
            raise ValueError(
                f"Failed to create an instance of `{self.__class__.__name__}`, because`config` is not initialized."
            )

        super().__init__(config)


# concrete classes
class PatchEmbedTime(PatchEmbed4d_):
    """
    Implements learned patch embedding for mapping fMRI data to a sequence of embeddings.

    This module divides the input fMRI into non-overlapping patches, each corresponding to a fixed spatial location in
    the brain and capturing the temporal signal across all volumes at that location. These patches are then linearly
    projected into a specified embedding space using a convolutional layer.
    Optionally, the input can be downsampled before embedding using either pooling or convolution.

    Note:
        Downsampling is determined by `reduction`.
        - "none": No downsampling is applied.
        - "pool": Average pooling is applied in both spatial and temporal dimensions. In essence a kernel of size
          (2, 2, 2, 2) with a stride of (2, 2, 2, 2) is used, reducing all dimensions by a factor of 2. This reduces
          the input size while retaining local spatial-temporal relationships.
        - "conv1": A 3D convolution with a kernel size of (2, 2, 2) and a stride of (2, 2, 2) is applied to each pair
          of consecutive volumes. The kernel weights are shared across all pairs. This reduces the input size by a
          factor of 2 in each dimension while retaining local spatial-temporal relationships. The output of the
          convolution is then passed through a BatchNorm3d layer and a ReLU activation function.
        - "conv2": A 3D convolution with a kernel size of (2, 2, 2) and a stride of (2, 2, 2) is applied. The number of
          output channels is set to the number of volumes // 2. This reduces the input size by a factor of 2 in all
          dimensions. The output of the convolution is then passed through a BatchNorm3d layer and a ReLU activation
          function.

    Attributes:
        config (PatchEmbedTimeConfig): Configuration of the module.
    """

    config: PatchEmbedTimeConfig

    def __init__(self, config: PatchEmbedTimeConfig):
        """
        Initializes the PatchEmbedTime4d class.

        Args:
            config (PatchEmbedTimeConfig): Configuration of the module.
        """
        super().__init__(config)

        volumes = self.config.input_shape[0]

        if self.config.reduction == "none":
            self._reduce = None
        else:
            if self.config.reduction == "pool":
                self._reduce = nn.Sequential(
                    CropToFactors([None, 2, 2, 2, 2]),
                    einops.Rearrange("b (k d) x y z -> b k x y (z d)", d=2),
                    nn.AvgPool3d(kernel_size=(2, 2, 4), stride=(2, 2, 4)),
                )
            elif self.config.reduction == "conv1":
                volumes_cropped = volumes if volumes % 2 == 0 else volumes - 1
                self._reduce = nn.Sequential(
                    CropToFactors([None, 2, 2, 2, 2]),
                    einops.Rearrange("b (v g) x y z -> (b v) g x y z", g=2),
                    nn.Conv3d(
                        in_channels=2,
                        out_channels=1,
                        kernel_size=2,
                        stride=2,
                        bias=False,
                    ),
                    einops.Rearrange("(b v) 1 x y z -> b v x y z", v=volumes_cropped // 2),
                    nn.BatchNorm3d(volumes_cropped // 2),
                    nn.ReLU(),
                )
            elif self.config.reduction == "conv2":
                self._reduce = nn.Sequential(
                    CropToFactors([None, None, 2, 2, 2]),
                    nn.Conv3d(in_channels=volumes, out_channels=volumes // 2, kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm3d(volumes // 2),
                    nn.ReLU(),
                )
            else:
                raise ValueError(f"Reduction `{self.config.reduction}` not supported.")

        self._pad = PadToFactors([None, None, *self.config.patch_size])

        self._embed = nn.Conv3d(
            in_channels=volumes if self.config.reduction == "none" else volumes // 2,
            out_channels=self.config.d_model,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self._embed.is_patch_embed = True  # type: ignore[assignment]
        self._rearrange = einops.Rearrange("b e x y z -> b (x y z) e")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbedTime class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *`input_shape`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches, `d_model`).

        """
        if self._reduce:
            x = self._reduce(x)

        x = self._pad(x)
        x = self._embed(x)
        x = self._rearrange(x)
        return x


class PatchEmbedVolume(PatchEmbed4d_):
    """
    Implements learned patch embedding for mapping fMRI data to a sequence of embeddings.

    This module treats each volume as a single patch, and projects it into a specified embedding space using a
    convolutional layer. The kernel weights are shared across all volumes.
    The input is spatially downsampled before embedding using either pooling or convolution.

    Note:
      Downsampling is determined by `reduction`:
      - "pool": Average pooling is applied to the spatial dimensions. A kernel of size
        (`reduction_factor`, `reduction_factor`, `reduction_factor`) with a stride of
        (`reduction_factor`, `reduction_factor`, `reduction_factor`) is used, reducing all spatial dimensions by
        `reduction_factor`.
      - "conv": A 3D convolution with a kernel size of (`reduction_factor`, `reduction_factor`, `reduction_factor`)
        and a stride of (`reduction_factor`, `reduction_factor`, `reduction_factor`) is applied to each volume. The
        kernel weights are shared across volumes. The output of the convolution is then passed through a BatchNorm3d
        layer and a ReLU activation function.

    Attributes:
        config (PatchEmbedVolumeConfig): Configuration of the module.
    """

    config: PatchEmbedVolumeConfig

    def __init__(self, config: PatchEmbedVolumeConfig) -> None:
        """
        Initializes the PatchEmbedVolume4d class.

        Args:
            config (PatchEmbedVolumeConfig): Configuration of the module.
        """
        super().__init__(config)

        volumes = self.config.input_shape[0]
        spatial_dims_reduced = cast(
            tuple[int, int, int], tuple(d // self.config.reduction_factor for d in self.config.input_shape[1:])
        )

        self._crop = CropToFactors(
            [None, None, self.config.reduction_factor, self.config.reduction_factor, self.config.reduction_factor]
        )

        if self.config.reduction == "pool":
            self._reduce = nn.Sequential(
                CropToFactors(
                    [
                        None,
                        None,
                        self.config.reduction_factor,
                        self.config.reduction_factor,
                        self.config.reduction_factor,
                    ]
                ),
                nn.AvgPool3d(kernel_size=self.config.reduction_factor, stride=self.config.reduction_factor),
            )
        elif self.config.reduction == "conv":
            self._reduce = nn.Sequential(
                CropToFactors(
                    [
                        None,
                        None,
                        self.config.reduction_factor,
                        self.config.reduction_factor,
                        self.config.reduction_factor,
                    ]
                ),
                einops.Rearrange("b v x y z -> (b v) 1 x y z"),
                nn.Conv3d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=self.config.reduction_factor,
                    stride=self.config.reduction_factor,
                    bias=False,
                ),
                einops.Rearrange("(b v) 1 x y z -> b v x y z", v=volumes),
                nn.BatchNorm3d(volumes),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Reduction `{self.config.reduction}` not supported.")

        self._rearrange1 = einops.Rearrange("b v x y z -> (b v) 1 x y z")
        self._embed = nn.Conv3d(
            in_channels=1,
            out_channels=self.config.d_model,
            kernel_size=spatial_dims_reduced,
            stride=spatial_dims_reduced,
        )
        self._embed.is_patch_embed = True  # type: ignore[assignment]
        self._rearrange2 = einops.Rearrange("(b v) k 1 1 1 -> b v k", v=volumes, k=self.config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbedVolume class.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *`input_shape`).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches, `d_model`).
        """
        x = self._reduce(x)
        x = self._rearrange1(x)
        x = self._embed(x)
        x = self._rearrange2(x)
        return x


# abstract data classes
@dataclass(init=False)
class PatchEmbed4dConfig_(ModuleConfig_):
    """
    Abstract base configuration class for PatchEmbed4d_ classes.

    Attributes:
        input_shape (tuple[int, int, int, int]): Shape of the input fMRI data as (volumes, x, y, z).
        d_model (int): Dimensionality of the patch embeddings.
        class_ (Type[PatchEmbed4d_]): Class of the module.
    """

    class_: ClassVar[Type[PatchEmbed4d_]]

    input_shape: tuple[int, int, int, int] = field(init=False)
    d_model: int = field(init=False)

    @validator.constraints("input_shape", "x > 0")
    @validator.constraints("d_model", "x > 0")
    def __init__(
        self,
        input_shape: tuple[int, int, int, int] | list[int] | None = None,
        d_model: int | None = None,
    ) -> None:
        if input_shape is not None and len(input_shape) != 4:
            raise ValueError(f"`input_shape` must be of length 4, got {len(input_shape)}.")

        if input_shape is not None:
            self.input_shape = cast(tuple[int, int, int, int], tuple(input_shape))

        if d_model is not None:
            self.d_model = d_model

    @abstractmethod
    def is_initialized(self, mode: Literal["as_dict", "use", "full"] = "full") -> bool:
        for attribute in ["input_shape", "d_model"]:
            if not self.attr_is_initialized(attribute):
                return False

        return True


# concrete data classes
@dataclass(init=False)
class PatchEmbedTimeConfig(PatchEmbed4dConfig_):
    """
    Configuration class for the PatchEmbedTime class.

    Attributes:
        input_shape (tuple[int, int, int, int]): Shape of the input fMRI data as (volumes, x, y, z).
        d_model (int): Dimensionality of the patch embeddings.
        patch_size (tuple[int, int, int]): Size of each patch.
        reduction (Literal["none", "pool", "conv1", "conv2"]): Reduction method to apply.
        class_ (Type[PatchEmbedTime]): Class of the module.

    Note:
        - All instance variables must be set to non-None values before calling `as_dict`.
    """

    class_: ClassVar[Type[PatchEmbedTime]] = PatchEmbedTime

    patch_size: tuple[int, int, int] = field(init=False)
    reduction: Literal["none", "pool", "conv1", "conv2"] = field(init=False)

    @validator.constraints("patch_size", "x > 0")
    def __init__(
        self,
        input_shape: tuple[int, int, int, int] | list[int] | None = None,
        d_model: int | None = None,
        patch_size: int | tuple[int, int, int] | list[int] | None = None,
        reduction: Literal["none", "pool", "conv1", "conv2"] | None = None,
    ):
        """
        Initializes the PatchEmbedTimeConfig class.

        Args:
            input_shape (tuple[int, int, int, int] | list[int] | None):
              Shape of the input fMRI data as (volumes, x, y, z).
              Must be a tuple or list of 4 positive integers, or None. Default: None
            d_model (int | None):
              Dimensionality of the patch embeddings. Must be a positive integer, or None. Default: None.
            patch_size (int | tuple[int, int, int] | list[int] | None):
              Size of each patch.
              Must be a positive integer, a tuple or list of 3 positive integers, or None. Default: None.
              If a single integer is provided, it will be broadcast to all three spatial dimensions: (x, y, z).
            reduction (Literal["none", "pool", "conv1", "conv2"] | None):
              Reduction to apply. Must be one of "none", "pool", "conv1", "conv2", or None. Default: None.
        """
        if isinstance(patch_size, (tuple, list)) and len(patch_size) != 3:
            raise ValueError(f"`patch_size` must be of length 3, got {len(patch_size)}.")

        super().__init__(input_shape=input_shape, d_model=d_model)

        if reduction is not None:
            self.reduction = reduction

        if patch_size is not None:
            patch_size_ = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
            self.patch_size = cast(tuple[int, int, int], patch_size_)

    def is_initialized(self, mode: Literal["as_dict", "use", "full"] = "full") -> bool:
        for attribute in ["patch_size", "reduction"]:
            if not self.attr_is_initialized(attribute):
                return False

        return super().is_initialized(mode=mode)


@dataclass(init=False)
class PatchEmbedVolumeConfig(PatchEmbed4dConfig_):
    """
    Configuration class for the PatchEmbedVolume class.

    Attributes:
        input_shape (tuple[int, int, int, int]): Shape of the input fMRI data as (volumes, x, y, z).
        d_model (int): Dimensionality of the patch embeddings.
        reduction (Literal["pool", "conv"]): Reduction method to apply.
        reduction_factor (int): Factor by which the spatial dimensions are reduced.
        class_ (Type[PatchEmbedVolume]): Class of the module.

    Note:
        - All instance variables must be set to non-None values before calling `as_dict`.
    """

    class_: ClassVar[Type[PatchEmbedVolume]] = PatchEmbedVolume

    reduction: Literal["pool", "conv"] = field(init=False)
    reduction_factor: int = field(init=False)

    @validator.constraints("reduction_factor", "x > 1")
    def __init__(
        self,
        input_shape: tuple[int, int, int, int] | list[int] | None = None,
        d_model: int | None = None,
        reduction: Literal["pool", "conv"] | None = None,
        reduction_factor: int | None = None,
    ) -> None:
        """
        Initializes the PatchEmbedVolumeConfig class.

        Args:
            input_shape (tuple[int, int, int, int] | list[int] | None):
              Shape of the input fMRI data as (volumes, x, y, z).
              Must be a tuple or list of 4 positive integers, or None. Default: None.
            d_model (int | None):
              Dimensionality of the patch embeddings. Must be a positive integer, or None. Default: None.
            reduction (Literal["pool", "conv"] | None):
              Reduction to apply. Must be one of "pool", "conv", or None. Default: None.
            reduction_factor (int | None):
              Factor by which the spatial dimensions are reduced. Must be an integer > 1, or None. Default: None.
        """
        super().__init__(input_shape=input_shape, d_model=d_model)

        if reduction is not None:
            self.reduction = reduction

        if reduction_factor is not None:
            self.reduction_factor = reduction_factor

    def is_initialized(self, mode: Literal["as_dict", "use", "full"] = "full") -> bool:
        for attribute in ["reduction", "reduction_factor"]:
            if not self.attr_is_initialized(attribute):
                return False

        return super().is_initialized(mode=mode)
