from data.HCP_1200.hcp_1200_data import HCP1200Data
from model.BrT.architecture.brain_transformer import BrainTransformerConfig
from model.BrT.architecture.patch_embed import (
    PatchEmbedTimeConfig,
    PatchEmbedVolumeConfig,
)

_WANG_N_VOLUMES = 27
_N_CLASSES = 7
_P_DROP = 0.1


# brt_small
## prep
### t4
def brt_small_t4n_prep() -> BrainTransformerConfig:  # 19,921,159 params; 9576 patches; 1074.054 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: none, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="none",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t4p_prep() -> BrainTransformerConfig:  # 12,800,263 params; 1200 patches; 40.330 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: pool, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="pool",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t4c1_prep() -> BrainTransformerConfig:  # 12,800,305 params; 1200 patches; 40.358 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: conv over local volumes, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv1",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t4c2_prep() -> BrainTransformerConfig:  # 12,803,097 params; 1200 patches; 40.712 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: conv over global volumes, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv2",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


### t8
def brt_small_t8n_prep() -> BrainTransformerConfig:  # 22,870,279 params; 1320 patches; 72.162 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: none, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="none",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t8p_prep() -> BrainTransformerConfig:  # 16,466,695 params; 150 patches; 4.951 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: pool, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="pool",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t8c1_prep() -> BrainTransformerConfig:  # 16,466,737 params; 150 patches; 4.979 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: conv over local volumes, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv1",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t8c2_prep() -> BrainTransformerConfig:  # 16,469,529 params; 150 patches; 5.333 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: conv over global volumes, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv2",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


### v/2
def brt_small_vp2_prep() -> BrainTransformerConfig:  # 63,545,863 params; 27 patches; 3.426 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: pool(2), embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=2),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_vc2_prep() -> BrainTransformerConfig:  # 63,545,925 params; 27 patches; 3.456 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: conv(2), embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=2),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


### v/4
def brt_small_vp4_prep() -> BrainTransformerConfig:  # 17,619,463 params; 27 patches; 1.055 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: pool(4), embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=4),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_vc4_prep() -> BrainTransformerConfig:  # 17,619,581 params; 27 patches; 0.975 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: conv(4), embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=4),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


## unp
### t4
def brt_small_t4n_unp() -> BrainTransformerConfig:  # 20,833,543 params; 10764 patches; 1325.150 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: none, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="none",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t4p_unp() -> BrainTransformerConfig:  # 12,956,935 params; 1404 patches; 49.821 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: pool, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="pool",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t4c1_unp() -> BrainTransformerConfig:  # 12,956,977 params; 1404 patches; 49.856 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: conv over local volumes, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv1",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t4c2_unp() -> BrainTransformerConfig:  # 12,959,769 params; 1404 patches; 50.294 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: conv over global volumes, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv2",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


### t8
def brt_small_t8n_unp() -> BrainTransformerConfig:  # 22,934,791 params; 1404 patches; 77.839 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: none, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="none",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t8p_unp() -> BrainTransformerConfig:  # 16,512,775 params; 210 patches; 7.039 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: pool, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="pool",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t8c1_unp() -> BrainTransformerConfig:  # 16,512,817 params; 210 patches; 7.074 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: conv over local volumes, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv1",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_t8c2_unp() -> BrainTransformerConfig:  # 16,515,609 params; 210 patches; 7.512 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: conv over global volumes, embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv2",
        ),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


### v/2
def brt_small_vp2_unp() -> BrainTransformerConfig:  # 75,956,743 params; 27 patches; 4.097 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: pool(2), embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=2),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_vc2_unp() -> BrainTransformerConfig:  # 75,956,805 params; 27 patches; 4.133 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: conv(2), embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=2),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


### v/4
def brt_small_vp4_unp() -> BrainTransformerConfig:  # 19,167,751 params; 27 patches; 1.030 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: pool(4), embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=4),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


def brt_small_vc4_unp() -> BrainTransformerConfig:  # 19,167,869 params; 27 patches; 1.066 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: conv(4), embed dim: 768, # heads: 3, # trans layers: 3, linear dim: 768",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=4),
        d_model=768,
        n_heads=3,
        n_layers=3,
        dff=768,
        p_drop=_P_DROP,
    )


# brt_medium
## prep
### t4
def brt_medium_t4n_prep() -> BrainTransformerConfig:  # 50,447,367 params; 9576 patches; 3011.015 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: none, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="none",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t4p_prep() -> BrainTransformerConfig:  # 40,952,839 params; 1200 patches; 128.168 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: pool, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="pool",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t4c1_prep() -> BrainTransformerConfig:  # 40,952,881 params; 1200 patches; 128.196 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: conv over local volumes, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv1",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t4c2_prep() -> BrainTransformerConfig:  # 40,955,673 params; 1200 patches; 128.550 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: conv over global volumes, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv2",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


### t8
def brt_medium_t8n_prep() -> BrainTransformerConfig:  # 54,379,527 params; 1320 patches; 179.992 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: none, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="none",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t8p_prep() -> BrainTransformerConfig:  # 45,841,415 params; 150 patches; 14.007 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: pool, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="pool",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t8c1_prep() -> BrainTransformerConfig:  # 45,841,457 params; 150 patches, 14.036 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: conv over local volumes, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv1",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t8c2_prep() -> BrainTransformerConfig:  # 45,844,249 params; 150 patches, 14.390 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: conv over global volumes, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv2",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


### v/2
def brt_medium_vp2_prep() -> BrainTransformerConfig:  # 108,613,639 params; 27 patches; 5.900 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: pool(2), embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=2),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_vc2_prep() -> BrainTransformerConfig:  # 108,613,701; 27 patches; 5.929 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: conv(2), embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=2),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


### v/4
def brt_medium_vp4_prep() -> BrainTransformerConfig:  # 47,378,439 params; 27 patches; 2.593 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: pool(4), embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=4),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_vc4_prep() -> BrainTransformerConfig:  # 47,378,557 params; 27 patches; 2.622 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: conv(4), embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=4),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


## unp
### t4
def brt_medium_t4n_unp() -> BrainTransformerConfig:  # 51,663,879 params; 10764 patches; 3698.821 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: none, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="none",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t4p_unp() -> BrainTransformerConfig:  # 41,161,735 params; 1404 patches; 156.982 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: pool, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="pool",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t4c1_unp() -> BrainTransformerConfig:  # 41,161,777 params; 1404 patches; 157.017 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: conv over local volumes, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv1",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t4c2_unp() -> BrainTransformerConfig:  # 41,164,569 params; 1404 patches; 157.455 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: conv over global volumes, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv2",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


### t8
def brt_medium_t8n_unp() -> BrainTransformerConfig:  # 54,465,543 params; 1404 patches; 194.339 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: none, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="none",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t8p_unp() -> BrainTransformerConfig:  # 45,902,855 params; 210 patches; 19.889 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: pool, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="pool",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t8c1_unp() -> BrainTransformerConfig:  # 45,902,897 params; 210 patches; 19.924 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: conv over local volumes, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv1",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_t8c2_unp() -> BrainTransformerConfig:  # 45,905,689 params; 210 patches; 20.362 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: conv over global volumes, embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv2",
        ),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


### v/2
def brt_medium_vp2_unp() -> BrainTransformerConfig:  # 125,161,479 params; 27 patches; 6.793 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: pool(2), embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=2),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_vc2_unp() -> BrainTransformerConfig:  # 125,161,541 params; 27 patches; 6.830 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: conv(2), embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=2),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


### v/4
def brt_medium_vp4_unp() -> BrainTransformerConfig:  # 49,442,823 params; 27 patches; 2.705 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: pool(4), embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=4),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


def brt_medium_vc4_unp() -> BrainTransformerConfig:  # 49,442,941 params; 27 patches; 2.740 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: conv(4), embed dim: 1024, # heads: 4, # trans layers: 6, linear dim: 1024",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=4),
        d_model=1024,
        n_heads=4,
        n_layers=6,
        dff=1024,
        p_drop=_P_DROP,
    )


# brt_large
## prep
### t4
def brt_large_t4n_prep() -> BrainTransformerConfig:  # 94,869,767 params; 9576 patches; 5305.504 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: none, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="none",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t4p_prep() -> BrainTransformerConfig:  # 83,001,607 params; 1200 patches; 250.541 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: pool, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="pool",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t4c1_prep() -> BrainTransformerConfig:  # 83,001,649 params; 1200 patches; 250.569 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: conv over local volumes, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv1",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t4c2_prep() -> BrainTransformerConfig:  # 83,004,441 params; 1200 patches; 250.923 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 4, patch reduction: conv over global volumes, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv2",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


### t8
def brt_large_t8n_prep() -> BrainTransformerConfig:  # 99,784,967 params; 1320 patches; 325.970 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: none, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="none",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t8p_prep() -> BrainTransformerConfig:  # 89,112,327 params; 150 patches; 27.243 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: pool, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="pool",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t8c1_prep() -> BrainTransformerConfig:  # 89,112,369 params; 150 patches; 27.272 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: conv over local volumes, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv1",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t8c2_prep() -> BrainTransformerConfig:  # 89,115,161 params; 150 patches; 27.626 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch size: 8, patch reduction: conv over global volumes, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv2",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


### v/2
def brt_large_vp2_prep() -> BrainTransformerConfig:  # 167,577,607 params; 27 patches; 9.145 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: pool(2), embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=2),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_vc2_prep() -> BrainTransformerConfig:  # 167,577,669 params; 27 patches; 9.175 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: conv(2), embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=2),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


# v/4
def brt_large_vp4_prep() -> BrainTransformerConfig:  # 91,033,607 params; 27 patches; 5.012 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: pool(4), embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=4),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_vc4_prep() -> BrainTransformerConfig:  # 91,033,725 params; 27 patches; 5.040 GFLOP
    return BrainTransformerConfig(
        description="preprocessed fMRIs, patch reduction: conv(4), embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=4),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


## unp
def brt_large_t4n_unp() -> BrainTransformerConfig:  # 96,390,407 params; 10764 patches; 6487.467 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: none, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="none",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t4p_unp() -> BrainTransformerConfig:  # 83,262,727 params; 1404 patches; 304.837 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: pool, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="pool",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t4c1_unp() -> BrainTransformerConfig:  # 83,262,769 params; 1404 patches; 304.872 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: conv over local volumes, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv1",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t4c2_unp() -> BrainTransformerConfig:  # 83,265,561 params; 1404 patches; 305.310 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 4, patch reduction: conv over global volumes, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=4,
            reduction="conv2",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


### t8
def brt_large_t8n_unp() -> BrainTransformerConfig:  # 99,892,487 params; 1404 patches; 351.534 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: none, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="none",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t8p_unp() -> BrainTransformerConfig:  # 89,189,127 params; 210 patches; 38.593 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: pool, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="pool",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t8c1_unp() -> BrainTransformerConfig:  # 89,189,169 params; 210 patches; 38.628 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: conv over local volumes, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv1",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_t8c2_unp() -> BrainTransformerConfig:  # 89,191,961 params; 210 patches; 39.066 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch size: 8, patch reduction: conv over global volumes, embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedTimeConfig(
            patch_size=8,
            reduction="conv2",
        ),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


### v/2
def brt_large_vp2_unp() -> BrainTransformerConfig:  # 188,262,407 params; 27 patches; 10.262 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: pool(2), embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=2),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_vc2_unp() -> BrainTransformerConfig:  # 188,262,469 params; 27 patches; 10.298 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: conv(2), embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=2),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


# v/4
def brt_large_vp4_unp() -> BrainTransformerConfig:  # 93,614,087 params; 27 patches; 5.151 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: pool(4), embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="pool", reduction_factor=4),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )


def brt_large_vc4_unp() -> BrainTransformerConfig:  # 93,614,205 params; 27 patches 5.187 GFLOP
    return BrainTransformerConfig(
        description="unprocessed fMRIs, patch reduction: conv(4), embed dim: 1280, # heads: 4, # trans layers: 8, linear dim: 1280",
        input_shape=(_WANG_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        n_classes=_N_CLASSES,
        patch_embed=PatchEmbedVolumeConfig(reduction="conv", reduction_factor=4),
        d_model=1280,
        n_heads=4,
        n_layers=8,
        dff=1280,
        p_drop=_P_DROP,
    )
