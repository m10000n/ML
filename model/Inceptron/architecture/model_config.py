from model.Inceptron.architecture.inceptron import InceptronConfig

_WANG_N_VOLUMES = 27
_N_CLASSES = 7


# BrainNet supports arbitrary 4D inputs by automatically resampling the volume to 112×112×112.
def inceptron() -> InceptronConfig:  # 15,161,751 params, 436.891 GFLOP
    return InceptronConfig(description="", input_shape=(_WANG_N_VOLUMES, 112, 112, 112), n_classes=_N_CLASSES)
