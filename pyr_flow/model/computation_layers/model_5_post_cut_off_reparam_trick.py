import torch.nn as nn

from pyr_flow.constants import *
from pyr_flow.model.layer_module import LayerModule


def inverse_reparam_trick(x: torch.Tensor, lnorm_map):
    return x, lnorm_map
    # x = 1/x
    # log |(x^-1) '| = log | -x^-2| = log ( |x|^-2) = -2 log (|x|)
    lnorm_map += -2 * x.abs().log()
    if (x == 0).any():
        print(x)

    x = 1. / x

    return x, lnorm_map

