"""training should invoke this function"""

import numpy as np
from torch import nn
from . import training


def count_params(model: nn.Module):
    count = 0
    for y in model.parameters():
        count += np.product(y.size())
    return count
