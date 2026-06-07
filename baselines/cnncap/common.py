import random
from collections import OrderedDict

import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count

    def update(self, value, count=1):
        self.sum += value * count
        self.count += count


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def clean_state_dict(state_dict):
    return OrderedDict((key.removeprefix("module."), value) for key, value in state_dict.items())
