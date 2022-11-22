"""
tensor处理相关的工具

"""
from datatransfer.type_def import *

from loguru import logger

import torch


def generate_label(shape: Sequence[int], indexes: Sequence[List[int]], dtype=torch.long):
    """
    创建一个shape形状的全0tensor，然后在indexes中的每个坐标位置填充1
    :param shape:
    :param indexes: index 为1～4维
    :return:
    """
    label = torch.zeros(*shape, dtype=dtype)

    for ind in indexes:
        if len(ind) == 1:
            label[ind[0]] = 1
        elif len(ind) == 2:
            label[ind[0]][ind[1]] = 1
        elif len(ind) == 3:
            label[ind[0]][ind[1]][ind[2]] = 1
        else:
            label[ind[0]][ind[1]][ind[2]][ind[3]] = 1
    return label