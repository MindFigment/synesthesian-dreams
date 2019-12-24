import math
import torch.nn as nn


def feature_maps_multiplies(img_size):
        layers_num = int(math.log(img_size, 2) - 2)
        ms = multiplies(layers_num)
        return ms
    

def multiplies(m):
    l = []
    for i in range(m - 1):
        if len(l) > 0:
            l.append(l[i-1] * 2)
        else:
            l.append(2)
    return l    

