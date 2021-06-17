import torch
import torch.nn as nn

def init_init(init_conf):
    init_class = eval(init_conf.get('select', 'nn.xavier_uniform'))
    layer_init = LAYER_INIT(init_class)
    return layer_init

def conv_layer_init(layer, init_func):
    init_func(layer.weight.data)
    if layer.bias is not None:
        if len(layer.bias.data.shape) > 1:
            init_func(layer.bias.data)
        else:
            torch.nn.init.normal_(layer.bias.data)

class LAYER_INIT:
    def __init__(self, init_class) -> None:
        self.init_class = init_class

    def __call__(self, layer):
        if isinstance(layer, nn.Conv2d):
            conv_layer_init(layer, self.init_class)
        elif isinstance(layer, nn.BatchNorm2d):
            torch.nn.init.normal_(layer.bias.data)