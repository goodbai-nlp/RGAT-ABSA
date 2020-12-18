import torch
from torch import nn, optim
from torch.optim import Optimizer

def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=l2) 
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=l2) 
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

# load args config
def load_config(filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    return dump['config']
