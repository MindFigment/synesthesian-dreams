import torch.nn as nn

def get_loss_criterion(loss):
    if loss == "BCE":
        return nn.BCELoss()