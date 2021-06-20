import torch.nn as nn

def get_criterion(criterion):
    if criterion == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif criterion == "NLLLoss":
        return nn.NLLLoss()
