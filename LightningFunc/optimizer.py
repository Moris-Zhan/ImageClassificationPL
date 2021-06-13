import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def configure_optimizers(self):
    optimizer = Adam(self.parameters(), lr=self.learning_rate)
    return {
        'optimizer': optimizer,
        'lr_scheduler': ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9),
        'monitor': 'Loss/Val'
    }

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']    