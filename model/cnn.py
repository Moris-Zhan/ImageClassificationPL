import pytorch_lightning as pl

import torch.nn as nn
import torch

from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.utils import *
from LightningFunc.losses import *


class CNN(pl.LightningModule):
    def __init__(self, num_classes, target, args):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.args = args

        self.criterion = get_criterion(self.args.criterion)# 使用序列工具快速构建
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(128 * 128 * 32, num_classes)
        self.target = target
        self.checkname = "CNN"

        setattr(CNN, "training_step", training_step)
        setattr(CNN, "training_epoch_end", training_epoch_end)
        setattr(CNN, "validation_step", validation_step)
        setattr(CNN, "validation_epoch_end", validation_epoch_end)
        setattr(CNN, "test_step", test_step)
        setattr(CNN, "test_epoch_end", test_epoch_end)
        setattr(CNN, "configure_optimizers", configure_optimizers)
        setattr(CNN, "accuracy_score", accuracy_score)   
        setattr(CNN, "saveDetail", saveDetail) 
        setattr(CNN, "writeCSV", writeCSV)
        setattr(CNN, "write_Best_model_path", write_Best_model_path)
        setattr(CNN, "read_Best_model_path", read_Best_model_path)



    def forward(self, x):
        # forward 在 pytorch_lightning 裡頭是用來做 prediction 
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        return out
        
    