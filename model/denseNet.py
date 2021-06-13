import pytorch_lightning as pl

from torchvision import models
import torch.nn as nn
from torch.nn import Dropout
import torch
import torchvision
import torch.nn.functional as F

from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.utils import *

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class DenseNet(pl.LightningModule):
    def __init__(self, num_classes, target):
        super(DenseNet, self).__init__()
        self.num_classes = num_classes

        self.__build_model()

        setattr(DenseNet, "training_step", training_step)
        setattr(DenseNet, "training_epoch_end", training_epoch_end)
        setattr(DenseNet, "validation_step", validation_step)
        setattr(DenseNet, "validation_epoch_end", validation_epoch_end)
        setattr(DenseNet, "test_step", test_step)
        setattr(DenseNet, "test_epoch_end", test_epoch_end)
        setattr(DenseNet, "configure_optimizers", configure_optimizers)
        setattr(DenseNet, "accuracy_score", accuracy_score)   
        setattr(DenseNet, "saveDetail", saveDetail) 
        setattr(DenseNet, "writeCSV", writeCSV)
        setattr(DenseNet, "write_Best_model_path", write_Best_model_path)
        setattr(DenseNet, "read_Best_model_path", read_Best_model_path)

        self.criterion = nn.CrossEntropyLoss()
        self.target = target
        self.checkname = self.backbone


    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        self.backbone = "densenet121"
        self.backbone = "densenet161"
        self.backbone = "densenet169"
        self.backbone = "densenet201"
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        for param in backbone.parameters():
            param.requires_grad = False

        # 2. Classifier:
        net = nn.Linear(backbone.classifier.in_features, self.num_classes)
        net.apply(init_normal)   
        backbone.classifier = net
        self.model = backbone
               
    def forward(self, x):
        return self.model(x)
        
   