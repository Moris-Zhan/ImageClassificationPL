import pytorch_lightning as pl

from torchvision import models
import torch.nn as nn
from torch.nn import Dropout
import torch

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


class VGGNet(pl.LightningModule):
    def __init__(self, num_classes, target):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes

        self.__build_model()

        setattr(VGGNet, "training_step", training_step)
        setattr(VGGNet, "training_epoch_end", training_epoch_end)
        setattr(VGGNet, "validation_step", validation_step)
        setattr(VGGNet, "validation_epoch_end", validation_epoch_end)
        setattr(VGGNet, "test_step", test_step)
        setattr(VGGNet, "test_epoch_end", test_epoch_end)
        setattr(VGGNet, "configure_optimizers", configure_optimizers)
        setattr(VGGNet, "accuracy_score", accuracy_score)   
        setattr(VGGNet, "saveDetail", saveDetail) 
        setattr(VGGNet, "writeCSV", writeCSV)
        setattr(VGGNet, "write_Best_model_path", write_Best_model_path)
        setattr(VGGNet, "read_Best_model_path", read_Best_model_path)
        
        self.criterion = nn.CrossEntropyLoss()
        self.target = target
        self.checkname = self.backbone
        

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        self.backbone = "vgg11_bn"
        self.backbone = "vgg11"
        self.backbone = "vgg13_bn"
        self.backbone = "vgg13"
        self.backbone = "vgg16_bn"
        self.backbone = "vgg16"
        self.backbone = "vgg19_bn"
        self.backbone = "vgg19"

        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        for param in backbone.parameters():
            param.requires_grad = False
    

        # 2. Classifier:
        net = nn.Linear(backbone.classifier[6].in_features, self.num_classes) 
        backbone.classifier[6] = net 
        self.model = backbone

    def forward(self, x):
        return self.model(x)
        
    