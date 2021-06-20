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
from LightningFunc.losses import *

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class Inception(pl.LightningModule):
    def __init__(self, num_classes, target, args):
        super(Inception, self).__init__()
        self.num_classes = num_classes
        self.args = args

        self.__build_model()
        self.criterion = get_criterion(self.args.criterion)
        setattr(Inception, "training_step", training_step)
        setattr(Inception, "training_epoch_end", training_epoch_end)
        setattr(Inception, "validation_step", validation_step)
        setattr(Inception, "validation_epoch_end", validation_epoch_end)
        setattr(Inception, "test_step", test_step)
        setattr(Inception, "test_epoch_end", test_epoch_end)
        setattr(Inception, "configure_optimizers", configure_optimizers)
        setattr(Inception, "accuracy_score", accuracy_score)   
        setattr(Inception, "saveDetail", saveDetail) 
        setattr(Inception, "writeCSV", writeCSV)
        setattr(Inception, "write_Best_model_path", write_Best_model_path)
        setattr(Inception, "read_Best_model_path", read_Best_model_path)

        self.target = target
        self.checkname = self.backbone
        
    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        self.backbone = "inception_v3"
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True, aux_logits=False)
        self.transform_input = backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # 2. Classifier:
        net = nn.Linear(backbone.fc.in_features, self.num_classes)
        net.apply(init_normal)   
        backbone.fc = net
        self.model = backbone

               
    def forward(self, x):
        return self.model(x)
        
   