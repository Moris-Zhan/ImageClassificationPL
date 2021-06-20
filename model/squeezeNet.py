import pytorch_lightning as pl

from torchvision import models
import torch.nn as nn
from torch.nn import Dropout
import torch
import torchvision
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


class SqueezeNet(pl.LightningModule):
    def __init__(self, num_classes, target, args):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.args = args

        self.__build_model()
        self.criterion = get_criterion(self.args.criterion)
        setattr(SqueezeNet, "training_step", training_step)
        setattr(SqueezeNet, "training_epoch_end", training_epoch_end)
        setattr(SqueezeNet, "validation_step", validation_step)
        setattr(SqueezeNet, "validation_epoch_end", validation_epoch_end)
        setattr(SqueezeNet, "test_step", test_step)
        setattr(SqueezeNet, "test_epoch_end", test_epoch_end)
        setattr(SqueezeNet, "configure_optimizers", configure_optimizers)
        setattr(SqueezeNet, "accuracy_score", accuracy_score)   
        setattr(SqueezeNet, "saveDetail", saveDetail) 
        setattr(SqueezeNet, "writeCSV", writeCSV)
        setattr(SqueezeNet, "write_Best_model_path", write_Best_model_path)
        setattr(SqueezeNet, "read_Best_model_path", read_Best_model_path)


        self.target = target
        self.checkname = self.backbone

    def __build_model(self):
        # 1. Load pre-trained network:
        self.backbone = "squeezenet1_0"
        self.backbone = "squeezenet1_1"
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        for param in backbone.parameters():
            param.requires_grad = False

        # 2. Classifier:
        net = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1,1)) 
        net.apply(init_weights)
        # 把squeezenet1的classifier替换成自己设置的classifier
        backbone.classifier[1] = net

        self.model = backbone

        

    def forward(self, x):
        return self.model(x)
        
   