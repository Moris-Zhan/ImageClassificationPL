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


class ShuffleNet(pl.LightningModule):
    def __init__(self, num_classes, target, args):
        super(ShuffleNet, self).__init__()
        self.num_classes = num_classes
        self.args = args

        self.__build_model()
        self.criterion = get_criterion(self.args.criterion)
        setattr(ShuffleNet, "training_step", training_step)
        setattr(ShuffleNet, "training_epoch_end", training_epoch_end)
        setattr(ShuffleNet, "validation_step", validation_step)
        setattr(ShuffleNet, "validation_epoch_end", validation_epoch_end)
        setattr(ShuffleNet, "test_step", test_step)
        setattr(ShuffleNet, "test_epoch_end", test_epoch_end)
        setattr(ShuffleNet, "configure_optimizers", configure_optimizers)
        setattr(ShuffleNet, "accuracy_score", accuracy_score)   
        setattr(ShuffleNet, "saveDetail", saveDetail) 
        setattr(ShuffleNet, "writeCSV", writeCSV)
        setattr(ShuffleNet, "write_Best_model_path", write_Best_model_path)
        setattr(ShuffleNet, "read_Best_model_path", read_Best_model_path)
               

        self.target = target
        self.checkname = self.backbone


    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        self.backbone = "shufflenet_v2_x0_5"
        self.backbone = "shufflenet_v2_x1_0"
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        for param in backbone.parameters():
            param.requires_grad = False

        # # 2. Feature extraction:
        # _layers = list(backbone.children())[:-1]        
        # self.feature_extractor = nn.Sequential(*_layers)

        # 3. Classifier:
        net = nn.Linear(backbone.fc.in_features, self.num_classes) 
        net.apply(init_normal)   
        backbone.fc = net
        self.model = backbone

        
               
    def forward(self, x):
        return self.model(x)
    
   