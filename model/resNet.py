
import pytorch_lightning as pl

import torch.nn as nn
import torch
from torchvision import models
from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.utils import *

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
# Residual Block
class ResidualBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class MyResNet(pl.LightningModule):
    def __init__(self, num_classes, target, checkname):
        super(MyResNet, self).__init__()
        # 使用序列工具快速构建
        block = ResidualBlock
        layers = [2, 2, 2, 2]
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(128 * 128, num_classes)
        

        setattr(MyResNet, "training_step", training_step)
        setattr(MyResNet, "training_epoch_end", training_epoch_end)
        setattr(MyResNet, "validation_step", validation_step)
        setattr(MyResNet, "validation_epoch_end", validation_epoch_end)
        setattr(MyResNet, "test_step", test_step)
        setattr(MyResNet, "test_epoch_end", test_epoch_end)
        setattr(MyResNet, "configure_optimizers", configure_optimizers)
        setattr(MyResNet, "accuracy_score", accuracy_score)   
        setattr(MyResNet, "saveDetail", saveDetail) 
        setattr(MyResNet, "writeCSV", writeCSV)
        setattr(MyResNet, "write_Best_model_path", write_Best_model_path)
        setattr(MyResNet, "read_Best_model_path", read_Best_model_path)

        self.criterion = nn.CrossEntropyLoss()
        self.target = target
        self.checkname = checkname

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # forward 在 pytorch_lightning 裡頭是用來做 prediction 
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class ResNet(pl.LightningModule):
    def __init__(self, num_classes, target):
        super(ResNet, self).__init__()
        self.num_classes = num_classes

        self.__build_model()

        setattr(ResNet, "training_step", training_step)
        setattr(ResNet, "training_epoch_end", training_epoch_end)
        setattr(ResNet, "validation_step", validation_step)
        setattr(ResNet, "validation_epoch_end", validation_epoch_end)
        setattr(ResNet, "test_step", test_step)
        setattr(ResNet, "test_epoch_end", test_epoch_end)
        setattr(ResNet, "configure_optimizers", configure_optimizers)
        setattr(ResNet, "accuracy_score", accuracy_score)   
        setattr(ResNet, "saveDetail", saveDetail) 
        setattr(ResNet, "writeCSV", writeCSV)
        setattr(ResNet, "write_Best_model_path", write_Best_model_path)
        setattr(ResNet, "read_Best_model_path", read_Best_model_path)
        
        self.criterion = nn.CrossEntropyLoss()
        self.target = target
        self.checkname = self.backbone


    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        self.backbone = "resnet34"
        self.backbone = "resnet50"
        self.backbone = "resnet101"
        self.backbone = "resnet152"
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        for param in backbone.parameters():
            param.requires_grad = False

        # 2. Classifier:
        net = nn.Linear(backbone.fc.in_features, self.num_classes)
        net.apply(init_normal)   
        backbone.fc = net

        self.model = backbone

        
        
     
    def forward(self, x):
        return self.model(x)