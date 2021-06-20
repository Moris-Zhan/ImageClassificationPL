# import os
# from argparse import ArgumentParser

from unicodedata import name
import pytorch_lightning as pl
# import torch
# from torch import nn as nn
# from torch.nn import functional as F
# from dataset import AOI
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, GPUStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


import pandas as pd
import os
import gc

import time
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
from model.cnn import CNN
from model.resNet import ResNet
from model.shuffleNet import ShuffleNet
from model.vggNet import VGGNet
from model.alexNet import AlexNet
from model.googleNet import GoogleNet
from model.denseNet import DenseNet
from model.inception import Inception
from model.mobileNet import MobileNet
from model.squeezeNet import SqueezeNet
from torchinfo import summary
from dataset.AOI import AOIModule
from dataset.EdgeAOI import EdgeAOIModule
from dataset.SCUT import SCUTModule
from dataset.HandWrite import HandWriteModule
from copy import deepcopy
import yaml
import argparse

def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            if isinstance(value, dict):
                for inside_key, inside_key_value in value.items():
                    setattr(args, inside_key, inside_key_value)
            else:
                setattr(args, key, value)
    return args

def load_data(args):
    dm = None
    if args.data_module == "AOIModule": dm = AOIModule(batch_size= args.batch_size)
    elif args.data_module == "EdgeAOIModule": dm = EdgeAOIModule(batch_size= args.batch_size)
    elif args.data_module == "SCUTModule": dm = SCUTModule(batch_size= args.batch_size)
    elif args.data_module == "HandWriteModule": dm = HandWriteModule(batch_size= args.batch_size)
    dm.setup(args.stage)
    return dm

def load_model(args, dm):
    model = None
    if args.model_name == "CNN": model = CNN(dm.get_classes(), dm.target, args)
    elif args.model_name == "ResNet": model = ResNet(dm.get_classes(), dm.target, args)
    elif args.model_name == "ShuffleNet": model = ShuffleNet(dm.get_classes(), dm.target, args)
    elif args.model_name == "VGGNet": model = VGGNet(dm.get_classes(), dm.target, args)
    elif args.model_name == "AlexNet": model = AlexNet(dm.get_classes(), dm.target, args)
    elif args.model_name == "GoogleNet": model = GoogleNet(dm.get_classes(), dm.target, args)
    elif args.model_name == "DenseNet": model = DenseNet(dm.get_classes(), dm.target, args)
    elif args.model_name == "MobileNet": model = MobileNet(dm.get_classes(), dm.target, args)
    elif args.model_name == "SqueezeNet": model = SqueezeNet(dm.get_classes(), dm.target, args)
    elif args.model_name == "Inception": model = Inception(dm.get_classes(), dm.target, args)

    return model

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/config.yaml',
            help='YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args)
    dm = load_data(args)
    model = load_model(args, dm)

    model.read_Best_model_path()

    root_dir = os.path.join('log_dir',args.data_module)
    logger = TensorBoardLogger(root_dir, name= model.checkname, default_hp_metric =False )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        dirpath= os.path.join(root_dir, model.checkname), 
        filename= model.checkname + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        verbose=True
    )
    setattr(model, "checkpoint_callback", checkpoint_callback)
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    gpu_stats = GPUStatsMonitor() 

    # AVAIL_GPUS = min(1, torch.cuda.device_count())

    trainer = Trainer.from_argparse_args(config, 
                                        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, gpu_stats],
                                        logger=logger)
    if args.tune:
        trainer.tune(model, datamodule=dm)

    trainer.fit(model, datamodule=dm)
    
    print("best_model_path : %s" % checkpoint_callback.best_model_path)

    if dm.name == "SCUTModule":
        import matplotlib.image as img 
        import cv2
        from LightningFunc.face import predict_image
        image = img.imread("extra//test.jpg")
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # opencvImage 
        score = predict_image(image, model)
        print("Predict Score : {}".format(score))

    else:
        dm.setup('test')
        trainer.test(model, datamodule=dm)

    # tensorboard --logdir=D:\WorkSpace\JupyterWorkSpace\ImageClassificationPL\log_dir

                