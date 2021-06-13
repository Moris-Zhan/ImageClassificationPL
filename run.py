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

if __name__ == '__main__':
    # dm = AOIModule(bsz=5)
    # dm = EdgeAOIModule(bsz=5)
    # dm = SCUTModule(bsz=5)
    dm = HandWriteModule(bsz=5)
    dm.setup('fit')

    # model = CNN(dm.get_classes(), dm.target)
    # model = ResNet(dm.get_classes(), dm.target)
    # model = ShuffleNet(dm.get_classes(), dm.target)
    # model = VGGNet(dm.get_classes(), dm.target)
    # model = AlexNet(dm.get_classes(), dm.target)
    # model = GoogleNet(dm.get_classes(), dm.target)
    # model = DenseNet(dm.get_classes(), dm.target)
    # model = MobileNet(dm.get_classes(), dm.target)
    model = SqueezeNet(dm.get_classes(), dm.target)
    # model = Inception(dm.get_classes(), dm.target)

    setattr(model, "data_name", dm.name)
    setattr(model, "learning_rate", 1e-3)
    model.read_Best_model_path()

    root_dir = os.path.join('log_dir',dm.name)
    logger = TensorBoardLogger(root_dir, name= model.checkname, default_hp_metric =False )

    checkpoint_callback = ModelCheckpoint(
        monitor='Loss/Val', 
        dirpath= os.path.join(root_dir, model.checkname), 
        filename= model.checkname + '-{epoch:02d}-{Loss/Val:.2f}',
        save_top_k=3,
        mode='min',
        verbose=True
    )
    setattr(model, "checkpoint_callback", checkpoint_callback)
    
    early_stop_callback = EarlyStopping(
        monitor='Loss/Val',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    gpu_stats = GPUStatsMonitor() 

    # AVAIL_GPUS = min(1, torch.cuda.device_count())
    trainer = Trainer(max_epochs = 10, gpus=-1, auto_select_gpus=True, logger=logger, num_sanity_val_steps=0, \
                        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, gpu_stats], \
                        accumulate_grad_batches=1, auto_lr_find=True, auto_scale_batch_size = 'power')

    # trainer.tune(model, datamodule=dm)

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

                