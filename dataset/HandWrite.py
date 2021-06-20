import pandas as pd
import os

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import shutil
from torchvision.transforms.transforms import Resize
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset
from glob import glob

class HandWrite(Dataset):    
    def __init__(self, processed = True):
        super().__init__()
        processed = False
        self.root = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\中文手寫影像辨識"

        self.classes = list()
        with open(os.path.join(self.root, "training data dic.txt"), "r", encoding="utf-8") as f:
            self.classes = f.readlines()
        
        self.text_to_id = {}
        self.id_to_text = {}

        if not processed:
            for idx, clas in enumerate(self.classes):
                clas = clas.strip()
                self.text_to_id[clas] = idx
                self.id_to_text[idx] = clas
                if not os.path.exists(os.path.join(self.root,"train", str(idx))):
                    os.makedirs(os.path.join(self.root,"train",str(idx)))

            files = glob(os.path.join(self.root,"train","*.jpg"))
            with tqdm(total=len(files)) as pbar:
                for idx, source in enumerate(files):
                    fclas= source.split("_")[1].split(".")[0]
                    fn = source.split("\\")[-1]
                    dest = os.path.join(self.root,"train", str(self.text_to_id[fclas]), fn)
                    shutil.move(source,dest)
                    pbar.update(1)
    
            

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        x = np.array(x) 
        transformed = self.transform(image=x)
        image, label = transformed["image"], y

        return image, label        


    def __len__(self):
        return len(self.subset)

class HandWriteModule(pl.LightningDataModule):
    # target = None
    def img_path(self, id):
        img_path = os.path.join(self.root, '%s_images\\%s' % ('train', id))
        return img_path
    def cls_path(self, id, cls):
        cls_path = os.path.join(self.root, '%s_images'%("test"), '%s\\%s'%(cls,id) )
        return cls_path   
    
    def __init__(self, batch_size):
        super().__init__()
        
        self.batch_size = batch_size
        self.name = "HandWriteModule"
        self.root = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\中文手寫影像辨識"
        self.classes = HandWrite().classes  
        self.target = None
    
        

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            full_dataset = ImageFolder(os.path.join(self.root, 'train'))
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_set, val_set = torch.utils.data.dataset.random_split(full_dataset, [train_size, val_size])
            self.train_dataset = DatasetFromSubset(
                train_set, 
                transform = A.Compose([
                        A.Resize(512, 512, p=1),
                        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.1),
                        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.1),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                        ToTensorV2(),
                    ])
            )
            self.val_dataset = DatasetFromSubset(
                val_set, 
                transform = A.Compose([
                        A.Resize(512, 512, p=1),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                        ToTensorV2(),
                    ])
            )
            self.num_classes = len(self.classes)

        if stage == 'test' or stage is None:             
            full_dataset = ImageFolder(os.path.join(self.root, 'train'))

            test_size = int(len(full_dataset) * 0.1)
            test_set, _ = torch.utils.data.dataset.random_split(full_dataset, [test_size, ( len(full_dataset) - test_size)])

            self.test_dataset = DatasetFromSubset(
                test_set, 
                transform = A.Compose([
                    A.Resize(512, 512, p=1),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ])
            )
    def prepare_data(self):
        # 在该函数里一般实现数据集的下载等，只有cuda:0 会执行该函数
        pass
    def train_dataloader(self):
        return DataLoader(
                dataset=self.train_dataset,# TensorDataset类型数据集
                batch_size=self.batch_size,# mini batch size
                shuffle=True,# 设置随机洗牌
                num_workers=5# 加载数据的进程个数
            )

    def val_dataloader(self):
        return DataLoader(
                dataset=self.val_dataset,# TensorDataset类型数据集
                batch_size=self.batch_size,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=5# 加载数据的进程个数
            )

    def test_dataloader(self):
        return DataLoader(
                dataset=self.test_dataset,# TensorDataset类型数据集
                batch_size=1,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=5# 加载数据的进程个数
            )
    def get_classes(self):
        return self.num_classes

if __name__ == '__main__':    
    dm = HandWriteModule(processed=True, batch_size=5)
    dm.setup('fit')
