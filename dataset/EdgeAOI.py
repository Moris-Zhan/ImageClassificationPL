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

class EdgeAOI(Dataset):    
    def img_path(self, id):
        img_path = os.path.join(self.root, '%s_images\\%s' % ('train', id))
        return img_path
    def cls_path(self, id, cls):
        cls_path = os.path.join(self.root, '%s_images'%("train"), '%s\\%s'%(cls,id) )
        return cls_path   
    
    def __init__(self, processed = True):
        super().__init__()
        self.root = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\Edge AI Competition – AOI"
        total_df = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.classes = []
        if not processed:
            total_df['cls_path'] = total_df.apply(lambda x : self.cls_path(x['ID'],x['Label']),axis=1)
            total_df['path'] = total_df['ID'].apply(self.img_path)
            self.classes = list(total_df["Label"].unique())
            for cls in self.classes:
                if not os.path.exists(os.path.join(self.root,"train_images", str(cls))):
                    os.makedirs(os.path.join(self.root,"train_images",str(cls)))

            with tqdm(total=len(total_df)) as pbar:
                for i in range(len(total_df)):
                    series = total_df.iloc[i]
                    source, dest = series['path'], series['cls_path']
                    shutil.move(source, dest)
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

class EdgeAOIModule(pl.LightningDataModule):
    # target = None
    def img_path(self, id):
        img_path = os.path.join(self.root, '%s_images\\%s' % ('train', id))
        return img_path
    def cls_path(self, id, cls):
        cls_path = os.path.join(self.root, '%s_images'%("test"), '%s\\%s'%(cls,id) )
        return cls_path   
    
    def __init__(self, bsz):
        super().__init__()
        EdgeAOI()
        self.batch_size = bsz
        self.name = "EdgeAOIModule"
        self.root = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\Edge AI Competition – AOI"
        self.classes = ["normal", "void", " horizontal defect", "vertical defect", "edge defect", "particle"]    

        # random generate test csv
        n = 100
        target = pd.read_csv(os.path.join(self.root, "train.csv"), header=0, skiprows=lambda i: i % n != 0)
        target['cls_path'] = target.apply(lambda x : self.cls_path(x['ID'], 0),axis=1)
        target['path'] = target['ID'].apply(self.img_path)
        self.target = target

        # test default class-0
        if not os.path.exists(os.path.join(self.root,"test_images", str(0))):
            os.makedirs(os.path.join(self.root,"test_images",str(0)))

        with tqdm(total=len(target)) as pbar:
            for i in range(len(target)):
                series = target.iloc[i]
                source, dest = series['path'], series['cls_path']
                shutil.copy(source, dest)
                pbar.update(1)      
        

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            full_dataset = ImageFolder(os.path.join(self.root, 'train_images'))
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
            # self.base_size, self.crop_size = 1024, 512
              
            test_dataset = ImageFolder(os.path.join(self.root, 'test_images'))

            self.test_dataset = DatasetFromSubset(
                test_dataset, 
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
    dm = EdgeAOIModule(processed=True, bsz=5)
    dm.setup('fit')