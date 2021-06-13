import pandas as pd
import os

import time
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from sklearn.utils import shuffle
import shutil
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset

class SCUT(pl.LightningDataModule):
    def train_img_path(self, id):
        return os.path.join(self.root, 'Images\\%s' % (id)) 
    def train_cls_path(self, id, cls):
        return os.path.join(self.root, 'train_images', '%s\\%s'%(int(cls),id))       
    def train_cls_path(self, id, cls):
        return os.path.join(self.root, 'test_images', '%s\\%s'%(int(cls),id))  
    
    def __init__(self, processed = True):
        super().__init__()
        self.root = 'D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\SCUT-FBP5500_v2\\'
        df = pd.read_excel(os.path.join(self.root, "All_Ratings.xlsx"), sheet_name="ALL") 
        df = df.groupby('Filename').mean().reset_index()
        df = df.round(0)
        df['Rating'] = df['Rating'].apply(lambda x : int(x)-1)
        df = shuffle(df)

        train_df = df[:int(len(df) * 0.8)]
        test_df = df[int(len(df) * 0.8):]

        if not processed:
            # ------------------------------------------------------------------------------------------       
            train_df['image_path'] = train_df['Filename'].apply(self.train_img_path)
            train_df['cls_path'] = train_df.apply(lambda x : self.train_cls_path(x['Filename'],x['Rating']),axis=1)

            test_df['image_path'] = test_df['Filename'].apply(self.train_img_path)
            test_df['cls_path'] = test_df.apply(lambda x : self.test_cls_path(x['Filename'],x['Rating']),axis=1)
            # ------------------------------------------------------------------------------------------   
            with tqdm(total=len(train_df)) as pbar:
                for i in range(len(train_df)):
                    series = train_df.iloc[i]
                    source, dest = series['image_path'], series['cls_path']
                    shutil.move(source,dest)
                    pbar.update(1)
                    
            with tqdm(total=len(test_df)) as pbar:
                for i in range(len(test_df)):
                    series = test_df.iloc[i]
                    source, dest = series['image_path'], series['cls_path']
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

class SCUTModule(pl.LightningDataModule):
       
    def __init__(self, bsz):
        super().__init__()
        SCUT()
        self.batch_size = bsz
        self.name = "SCUTModule"
        self.root = 'D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\SCUT-FBP5500_v2\\'
        self.classes = list(range(1,6))    
        self.target = None
         
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            train_set = ImageFolder(os.path.join(self.root, 'train_images'))
            val_set = ImageFolder(os.path.join(self.root, 'test_images'))  
            
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
    dm = SCUTModule(processed=True, bsz=5)
    dm.setup('fit')