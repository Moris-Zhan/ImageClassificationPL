import os
from torchinfo import summary
import torch
import numpy as np
import matplotlib.pyplot as plt

def saveDetail(self):
    self.sample = (8, 3, 512, 512)
    model_stats = summary(self.model.cuda(), self.sample)
    dir = os.path.join("log_dir", self.data_name, self.checkname)
    os.makedirs(dir, exist_ok=True)
    summary_str = str(model_stats)
    summaryfile = os.path.join(dir, 'summary.txt')
    summary_file = open(summaryfile, 'w', encoding="utf-8")
    summary_file.write(summary_str + '\n')
    summary_file.close()


def writeCSV(self):    
    dir = os.path.join("log_dir", self.data_name ,self.checkname)  
    self.target.to_csv(os.path.join(dir, 'pred.csv'), index=False)     
    tensorboard_logs = {'Info': "Write Successed!!!"}
    return tensorboard_logs

def write_Best_model_path(self):  
    dir = os.path.join("log_dir", self.data_name ,self.checkname)  
    best_model_path = self.checkpoint_callback.best_model_path
    if (best_model_path!=""):
        best_model_file = os.path.join(dir, 'best_model_path.txt')
        best_model_file = open(best_model_file, 'w', encoding="utf-8")
        print("\nWrite best_model_path : %s \n"% best_model_path)
        best_model_file.write(best_model_path + '\n')
        best_model_file.close()

def read_Best_model_path(self):  
    dir = os.path.join("log_dir", self.data_name ,self.checkname)  
    best_model_file = os.path.join(dir, 'best_model_path.txt')
    if os.path.exists(best_model_file):
        best_model_file = open(best_model_file, 'r', encoding="utf-8")
        best_model_path = best_model_file.readline().strip()
        print("\nLoad best_model_path : %s \n"% best_model_path)
        self.load_from_checkpoint(checkpoint_path=best_model_path, num_classes=self.num_classes, target=self.target)
        best_model_file.close()    
    else:
        print("No model can load \n")
        if not os.path.exists(best_model_file):
            self.saveDetail()


def makegrid(output,numrows):
    outer=(torch.Tensor.cpu(output).detach())
    plt.figure(figsize=(20,5))
    b=np.array([]).reshape(0,outer.shape[2])
    c=np.array([]).reshape(numrows*outer.shape[2],0)
    i=0
    j=0
    while(i < outer.shape[1]):
        img=outer[0][i]
        b=np.concatenate((img,b),axis=0)
        j+=1
        if(j==numrows):
            c=np.concatenate((c,b),axis=1)
            b=np.array([]).reshape(0,outer.shape[2])
            j=0            
        i+=1
    return c
                    