from glob import glob
import os
from extra.face import get_face, predict_image
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

DCARD_DIR = "D:\\WorkSpace\\JupyterWorkSpace\\AUTO_DCARD\\DCARD\\Image"

paths = glob(os.path.join(DCARD_DIR, "*","*.jpg"))

len(paths)

# load best face rank model
model = torch.load('run\SCUT\pre_googlenet\experiment_6\pre_googlenet.pkl')
model.load_state_dict(torch.load('run\SCUT\pre_googlenet\experiment_6\checkpoint.pth.tar')['state_dict'])
model.eval()
# model = model.cpu()


scores = []
move_path = []
names = []
with tqdm(total = len(paths)) as pbar:
    for idx, img_path in enumerate(paths):
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1)
        pbar.update(1)
        if type(image) == type(None): continue
        image = image[:, :, :3]
        score = predict_image(None, image, model) # predict target
        if score == 0: continue
        pbar.set_description("Path: {}, Score: {}".format(os.path.basename(img_path), score))
        names.append(os.path.basename(img_path))
        move_path.append(img_path)
        scores.append(score)

# make score df
d = {'paths': move_path, 'names':names, 'scores': scores }
df = pd.DataFrame(data=d)
df.to_excel("dcard_score.xlsx")