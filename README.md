# ImageClassification
My Frame work for ImageClassification 
## Overview
I organizize the object detection algorithms proposed in recent years, and focused on **`AOI`** and **`SCUT`** Dataset.
This frame work also include **`EarlyStopping mechanism`**.


## Datasets:

I used 3 different datases: **`AOI`, `SCUT`** . Statistics of datasets I used for experiments is shown below

- **AOI**:
This topic takes flexible electronic displays as the inspection target, and hopes to interpret the classification of defects through data science to improve the effectiveness of AOI.

  Download the classification images and organize folder from [AOI](https://aidea-web.tw/topic/252eb73e-78d0-4024-8937-40ed20187fd8). Make sure to put the files as the following structure:
  
The image data provided in this topic includes 6 categories (normal category + 5 defect categories).

```
  AOI
  ├── train_images
  │   ├── 0
  │   ├── 1
  │   ├── 2  
  │   ├── 3 
  │   ├── 4 
  │   ├── 5 
  │     
  │── test_images
      ├── 0 (Default)
```
- **SCUT-FBP5500**:
A diverse benchmark database (Size = 172MB) for multi-paradigm facial beauty prediction is now released by Human Computer Intelligent Interaction Lab of South China University of Technology.

Download the classification images and organize folder from [SCUT](https://drive.google.com/open?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf). Make sure to put the files as the following structure:
  
The SCUT-FBP5500 dataset has totally 5500 frontal faces with diverse properties (male/female, Asian/Caucasian, ages) and diverse labels (facial landmarks, beauty scores in 5 scales, beauty score distribution).
I use the **`round of mean beauty score`** to train the classification model.

```
  SCUT-FBP5500_v2
  ├── train_images
  │   ├── 0
  │   ├── 1
  │   ├── 2  
  │   ├── 3 
  │   ├── 4 
  │     
  │── test_images
      ├── 0 (Default)
```

## Methods
- **cnn**
- **resnet**
- **pre_resnet**
- **pre_squeezenet**
- **pre_mobilenet**
- **pre_shufflenet**
- **pre_googlenet**
- **pre_inception**

## Prerequisites
* **Windows 10**
* **CUDA 10.1 (lower versions may work but were not tested)**
* **NVIDIA GPU 1660 + CuDNN v7.3**
* **python 3.6.9**
* **pytorch 1.10**
* **opencv (cv2)**
* **numpy**
* **torchvision 0.4**
* **torchsummary**
* **dlib==19.21**

## Usage
### 0. Prepare the dataset
* **Download custom ImageFolder dataset in the  `data_paths`.** 
* **And create custom dataset `custom_dataset.py` in the `dataset`.**
* **For `SCUT` I prepare code to predict face in `predict.py`, which need prepare face_landmarks and dlib library to make `frontal_face_detector`, these data are put in extra**.
* **`shape_predictor_68_face_landmarks.dat` download from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)**.

### 1. Train + Evaluate
#### AOI (include predict Dataframe)
```python
python train.py --model pre_googlenet --dataset AOI --batch_size 4 --n_gpu 1
```

#### SCUT
```python
python train_facerank.py --model pre_googlenet --dataset SCUT --batch_size 4 --n_gpu 1
```

### 2. Predict Face Score
```python
python predict.py --model pre_googlenet --dataset SCUT --experiment_dir "run\AOI\pre_googlenet\experiment_2"
```

## Reference
- dlib-models : https://github.com/davisking/dlib-models
- Transfer Learning - Fine tune : https://hackmd.io/@lido2370/HyLTOlSn4?type=view
- ImageFolder : https://blog.csdn.net/TH_NUM/article/details/80877435
- pytorch-summary : https://github.com/sksq96/pytorch-summary
