# TFBS model
The source code of TFBS model

## Introduction

TFBS model contains four module including input module, temporal feature extraction module, segmentation module and output module. It employs an LSTM model matrix to learn masses of temporal features from raw time-series satellite image and produces an image consists of these temporal features. The temporal feature image is then input to an UNET module to extract spatial context information form temporal features and produce a segmentation image

The architecture of the TFBS model is shown below:
![Image text](https://github.com/younglimpo/TFBSmodel/blob/master/TFBS%20architecture.png)

## Environment

| Library | Version | 
| :-----:| :----: | 
| Python | 3.7.0 | 
| Tensorflow | 2.2.0 | 
| Keras | 2.3.1 | 
| scikit-learn | 0.20.3 | 
| gdal | 2.3.3 | 
| scikit-learn | 0.20.3 | 

## Dataset 

The training dataset are shared by google drive: 
https://drive.google.com/drive/folders/120X2tLv4-6pxIREOMFFGILId4R98gdWK?usp=sharing

The dataset is generated from time-series Sentinel-1 SAR images in 2019 in AR,MS, MO, TN of the United States, and Cropland Data Layer (CDL) is used as the label data.

The time-series Sentinel-1 SAR images is preprocessed and downloaded by Google Earth Engine and the linke of the code can be found below:
https://code.earthengine.google.com/49f8e2532075272a79883ad8fbf41ccb

Download two compressed files named 'src' and 'label' to your local computer and unzip them to the same directory.
![Image text](https://github.com/younglimpo/TFBSmodel/blob/master/Img/dataset.png)

Each image tile in the src folder contains 18 channels with a spatial size of 128 × 128.


## User Guide

Open TFBSmodel.py, change the 'filepath' parameter to the directory where the src and label folers is.
Configure the BS (batch size) according to the memory size of the vedio card on your compurter.
The specific parameters of the TFBS model can be found in the 'TFBS()' function.
The result of the cross-validation could be found in the 'logs' folder and opened by tensorboard

 ```python
 if __name__ == '__main__':
    # directory where the src and label folders is
    filepath = 'E:/TFBS/Dataset/'
    EPOCHS = 30 # epoch of training
    BS = 16 # batch size

    # 10-fold cross validation
    CrossValidation()
```

 ## Results
 ![Loss](https://github.com/younglimpo/TFBSmodel/blob/master/Img/loss.png) ![Overall accuracy](https://github.com/younglimpo/TFBSmodel/blob/master/Img/OA.png) ![F-score](https://github.com/younglimpo/TFBSmodel/blob/master/Img/f-score.png)![Kappa](https://github.com/younglimpo/TFBSmodel/blob/master/Img/kappa.png) ![Recall](https://github.com/younglimpo/TFBSmodel/blob/master/Img/recall.png) ![Precision](https://github.com/younglimpo/TFBSmodel/blob/master/Img/precision.png)
 
