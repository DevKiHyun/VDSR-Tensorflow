# VDSR-Tensorflow

## Introduction
We implement a tensorflow model for ["Accurate Image Super-Resolution Using Very Deep Convolutional Networks", CVPR 16'](http://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf).
- [The author's project page](http://cv.snu.ac.kr/research/VDSR/)
- We use 291 dataset as training dataset.

## Environment
- Ubuntu 16.04
- Python 3.5

## Depenency
- Numpy
- Opencv2
- matplotlib

## Files
- main.py : Execute train.py and pass the default value.
- vdsr.py : VDSR model definition.
- train.py : Train the VDSR model and represent the test performance.
- util.py : Utility functions for this project.
- log.txt : The log of training process.
- model : The save files of the trained VDSR.

## How to use
### Training
```
python main.py

# if you want to change training epoch ex) 80 epoch (default) -> 120 epoch
python main.py --training_epoch 120
```
