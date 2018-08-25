# VDSR-Tensorflow (2018/08/14)

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
```shell
python main.py

# if you want to change training epoch ex) 80 epoch (default) -> 120 epoch
python main.py --training_epoch 120
```

### Test
```shell
python demo.py

# default args: image_index = 1, scale = 2, coordinate = [50,50], interval = 30 
# you can change args: image_index = 13, scale = 4, coorindate [100,100], interval = 50

python demo.py --image_index 13 --scale 4 --coordinate [100,100] --interval 50
```

## Result
##### Results on Set 5

|  Scale    | Bicubic | tf_SRCNN | tf_VDSR |
|:---------:|:-------:|:----:|:----:|
| 2x - PSNR|   33.33 |   36.70 |   37.10 |

##### Results on Urban 100 (visual)
- Original (Urban100 / index 1)

  ![Imgur](https://github.com/DevKiHyun/VDSR-Tensorflow/blob/master/result/original.png)
 
 - Bicubic (Urban100 / index 1)

    ![Imgur](https://github.com/DevKiHyun/VDSR-Tensorflow/blob/master/result/bicubic.png)
 
 - VDSR (Urban100 / index 1)
 
    ![Imgur](https://github.com/DevKiHyun/VDSR-Tensorflow/blob/master/result/VDSR.png)
