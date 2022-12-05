# AVS1 

This repository contains implementation for our paper "Semantic Segmentation of Golf Courses for Course Rating Assistance"
## Folder Structure

"unet" contains code for training and validating unet models. 
    --"train.py" trains the model on 1.123 training/validation images based on the given parameters.
    --"utils.py" handles various functions such as loading images, loading models, checking accuracy for validation/testing etc.
    --"test.py" tests the model on 93 test images and outputs the accuracy for each class.

"course_rating" handles 

## Tensorboard
Tensorboard is used to visualize the train loss and validation loss.
It can be opened using the following command:
```
tensorboard --logdir=tensorboard
```

## Setting up Strato
```
sudo apt install python3.9
```
```
sudo apt install python3-pip
```
```
pip install numpy
```
```
pip3 install torch torchvision
```
```
pip install -U albumentations
```
```
pip install segmentation-models-pytorch
```
```
pip install tensorboard
```

If you need to unzip the image folders you can use:

https://iq.direct/blog/49-how-to-unzip-file-on-ubuntu-linux.html

GPU-support for strato (bottom of page)
https://www.strato-docs.claaudia.aau.dk/guides/Image-guides/ubuntu/ubuntu_20-04/

Run the train.py file
```
/usr/bin/python3 /home/ubuntu/AVS1/train.py
```
