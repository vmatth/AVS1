# AVS1 

This repository contains our implementation of the UNet architecture for semantic segmentation in golf courses.

## Files

dataset.py handles all the loading of data.

model.py contains the UNet architecture.

train.py trains our model on the data stored in data/

utils.py is used for checking accuracy, saving prediction images and load/saving checkpoints.

The folder 'file_management' contains python files for managing our images and masks.

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
