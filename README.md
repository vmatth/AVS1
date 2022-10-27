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
