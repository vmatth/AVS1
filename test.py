import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_best_model
)



def main():
    print("Testing model on test data")
    # Load our model similar to utils.py/load_checkpoint. Our model will be saved_models/best_model.pth.tar
    # Calculate the accuracy using utils.py/check_accuracy  
    # (you will need to create a new loader. Currently we have train_loader and val_loader to load the train and validation images)

if __name__ == "__main__":
    main()