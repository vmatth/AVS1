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
    get_test,
    check_accuracy,
    check_test_accuracy,
    save_predictions_as_imgs,
    save_best_model
)
from train import(
    ACTIVATION, 
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    DEVICE,
    NUM_WORKERS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,  
    PIN_MEMORY
)

BEST_M_CHECKPOINT_DIR = "C:\\Users\\Vini\\Desktop\\Mini Project\\Pre-Trained (resnet 152)\\29_best_model.pth.tar"#"./saved_models/best_model.pth.tar"
TEST_IMG_DIR = "C:\\Users\\Vini\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\test_images_trees\\" # "/home/ubuntu/project/data/test_images/"
TEST_MASK_DIR = "C:\\Users\\Vini\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\test_masks\\" #"../data/test_masks/"
TEST_RES_DIR = "data/saved_test_images" #"../data/saved_test_images"
BATCH_SIZE = 1

def load_best_checkpoint(model):
    best_checkpoint = torch.load(BEST_M_CHECKPOINT_DIR)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()

    return model

def main():
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    testloader=get_test(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        NUM_WORKERS,
        test_transform,
        PIN_MEMORY
        )
    
    print("Testing model on test data")
    
    # Transfer learning model
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=6,
        activation=ACTIVATION,
    ).to(DEVICE)

    # No transfer learning model
    #model = UNET(in_channels=3, out_channels=6).to(DEVICE)


    model = load_best_checkpoint(model)
    print("Loading model is done.")

    loss_fn = nn.CrossEntropyLoss()
    
    #Check accuracy of the model using the testing data
    #check_test_accuracy(testloader, model, loss_fn, DEVICE, True)
    save_predictions_as_imgs(testloader, model, BATCH_SIZE, TEST_RES_DIR, DEVICE, testing=True)

if __name__ == "__main__":
    main()