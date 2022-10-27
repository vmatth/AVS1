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
import matplotlib.pyplot as plt
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 #CAN BE CHANGED TO 32
NUM_EPOCHS = 100 #CAN BE 100
NUM_WORKERS = 2
IMAGE_HEIGHT =  256 # 900 originally
IMAGE_WIDTH = 416    # 1600 originally
PIN_MEMORY = True
LOAD_MODEL = False
#TRAIN_IMG_DIR = "data/train_images/"
TRAIN_IMG_DIR = "C:\\Users\\Vini\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\train_images\\"
#TRAIN_MASK_DIR = "data/train_masks/"
TRAIN_MASK_DIR = "C:\\Users\\Vini\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\train_masks\\"
#VAL_IMG_DIR = "data/val_images/"
VAL_IMG_DIR = "C:\\Users\\Vini\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\val_images\\"
#VAL_MASK_DIR = "data/val_masks/"
VAL_MASK_DIR = "C:\\Users\\Vini\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\val_masks\\"
SAVED_IMG_DIR = "data/saved_images/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    losses = []

    # Data size is [batches, in_channels, image_height, image_width]
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            #print("Loss: ", loss.item())

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())
    
    mean_loss = np.mean(losses)
    print("Training Loss: ", mean_loss)
    return mean_loss




def main():
    print("Preparing to train data with the following settings")
    print("     Batch Size: ", BATCH_SIZE)
    print("     Number of Workers: ", NUM_WORKERS)
    print("     Number of Epochs: ", NUM_EPOCHS)
    print("     Image Size (wxh):", IMAGE_WIDTH, "x", IMAGE_HEIGHT)
    
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


    model = smp.Unet(
        encoder_name="resnet152",
        encoder_weights="imagenet",
        in_channels=3,
        classes=6,
        activation="softmax2d",
    ).to(DEVICE)
    
    #print(model)

    #model = UNET(in_channels=3, out_channels=6).to(DEVICE)  # change out_channels according to numbers of classes 
    loss_fn = nn.CrossEntropyLoss()
    #nn.BCEWithLogitsLoss() # if we have more than one class, change loss function to cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #THE OPTIMIZER CAN BE CHANGED

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    writer = SummaryWriter('tensorboard/')
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        print("-------------------------------------")
        print("Epoch: ", epoch + 1)
        mean_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        #Add loss to tensorboard for visualization
        writer.add_scalar('Loss/train', mean_loss, epoch)
        writer.close()

        #Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),

        }
        save_checkpoint(checkpoint)

        #Check accuracy
        check_accuracy(val_loader, model, epoch, loss_fn, writer, device=DEVICE)

        #Print examples to folder
        save_predictions_as_imgs(val_loader, model, BATCH_SIZE, folder=SAVED_IMG_DIR, device=DEVICE)


if __name__ == "__main__":
    main()