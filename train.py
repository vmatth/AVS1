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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Settings for the image
BATCH_SIZE = 16 
NUM_EPOCHS = 20
NUM_WORKERS = 4 
IMAGE_HEIGHT =  256 # 900 originally
IMAGE_WIDTH = 416   # 1600 originally
PIN_MEMORY = True
# Load/Save Settings
LOAD_MODEL = False
LOAD_CHECKPOINT = False
SAVE_CHECKPOINT = True
# UNet Model transfer learning 
ACTIVATION = "softmax2d"
ENCODER_NAME = "resnet152"
ENCODER_WEIGHTS="imagenet"
# Directories
TRAIN_IMG_DIR = "/home/ubuntu/project/data/train_images"
TRAIN_MASK_DIR = "../data/train_masks/"
VAL_IMG_DIR = "../data/val_images/"
VAL_MASK_DIR = "../data/val_masks/"
SAVED_IMG_DIR = "../data/saved_images/"
CHECKPOINT_DIR = "saved_models/my_checkpoint.pth.tar"

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
    print("     Load Checkpoint: ", LOAD_CHECKPOINT)

    # Create instance for save best model class
    best_model = save_best_model()
    
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


    # Create the UNet model
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=6,
        activation=ACTIVATION,
    ).to(DEVICE)

    #model = UNET(in_channels=3, out_channels=6).to(DEVICE)  # change out_channels according to numbers of classes 

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss() #CrossEntropyLoss for multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #THE OPTIMIZER CAN BE CHANGED

    loaded_epoch = 0
    # Load a UNet model checkpoint
    if LOAD_CHECKPOINT:
        model, optimizer, loaded_epoch = load_checkpoint(CHECKPOINT_DIR, model, optimizer)
        model.train() 

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

    # Setup tensorboard for visualizing the train loss and validation loss
    writer = SummaryWriter('tensorboard/')
    
    scaler = torch.cuda.amp.GradScaler()
    # Loop all epochs and train
    for epoch in range(loaded_epoch, NUM_EPOCHS):
        print("-------------------------------------")
        print("Epoch: ", epoch)
        # Train the model for this epoch and return the loss for that epoch
        mean_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        #Add loss to tensorboard for visualization
        writer.add_scalar('Loss/train', mean_loss, epoch)
        writer.close()
        #Check accuracy of the model using the validation data
        current_val_loss = check_accuracy(val_loader, model, epoch, loss_fn, writer, device=DEVICE)

        #Save model if validation loss is best
        best_model(current_val_loss, epoch, model, optimizer, loss_fn)
    
        #Save validation ground truth and prediction
        save_predictions_as_imgs(val_loader, model, BATCH_SIZE, folder=SAVED_IMG_DIR, device=DEVICE)

        #Save the model as a checkpoint in case of inteference
        if SAVE_CHECKPOINT:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                'epoch': epoch + 1, #Save the next epoch, so the model resumes training at the next one.
            }
            save_checkpoint(checkpoint, CHECKPOINT_DIR)



if __name__ == "__main__":
    main()