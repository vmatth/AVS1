import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
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
BATCH_SIZE = 1 #CAN BE CHANGED TO 32
NUM_EPOCHS = 3 #CAN BE 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 80  # 900 originally
IMAGE_WIDTH = 160    # 1600 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
SAVED_IMG_DIR = "data/saved_images/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    print("Train function")
    loop = tqdm(loader)

    # Data size is [batches, in_channels, image_height, image_width]
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        #targets = targets.reshape(BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            print("predictions shape", predictions.shape)
            print("targets", targets.shape)
            loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    print("Preparing to train data")
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



    model = UNET(in_channels=3, out_channels=6).to(DEVICE)  # change out_channels according to numbers of classes 
    loss_fn = nn.CrossEntropyLoss()
    #nn.BCEWithLogitsLoss() # if we have more than one class, change loss function to cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        print("Epoch: ", epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        #Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),

        }

        save_checkpoint(checkpoint)


        #Check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        #Print examples to folder
        save_predictions_as_imgs(
            val_loader, model, folder=SAVED_IMG_DIR, device=DEVICE
        )


if __name__ == "__main__":
    main()