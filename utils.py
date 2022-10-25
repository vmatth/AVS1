from cmath import isnan
import torch
import torchvision
from dataset import GolfDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import math

BATCH_SIZE = 1 #CAN BE CHANGED TO 32


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    #print("=> Saving checkpoint :)")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint :)")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(  
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):

    train_ds = GolfDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = GolfDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    print("Calculating Accuracy")
    model.eval()

    IoU_fairways = []
    IoU_greens = []
    IoU_tees = []
    IoU_bunkers = []
    IoU_waters = []   

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)

            _, tags = torch.max(preds, dim = 1)

            #Calculate accuracy of each class using TP FP and FN
            #Classes | 0: Background | 1: Fairway | 2: Green | 3: Tees | 4: Bunkers | 5: Water |

            #Accuracy =   True Positives               / (True Positives              +  False Positive              + False Negative 
            IoU_fairway = (torch.sum(tags[y == 1] == 1) / (torch.sum(tags[y == 1] == 1) +  torch.sum(tags[y != 1] == 1) + torch.sum(tags[y == 1] != 1))*100).item()
            IoU_green = (torch.sum(tags[y == 2] == 2) / (torch.sum(tags[y == 2] == 2) +  torch.sum(tags[y != 2] == 2) + torch.sum(tags[y == 2] != 2))*100).item()
            IoU_tee = (torch.sum(tags[y == 3] == 3) / (torch.sum(tags[y == 3] == 3) +  torch.sum(tags[y != 3] == 3) + torch.sum(tags[y == 3] != 3))*100).item()
            IoU_bunker = (torch.sum(tags[y == 4] == 4) / (torch.sum(tags[y == 4] == 4) +  torch.sum(tags[y != 4] == 4) + torch.sum(tags[y == 4] != 4))*100).item()
            IoU_water = (torch.sum(tags[y == 5] == 5) / (torch.sum(tags[y == 5] == 5) +  torch.sum(tags[y != 5] == 5) + torch.sum(tags[y == 5] != 5))*100).item()

            #In specific cases where ground truth does not have a class that is predicted we get a NaN error as we try to divide by 0
            if math.isnan(IoU_fairway): IoU_fairway = 0
            if math.isnan(IoU_green): IoU_green = 0
            if math.isnan(IoU_tee): IoU_tee = 0
            if math.isnan(IoU_bunker): IoU_bunker = 0
            if math.isnan(IoU_water): IoU_water = 0

            #Append to list to calculate the mean of a list using np.mean
            IoU_fairways.append(IoU_fairway)
            IoU_greens.append(IoU_green)
            IoU_tees.append(IoU_tee)
            IoU_bunkers.append(IoU_bunker)
            IoU_waters.append(IoU_water)
            
        mean_IoU_fairways = np.mean(IoU_fairways)
        mean_IoU_greens = np.mean(IoU_greens)
        mean_IoU_tees = np.mean(IoU_tees)
        mean_IoU_bunkers = np.mean(IoU_bunkers)
        mean_IoU_waters = np.mean(IoU_waters)
        
        print("Fairway accuracy: ", mean_IoU_fairways, "%")
        print("Green accuracy: ", mean_IoU_greens, "%")
        print("Tee accuracy: ", mean_IoU_tees, "%")
        print("Bunker accuracy: ", mean_IoU_bunkers, "%")
        print("Water accuracy: ", mean_IoU_waters, "%")

    model.train()

def save_predictions_as_imgs(
    loader, model, folder="data/saved_images/", device="cuda"
):
    print("Validating model")
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = model(x)
            # print("preds: ", preds)
            # print("preds[]: ", preds[0])
            # print("preds[][]: ", preds[0][0])
            # preds = torch.sigmoid(model(x))   <- This is used for binary images
            # preds = (preds > 0.5).float()
        #Classes | 0: Background | 1: Fairway | 2: Green | 3: Tees | 4: Bunkers | 5: Water |
        class_to_color = [torch.tensor([0.0, 0.0, 0.0], device='cuda'), torch.tensor([0.0, 140.0/255, 0.0],  device='cuda'), torch.tensor([0.0, 255.0/255, 0.0],  device='cuda'), torch.tensor([255.0/255, 0.0, 0.0],  device='cuda'), torch.tensor([217.0/255, 230.0/255, 122.0/255],  device='cuda'), torch.tensor([7.0/255, 15.0/255, 247.0/255],  device='cuda')]
        output = torch.zeros(BATCH_SIZE, 3, preds.size(-2), preds.size(-1), dtype=torch.float,  device='cuda')
        for class_idx, color in enumerate(class_to_color):
            mask = preds[:,class_idx,:,:] == torch.max(preds, dim=1)[0]
            mask = mask.unsqueeze(1)
            curr_color = color.reshape(1, 3, 1, 1)
            segment = mask*curr_color 
            output += segment

        y_output = torch.zeros(BATCH_SIZE, 3, preds.size(-2), preds.size(-1), dtype=torch.float,  device='cuda')
        for class_idx, color in enumerate(class_to_color):
            mask = y[:,:,:] == class_idx
            mask = mask.unsqueeze(1)
            #print("mask shape", mask.shape)
            curr_color = color.reshape(1, 3, 1, 1)
            segment = mask*curr_color 
            y_output += segment


        torchvision.utils.save_image(
            output, f"{folder}/Prediction_{idx+1}.png"
        )
        torchvision.utils.save_image(y_output, f"{folder}/Ground_Truth_{idx+1}.png")

    model.train()







    