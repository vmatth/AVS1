import torch
import torchvision
from dataset import GolfDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import segmentation_models_pytorch as smp
import tqdm

# BATCH_SIZE = 1 #CAN BE CHANGED TO 32
# BATCH_SIZE = get_loaders()

class save_best_model:
    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.best_valid_loss_epoch = float("inf")

    def __call__(self, current_valid_loss, epoch, model, optimizer, loss_fn):
        print(f"Current Best Validation Loss: ({self.best_valid_loss})", f"at epoch [{self.best_valid_loss_epoch}]")
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_valid_loss_epoch = epoch
            print(f"New Best Validation Loss: ({self.best_valid_loss})", f"at epoch [{self.best_valid_loss_epoch}]")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimer_state_dict": optimizer.state_dict(),
                "loss": loss_fn,
                }, "saved_models/best_model.pth.tar")


def save_checkpoint(state, filename):
    try:
        print("Saving Checkpoint")
        torch.save(state, filename)
    except:
        print("Saving Checkpoint Failed")


def load_checkpoint(PATH, model, optimizer):
    try:
        print("Loading checkpoint")
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_epoch = checkpoint['epoch']
        print("     Successfully Loaded at Epoch: ", loaded_epoch)
        model.eval()
        return model, optimizer, loaded_epoch
    except:
        print("     Loading Checkpoint Failed")
        return model, optimizer, 0


def get_loaders(  
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers,
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


# Calculates the Intersection over Union of prediction (tags) compared to the ground truth (mask)
def calc_IoU(tags, mask):
    #IoU =   True Positives               / (True Positives              +  False Positive              + False Negative )
    IoU_fairway = (torch.sum(tags[mask == 1] == 1) / (torch.sum(tags[mask == 1] == 1) +  torch.sum(tags[mask != 1] == 1) + torch.sum(tags[mask == 1] != 1))*100).item()
    IoU_green = (torch.sum(tags[mask == 2] == 2) / (torch.sum(tags[mask == 2] == 2) +  torch.sum(tags[mask != 2] == 2) + torch.sum(tags[mask == 2] != 2))*100).item()
    IoU_tee = (torch.sum(tags[mask == 3] == 3) / (torch.sum(tags[mask == 3] == 3) +  torch.sum(tags[mask != 3] == 3) + torch.sum(tags[mask == 3] != 3))*100).item()
    IoU_bunker = (torch.sum(tags[mask == 4] == 4) / (torch.sum(tags[mask == 4] == 4) +  torch.sum(tags[mask != 4] == 4) + torch.sum(tags[mask == 4] != 4))*100).item()
    IoU_water = (torch.sum(tags[mask == 5] == 5) / (torch.sum(tags[mask == 5] == 5) +  torch.sum(tags[mask != 5] == 5) + torch.sum(tags[mask == 5] != 5))*100).item()

    #In specific cases where ground truth does not have a class that is predicted we get a NaN error as we try to divide by 0
    if math.isnan(IoU_fairway): IoU_fairway = 0
    if math.isnan(IoU_green): IoU_green = 0
    if math.isnan(IoU_tee): IoU_tee = 0
    if math.isnan(IoU_bunker): IoU_bunker = 0
    if math.isnan(IoU_water): IoU_water = 0

    return IoU_fairway, IoU_green, IoU_tee, IoU_bunker, IoU_water
# Sensitivity also known as Recall
def calc_sensitivity(tags, mask):    
    #Sens =          True Positives               / (True Positives                                + False Negative )
    sens_fairway = (torch.sum(tags[mask == 1] == 1) / (torch.sum(tags[mask == 1] == 1)  + torch.sum(tags[mask == 1] != 1))*100).item()
    sens_green = (torch.sum(tags[mask == 2] == 2) / (torch.sum(tags[mask == 2] == 2) + torch.sum(tags[mask == 2] != 2))*100).item()
    sens_tee = (torch.sum(tags[mask == 3] == 3) / (torch.sum(tags[mask == 3] == 3) + torch.sum(tags[mask == 3] != 3))*100).item()
    sens_bunker = (torch.sum(tags[mask == 4] == 4) / (torch.sum(tags[mask == 4] == 4) + torch.sum(tags[mask == 4] != 4))*100).item()
    sens_water = (torch.sum(tags[mask == 5] == 5) / (torch.sum(tags[mask == 5] == 5) + torch.sum(tags[mask == 5] != 5))*100).item()

    #In specific cases where ground truth does not have a class that is predicted we get a NaN error as we try to divide by 0
    if math.isnan(sens_fairway): sens_fairway = 0
    if math.isnan(sens_green): sens_green = 0
    if math.isnan(sens_tee): sens_tee = 0
    if math.isnan(sens_bunker): sens_bunker = 0
    if math.isnan(sens_water): sens_water = 0

    return sens_fairway, sens_green, sens_tee, sens_bunker, sens_water

# Positive Predictive Value also known as precision
def PPV(tags, mask):
    # TP / TP + FP
    
    pass


def check_accuracy(loader, model, epoch, loss_fn, writer, device="cuda"):
    print("-----Calculating Accuracy-----")
    model.eval()

    IoU_fairways, IoU_greens, IoU_tees, IoU_bunkers, IoU_waters = ([] for _ in range(5))   
    sens_fairways, sens_greens, sens_tees, sens_bunkers, sens_waters = ([] for _ in range(5))

    losses = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)

            #Calculate loss here
            loss = loss_fn(preds, y.long())
            losses.append(loss.item())
            _, tags = torch.max(preds, dim = 1)
            
            #Calculate IoU, Sensitivity and
            IoU_fairway, IoU_green, IoU_tee, IoU_bunker, IoU_water = calc_IoU(tags, y)
            sens_fairway, sens_green, sens_tee, sens_bunker, sens_water = calc_sensitivity(tags, y)

            #Append to list to calculate the mean of a list using np.mean
            IoU_fairways.append(IoU_fairway)
            IoU_greens.append(IoU_green)
            IoU_tees.append(IoU_tee)
            IoU_bunkers.append(IoU_bunker)
            IoU_waters.append(IoU_water)
            
            sens_fairways.append(sens_fairway)
            sens_greens.append(sens_green)
            sens_tees.append(sens_tee)
            sens_bunkers.append(sens_bunker)
            sens_waters.append(sens_water)

        mean_IoU_fairways = np.mean(IoU_fairways)
        mean_IoU_greens = np.mean(IoU_greens)
        mean_IoU_tees = np.mean(IoU_tees)
        mean_IoU_bunkers = np.mean(IoU_bunkers)
        mean_IoU_waters = np.mean(IoU_waters)

        mean_sens_fairways = np.mean(sens_fairways)
        mean_sens_greens = np.mean(sens_greens)
        mean_sens_tees = np.mean(sens_tees)
        mean_sens_bunkers = np.mean(sens_bunkers)
        mean_sens_waters = np.mean(sens_waters)
        
        print("     Fairway IoU: ", mean_IoU_fairways, "%")
        print("     Green IoU: ", mean_IoU_greens, "%")
        print("     Tee IoU: ", mean_IoU_tees, "%")
        print("     Bunker IoU: ", mean_IoU_bunkers, "%")
        print("     Water IoU: ", mean_IoU_waters, "%")

        print("         Fairway Sensitivity: ", mean_sens_fairways, "%")
        print("         Green Sensitivity: ", mean_sens_greens, "%")
        print("         Tee Sensitivity: ", mean_sens_tees, "%")
        print("         Bunker Sensitivity: ", mean_sens_bunkers, "%")
        print("         Water Sensitivity: ", mean_sens_waters, "%")

        mean_loss = np.mean(losses)
        print("Validation Loss: ", mean_loss)
        writer.add_scalar("Loss/val", mean_loss, epoch)
        writer.close()

    model.train()
    return mean_loss

def save_predictions_as_imgs(
    loader, model, batch_size, folder="data/saved_images/", device="cuda"
):
    print("Saving validation images")
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
        output = torch.zeros(preds.shape[0], 3, preds.size(-2), preds.size(-1), dtype=torch.float,  device='cuda') #Output size is set to preds.shape[0] as the size automatically changes to fit the remaining batch_size.
        for class_idx, color in enumerate(class_to_color):
            mask = preds[:,class_idx,:,:] == torch.max(preds, dim=1)[0]
            mask = mask.unsqueeze(1)
            curr_color = color.reshape(1, 3, 1, 1)
            segment = mask*curr_color 
            output += segment
        

        y_output = torch.zeros(y.shape[0], 3, preds.size(-2), preds.size(-1), dtype=torch.float,  device='cuda')
        for class_idx, color in enumerate(class_to_color):
            mask = y[:,:,:] == class_idx
            mask = mask.unsqueeze(1)
            #print("mask shape", mask.shape)
            curr_color = color.reshape(1, 3, 1, 1)
            segment = mask*curr_color 
            y_output += segment


        #Save images to our saved_images folder
        torchvision.utils.save_image(output, f"{folder}/{idx+1}_prediction.png")
        torchvision.utils.save_image(y_output, f"{folder}/{idx+1}_groundtruth.png")

    model.train()







    
# %%
