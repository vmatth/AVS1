import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# This class handles loading the golf dataset
class GolfDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir 
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    # This function loads all of the train images and train masks from their respective folder
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        #mask_path = os.path.join(self.mask_dir, self.images[index]) #optional is to .replace(".jpg, "_mask.gif")
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        image = np.array(Image.open(img_path).convert("RGB")) 
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #L for Grayscale, maybe use RGB instead?
        mask[mask == 255.0] = 1.0 #Changes the white colors (255) to 1.0 to use a sigmoid function - this can change depending on our colors

        if(self.transform is not None):
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask