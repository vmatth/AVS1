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
        image = np.array(Image.open(img_path).convert("RGB")) 
        if(self.transform is not None):
            augmentations = self.transform(image=image, mask=None)
            image = augmentations["image"]
        return image, None