import glob
import cv2
import numpy as np
from natsort import natsorted

def load_images(path):
    cv_img = []
    for file in natsorted(glob.glob(path)):
        if "__" in file:
            cv_img.append(file)
            
        elif "prediction" in file:
            cv_img.append(file)
        
    for i in range(0, len(cv_img), 2):
        original = cv2.imread(cv_img[i])
        prediction = cv2.imread(cv_img[i+1])

    return original, prediction
