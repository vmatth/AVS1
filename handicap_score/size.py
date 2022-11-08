#This python file calculates the size of a given class in an image.
#This is done by calculating the amount of pixels in the blob
import cv2
import numpy as np
from get_classes import get_class_coords
import convert

def get_green_size(image, image_px_size, scale=1000, color='unet'):
    _, _, _, green, _ = get_class_coords(image, color)

    #The test image only contains 1 green. Thus we can calculate the size like this
    green_pxs = np.sum(green == 255)
    green_m2 = convert.convert_to_m2(image_px_size, green_pxs, scale)
  
    return green_m2

def get_bunker_size(image, image_px_size, scale=1000, color='unet'):
    _, _, _, _, bunker = get_class_coords(image)

    #The test image can contain multiple bunkers. We therefore have to calculate the size for each bunker using contours
    contours, _ = cv2.findContours(bunker, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0,255,0), 1)
    
    bunker_pxs = []
    bunker_m2 = []
    for cnt in contours:
        bunker_pxs.append(cv2.contourArea(cnt))
        bunker_m2.append(convert.convert_to_m2(image_px_size, cv2.contourArea(cnt), scale))

    return bunker_m2
