# -*- coding: utf-8 -*-
import cv2
import os
import glob
import sys
import numpy
from tqdm import tqdm

# This file converts segmentation masks (3 channels RGB) to a class mask (1 channel with values from 0-5 representing each class).
# The files are loaded from the folder specified.
# The files are outputted to the current location in the terminal.

PATH = 'C:\\Users\\Vini\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\2. segmentation masks\\'

def main():

    images = [open(file, "rb") for file in glob.glob(PATH+'*.png')]

    #Loop all images
    counter = 1
    for i in tqdm(images):
        print(" Converting picture", counter)
        #Do some magic to open the images as we have 'ae oe aa'
        bytes = bytearray(i.read())
        numpyarray = numpy.asarray(bytes, dtype=numpy.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Grab the image dimensions
        h = img.shape[0]
        w = img.shape[1]
        
        #Loop over the image, pixel by pixel
        for y in range(0, h):
            for x in range(0, w):
                if img[y,x] == 123: #Fairway
                    img[y,x] = 1
                elif img[y,x] == 199: #Green
                    img[y,x] = 2
                elif img[y,x] == 96: #Tees
                    img[y,x] = 3
                elif img[y,x] == 236: #Bunker
                    img[y,x] = 4
                elif img[y,x] == 157: #Water
                    img[y,x] = 5

        #Pixel values for each class (in grayscale)
        #Fairway | 123
        #Green   | 199
        #Tees    | 96
        #Water   | 157
        #Bunkers | 236

        #Save image
        img_name = i.name.replace(PATH, ' ')
        cv2.imwrite(img_name, img)

        counter = counter + 1





if __name__ == "__main__":
    print("Converting from segmentation mask to classes")
    main()


