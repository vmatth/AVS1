# -*- coding: utf-8 -*-
from fileinput import close
import cv2
import os
import glob
import sys
import numpy
from tqdm import tqdm

# This file replaces specific symbols of all files in a folder.
# The specified symbols are the danish letters æ ø å as they cause trouble when loading/saving data in other files.

def main():

    images = [open(file, "rb") for file in glob.glob(u'C:\\Users\\Vini\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\Data_10_images\\*.jpg ')]

    #Loop all images
    counter = 1
    for img in tqdm(images):
        print(" Converting picture", counter)
        name = img.name
        print("oldname", name)
        name =  name.replace("æ", "ae")
        name =  name.replace("å", "aa")
        name =  name.replace("ø", "oe")
        print("newname", name)
        img.close()
        os.rename(img.name, name)
        counter = counter + 1
        




if __name__ == "__main__":
    print("Renaming files")
    main()


