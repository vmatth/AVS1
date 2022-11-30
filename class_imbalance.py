import cv2
import os
import glob
import sys
import numpy as np
from tqdm import tqdm
from course_rating.get_classes import get_class_coords

PATH = 'C:\\Users\\jespe\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\2. segmentation masks\\'


def load_images_from_folder(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        if img is not None:
            images.append(img)
        # return images
    return images

def main():

    #images = [open(file, "rb") for file in glob.glob(u'C:\\Users\\jespe\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\2. segmentation masks\\*.png')]
    #images = [open(file, "rb") for file in glob.glob(PATH+'*.png')]
    #Loop all images
    counter = 1
    images = load_images_from_folder(PATH)
    print("Amount of images: ", len(images))
    tee_list = []
    water_list = []
    fairway_list = []
    green_list = []
    bunker_list = []
    inst_tee_list = []
    inst_water_list = []
    inst_fairway_list = []
    inst_green_list = []
    inst_bunker_list = []

    for img in tqdm(images):
        print("Getting class data for image: ", counter)

        tee, water, fairway, green, bunker = get_class_coords(img, colors='cvat')

        tee_pixels = np.sum(tee == 255)
        water_pixels = np.sum(water == 255)
        fairway_pixels = np.sum(fairway == 255)
        green_pixels = np.sum(green == 255)
        bunker_pixels = np.sum(bunker == 255)

        # print("Amount of pixels of each class: ")
        # print("Tees: ", tee_pixels)
        # print("Water: ", water_pixels)
        # print("Fairway: ", fairway_pixels)
        # print("Green: ", green_pixels)
        # print("Bunker: ", bunker_pixels)
        tee_list.append(tee_pixels)
        water_list.append(water_pixels)
        fairway_list.append(fairway_pixels)
        green_list.append(green_pixels)
        bunker_list.append(bunker_pixels)


        # Find contours
        tee_contours, hierarchy = cv2.findContours(tee, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inst_tee_list.append(len(tee_contours))
        
        water_contours, hierarchy = cv2.findContours(water, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inst_water_list.append(len(water_contours))
        
        fairway_contours, hierarchy = cv2.findContours(fairway, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inst_fairway_list.append(len(fairway_contours))
        
        green_contours, hierarchy = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inst_green_list.append(len(green_contours))
        
        bunker_contours, hierarchy = cv2.findContours(bunker, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inst_bunker_list.append(len(bunker_contours))
        #cv2.drawContours(img, tee_contours, -1, (0,255,0), 1)
        #print("Tee instances: ", len(tee_contours))
        



        # cv2.imshow('original', img)
        # cv2.imshow('tee', tee)
        # cv2.imshow('water', water)
        # cv2.imshow('fair', fairway)
        # cv2.imshow('green', green)
        # cv2.imshow('bunker', bunker)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        counter = counter + 1
    
    num_px_tee = sum(tee_list)
    num_px_water = sum(water_list)
    num_px_fairway = sum(fairway_list)
    num_px_green = sum(green_list)
    num_px_bunker = sum(bunker_list)

    print("num px tee: ", num_px_tee)
    print("num px water: ", num_px_water)
    print("num px fairway: ", num_px_fairway)
    print("num px green: ", num_px_green)
    print("num px bunker: ", num_px_bunker)
        
    num_inst_tee = sum(inst_tee_list)
    num_inst_water = sum(inst_water_list)
    num_inst_fairway = sum(inst_fairway_list)
    num_inst_green = sum(inst_green_list)
    num_inst_bunker = sum(inst_bunker_list)

    print("num instances tees: ", num_inst_tee)
    print("num instances waters: ", num_inst_water)
    print("num instances fairways: ", num_inst_fairway)
    print("num instances greens: ", num_inst_green)
    print("num instances bunkers: ", num_inst_bunker)




if __name__ == "__main__":
    print("Collecting data for each class")
    main()


