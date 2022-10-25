import splitfolders
input_folder = "C:\\Users\\Vini\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\2. segmentation masks"

# This file splits images in a given folder to a 7/3 train/validation split in an output folder.

#Train, val, test
splitfolders.ratio(input_folder, output="C:\\Users\\Vini\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\1. 1000\\train_masks", 
                    seed=42, ratio=(.7, .3), 
                    group_prefix=None) 