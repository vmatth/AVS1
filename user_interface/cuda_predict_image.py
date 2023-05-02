##########################
#
# This file outputs a prediction image using the model.
# Intended for cloud services such as strato that does not have a user interface.
# The user_interface files can then read the prediction image file to visualize the data using Tkinter.
#
##########################
import os
import model_utils as mu
import cv2
import numpy as np
from get_classes import get_class_coords as gcc

IMAGES_DIR = os.path.expanduser('~') + "/roldskov/" #The folder with all images to predict
OUTPUT_DIR = os.path.expanduser('~') + "/prediction_images_output/" #Output folder where predicted images are saved

#Runs the model on the image (using CUDA)
#Saves images to output_path
#Saves two images: 
# The rgb image with the predicted classes as outlines (contours)
# The class masks
def predict_image_cuda(model, img_name, path, output_path):
    """
    Predicts the golf features from an image using the model (with CUDA)
    
    Saves images to output_path

    Saves two images: 
    The rgb image with the predicted classes as outlines (contours) "name_image.png"
    The class masks  "name_prediction.png"

    """
    if len(path) > 0:
        print("Loading image: ", path)
        image = cv2.imread(path)
        # Prediction image using the model
        prediction = mu.predict(path, model)
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)     
        # Resize image to be the same as the prediction
        image = cv2.resize(image, (mu.IMAGE_WIDTH, mu.IMAGE_HEIGHT))
        
        # Colors for each class
        colors = [[0, 36, 250], [231, 200, 46], [77 ,156, 77], [122, 243, 142], [158, 246, 246]]
        masks = gcc(prediction)

        # Draw contours for each class
        for i, m in enumerate(masks):
            if np.sum(m) > 0:
                c, _  = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(image, c, -1, colors[i], 3)

        img_name = img_name[:-4] #Remove .jpg

        new_img_dir = output_path + img_name + "_image.png"
        new_prediction_dir = output_path + img_name + "_prediction.png"

        cv2.imwrite(new_img_dir, image)
        cv2.imwrite(new_prediction_dir, prediction)
        print("Succesfully saved images to: \n ", new_img_dir, " \n ", new_prediction_dir)
        return 0
    raise ValueError("Could not open the selected image.")

def main():
    print("-----Predicting Image with CUDA-----")
    model = mu.load_model()

    images = os.listdir(IMAGES_DIR)
    for file in images:
        print("----------Predicting: ", file, "----------")
        IMAGE_DIR = os.path.join(IMAGES_DIR, file)
        predict_image_cuda(model, file, IMAGE_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()