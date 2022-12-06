from tkinter import *
from tkinter import filedialog
import cv2
from PIL import Image
from PIL import ImageTk
import model_utils
import numpy as np
from get_classes import get_class_coords as gcc

def select_image(model):
    path = filedialog.askopenfilename()

    if len(path) > 0:
        image = cv2.imread(path)

        # Prediction image using the model
        prediction = model_utils.predict(path, model)
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)     
        
        # Resize image to be the same as the prediction
        image = cv2.resize(image, (model_utils.IMAGE_WIDTH, model_utils.IMAGE_HEIGHT))
        
        # Colors for each class
        colors = [[0, 36, 250], [231, 200, 46], [77 ,156, 77], [122, 243, 142], [158, 246, 246]]
        masks = gcc(prediction)

        # Draw contours for each class
        for i, m in enumerate(masks):
            if np.sum(m) > 0:
                c, _  = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(image, c, -1, colors[i], 3)    
        
        # Convert to PhotoImage so Tkinter can display it
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        cv_image = image     
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        return image, cv_image, prediction #Returns the PIL image and the cv image
    raise ValueError("Could not open the selected image.")
