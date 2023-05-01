from tkinter import *
from tkinter import filedialog
import cv2
from PIL import Image
from PIL import ImageTk
import numpy as np
from get_classes import get_class_coords as gcc
import customtkinter as ctk

def select_image():
    path = filedialog.askopenfilename()
    if len(path) > 0:
        print("Opening image in: ", path)

        if('_prediction' in path):
            pred_image = path
            rgb_image = pred_image[:-14] + "image.png"
            print("Found corresponding RGB image in: ", rgb_image)
        elif('_image' in path):
            rgb_image = path
            pred_image = rgb_image[:-9] + "prediction.png"  
            print("Found corresponding prediction image in: ", pred_image)     
        else:
            raise ValueError("Could not find _prediction or _image in filename")


        rgb_image = cv2.imread(rgb_image)
        pred_image = cv2.imread(pred_image)
  
        # Convert to PhotoImage so Tkinter can display it
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) 
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB) 
   
        PIL_image = Image.fromarray(rgb_image)
        PIL_image = ImageTk.PhotoImage(PIL_image)

        return PIL_image, rgb_image, pred_image #Returns the PIL image and the cv image
    raise ValueError("Could not open the selected image.")
