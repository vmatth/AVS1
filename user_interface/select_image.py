from tkinter import *
from tkinter import filedialog
import cv2
from PIL import Image
from PIL import ImageTk
import model_utils
import numpy as np
from get_classes import get_class_coords as gcc

def select_image(panelA, model):
    #global panelA

    path = filedialog.askopenfilename()

    if len(path) > 0:
        image = cv2.imread(path)

        prediction = model_utils.predict(path, model)
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)     
        
        image = cv2.resize(image, (model_utils.IMAGE_WIDTH, model_utils.IMAGE_HEIGHT))
        
        colors = [[0, 36, 250], [231, 200, 46], [77 ,156, 77], [122, 243, 142], [158, 246, 246]]
        masks = gcc(prediction)
 
        for i, m in enumerate(masks):
            if np.sum(m) > 0:
                c, _  = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(image, c, -1, colors[i], 3)    
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

    if panelA is None:
        panelA = Label(image=image)
        panelA.image = image
        panelA.pack(side="left", padx=10, pady=10)
    else:
        panelA.configure(image=image)
        panelA.image = image
        panelA.pack(side="left", padx=10, pady=10)