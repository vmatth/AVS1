import cv2
import numpy as np

def get_class_coords(image):
    #r g b -> b g r
    #colors of each class
    colors_rgb= {
        'fairway': np.array([0, 140, 0]), #dark green
        'green': np.array([0, 255, 0]), #light green
        'tee': np.array([0, 0, 255]), #red
        'bunker': np.array([122, 230, 217]), #yellow
        'water': np.array([247, 15, 247]) #blue
    }
    
    #mask for each class
    mask = {}
    for item in colors_rgb:
        mask.update({item : cv2.inRange(image, colors_rgb[item], colors_rgb[item])})
    
    return mask['tee'], mask['water'], mask['fairway'], mask['green'], mask['bunker']
