import cv2
import numpy as np

def get_class_coords(image, colors='unet'):
    #r g b -> b g r
    #colors of each class
    if colors == 'cvat':
        colors_bgr = {
            'fairway': np.array([77, 156, 77]), #dark green
            'green': np.array([122, 243, 142]), #light green
            'tee': np.array([0, 36, 250]), #red
            'bunker': np.array([158, 246, 246]), #yellow
            'water': np.array([231, 200, 46]) #blue
        }
    else: 
        colors_bgr = {
            'fairway': np.array([0, 140, 0]), #dark green
            'green': np.array([0, 255, 0]), #light green
            'tee': np.array([0, 0, 255]), #red
            'bunker': np.array([122, 230, 217]), #yellow
            'water': np.array([247, 15, 7]) #blue
        }
    
    #mask for each class
    mask = {}
    for item in colors_bgr:
        mask.update({item : cv2.inRange(image, colors_bgr[item], colors_bgr[item])})
    
    return [mask['tee'], mask['water'], mask['fairway'], mask['green'], mask['bunker']]
