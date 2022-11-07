import cv2
import numpy as np

# callback function for cv2.setMouseCallback()
# params[0]: image
# params[1]: coordinates - empty list
def click_event(event, x, y, flags, params):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(params[0], (x, y), 5, (0, 0, 255), -1)
        params[1].append([x, y])

# click on picture to get point coords, until ESC
def get_click_coords(image, coords):
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event, [image, coords])

    while(1):
        cv2.imshow('image', image)
        k = cv2.waitKey(20) & 0xFF
            
        if k == 27:
            break

    #convert coords list --> np.array, later easier to make calculations
    return np.asarray(coords)