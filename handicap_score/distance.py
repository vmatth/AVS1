import convert
from get_classes import get_class_coords
import numpy as np
import cv2
from operator import itemgetter

# Calculates the distance between two points
def distance_two_points(a, b, image_shape):
    px_side_size = convert.get_px_side(image_shape)
    return convert.convert_px_to_m(px_side_size, np.linalg.norm(a - b))

#Returns the distance to nearby objects (that are closer than the max_distance [m]) along with the pixel coordinates for each object.
#The output is a list for each class. E.g bunker_dists contains the distances and pixel coordinates for each bunker in the image.
# bunker_dists[i][0] will be the pixel coordinate,
# bunker_dists[i][1] will be the distance from the starting point to this coordinate
def distance_to_objects(image, point, max_distance=50, color='unet'):
    _, water, _, _, bunker = get_class_coords(image, color)

    #Find contours to get all instances of each class along with the contours points.
    water_contours, _ = cv2.findContours(water, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bunker_contours, _ = cv2.findContours(bunker, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    water_dists = []
    bunker_dists = []
    
    for cnt in bunker_contours:
        dists = []

        # Calculate the distance between the contours and the starting point
        for p in cnt:
            # Append distance and the contour point to lists
            dists.append([p[0].tolist(), distance_two_points(p, point, image.shape)])

        # Black Magic - https://stackoverflow.com/questions/16036913/minimum-of-list-of-lists
        min_dist = min(dists, key=itemgetter(1))

        if min_dist[1] < max_distance:
            bunker_dists.append(min_dist)
    
    for cnt in water_contours:
        dists = []

        # Calculate the distance between the contours and the starting point
        for p in cnt:
            # Append distance and the contour point to lists
            dists.append([p[0].tolist(), distance_two_points(p, point, image.shape)])

        # Black Magic 
        min_dist = min(dists, key=itemgetter(1))

        if min_dist[1] < max_distance:
            water_dists.append(min_dist)
            
    print(f'Bunkers: {bunker_dists}')
    print(f'Waters: {water_dists}')
    return bunker_dists, water_dists
