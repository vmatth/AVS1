#This python file calculates the size of a given class in an image.
#This is done by calculating the amount of pixels in the blob
import cv2
import numpy as np
from get_classes import get_class_coords
import convert
from distance import distance_two_points
from operator import itemgetter
from sklearn.preprocessing import normalize
import math
import os

def midpoint(ptA, ptB):
	return (int((ptA[0] + ptB[0]) * 0.5), int((ptA[1] + ptB[1]) * 0.5))

def get_green_size(image, image_px_size, scale=1000, color='unet'):
    _, _, _, green, _ = get_class_coords(image, color)

    #The test image only contains 1 green. Thus we can calculate the size like this
    green_pxs = np.sum(green == 255)
    green_m2 = convert.convert_to_m2(image_px_size, green_pxs, scale)
  
    return green_m2

def get_bunker_size(image, image_px_size, scale=1000, color='unet'):
    _, _, _, _, bunker = get_class_coords(image)

    #The test image can contain multiple bunkers. We therefore have to calculate the size for each bunker using contours
    contours, _ = cv2.findContours(bunker, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0,255,0), 1)
    
    bunker_pxs = []
    bunker_m2 = []
    for cnt in contours:
        bunker_pxs.append(cv2.contourArea(cnt))
        bunker_m2.append(convert.convert_to_m2(image_px_size, cv2.contourArea(cnt), scale))

    return bunker_m2

# This function gets the length and width of a green.
def new_green_size(image, color='unet'):
    _, _, _, green, _ = get_class_coords(image, color)

    if np.sum(green == 255) == 0:
        print("There is no green on this image")
        return None, None
    
    contours, _ = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    shortPoint1, shortPoint2  = 0, 0
    max_dist = 0

    for cnt in contours:
        dists = []

        # Calculate the distance between the all of the points in the contour and the starting point
        for p in cnt:
            for o in cnt:
                dists.append([p[0].tolist(), o[0].tolist(), distance_two_points(p, o, image.shape)])
       
        # Max distance for one point
        max_dist = max(dists, key=itemgetter(2))

        ## Draw a diagonal blue line with thickness of 5 px
        cv2.line(image, tuple(max_dist[0]), tuple(max_dist[1]),(255,0,255),2)

        cv2.circle(image, tuple(max_dist[0]), 2, (255,0,255), -1)
        cv2.circle(image, tuple(max_dist[1]), 2, (255,0,255), -1)
        
        #quick maths kata
        mp = midpoint(max_dist[0], max_dist[1])

        #Vector for longest line
        #    B.x              A.x             B.y              A.y
        v = [max_dist[1][0] - max_dist[0][0], max_dist[1][1] - max_dist[0][1]]
        #Normalize the vector
        mag = math.sqrt(v[0]**2 + v[1]**2)
        v[0] /= mag
        v[1] /= mag
        
        #Calculate the perpendicular vector
        v = [-v[1], v[0]]
        i = 0

        #Find the intersection between this perpendicular vector and the contour
        found1, found2 = False, False
        while(1):
            newPoint = [int(mp[0] + (v[0] * i)), int(mp[1] + (v[1] * i))] #Start at the middle point and go +1 pixel in the vector's direction
            newPoint2 =  [int(mp[0] - (v[0] * i)), int(mp[1] - (v[1] * i))] #Start at the middle point and go -1 pixel in the vector's direction
            
            for p in cnt:
                if p[0][0] == newPoint[0] and p[0][1] == newPoint[1]: #Check if the newpoint is equal to any of the contours points
                    cv2.circle(image, tuple(newPoint), 2, (255,0,255), -1)
                    shortPoint1 = newPoint
                    found1 = True
                if p[0][0] == newPoint2[0] and p[0][1] == newPoint2[1]: #Check if the newpoint is equal to any of the contours points
                    cv2.circle(image, tuple(newPoint2), 2, (255,0,255), -1)
                    shortPoint2 = newPoint2
                    found2 = True
            if found1 and found2: 
                break     
            i += 0.01
        
        ## Draw a diagonal blue line with thickness of 5 px
        cv2.line(image, shortPoint1, shortPoint2,(255,0,255),2)
        
    min_dist = [shortPoint1, shortPoint2, distance_two_points(np.asarray(shortPoint1), np.asarray(shortPoint2), image.shape)]
    cv2.circle(image, mp, 3, (0,255,255), -1)

    cv2.imshow("Image with green sizes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return max_dist, min_dist



PATH = "C:\\Users\\Vini\\Desktop\\AVS1\\data\\saved_test_images\\"
def load_images_from_folder(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        if img is not None:
            images.append(img)
        # return images
    return images

for image in load_images_from_folder(PATH):
    max_dist, min_dist = new_green_size(image, color='unet')

    print(max_dist, min_dist)
