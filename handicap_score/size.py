#This file contains functions to calculate the size of a green and the size of bunkers in an image
import cv2
import numpy as np
from get_classes import get_class_coords
import convert
from distance import distance_two_points
from operator import itemgetter
import math
import os

def get_distance_to_front_and_back_green(image, landing_point, green_centerpoint, scale, color='unet'):  
    px_length_cm = convert.get_px_side(image.shape)
    v = [landing_point[0]-green_centerpoint[0], landing_point[1]-green_centerpoint[1]]
    #Normalize the vector
    mag = math.sqrt(v[0]**2 + v[1]**2)
    
    if mag != 0:
        v[0] /= mag
        v[1] /= mag

    _, _, _, green, _ = get_class_coords(image, color)
    green_contours, _ = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #get distance from landing_zone to center of green
    
    area = 0
    contour = 0
    for cnt in green_contours:
        temp_area = cv2.contourArea(cnt)
        if temp_area > area:
            area = temp_area
            contour = cnt
    min_dist = get_min_dist_cnt(contour, green_centerpoint, v, image, scale)
    min_dist.pop()
    distance_front_green = convert.convert_px_to_m(px_length_cm, np.linalg.norm(landing_point-min_dist[0]), scale)
    distance_back_green = convert.convert_px_to_m(px_length_cm, np.linalg.norm(landing_point-min_dist[1]), scale)
    return distance_front_green,distance_back_green

def midpoint(ptA, ptB):
	return (int((ptA[0] + ptB[0]) * 0.5), int((ptA[1] + ptB[1]) * 0.5))

# returns [[p1 p2], max_distance] of ONE contour (to find the length of a green)
def get_max_dist_cnt(cnt, image_shape, scale):
    d = []

    for p1 in cnt:
        for p2 in cnt:
            d.append([p1[0].tolist(), p2[0].tolist(), distance_two_points(p1, p2, image_shape, scale)])
        
        # Max distance for one point
    max_d = max(d, key=itemgetter(2))
    return max_d

# returns [p1, p2, dist] that defines the minimum width of a green.
def get_min_dist_cnt(cnt, mp, v, image, scale):
    #Find the intersection between this perpendicular vector and the contour
    shortPoint1, shortPoint2  = 0, 0
    i = 0
    height, width, channels = image.shape
    
    found1, found2 = False, False
    while(1):
        newPoint = [int(mp[0] + (v[0] * i)), int(mp[1] + (v[1] * i))] #Start at the middle point and go +1 pixel in the vector's direction
        newPoint2 =  [int(mp[0] - (v[0] * i)), int(mp[1] - (v[1] * i))] #Start at the middle point and go -1 pixel in the vector's direction
        
        #Check if any of the newPoints have reached out of bounds and stop the while loop so we don't go to infinity 
        # (This isn't really necesarry since I fixed the cause. but I'm keeping it just in case)
        if newPoint[0] > width or newPoint[0] < 0: # If the x point is outside the image width
            if newPoint2[1] > height or newPoint2[1] < 0: #If the y point is outside the image height
                print("Could not find a green width for this image")
                return None, None, None

        for p in cnt:
            # Check if the distance between the estimated point and the contour point is equal or under 1 pixel. 
            # This has been done since the contour may not store the point that the perpendicular vector intersects.
            # I.e in cases where the contour point is not stored at a diagonal pixel.
            # However, this approach means that on a good green, we lose 2 pixels of information.
            if np.linalg.norm(p[0] - newPoint) <= 1:
                #cv2.circle(image, tuple(newPoint), 1, (255,0,255), -1)
                shortPoint1 = newPoint
                found1 = True
            if np.linalg.norm(p[0] - newPoint2) <= 1:
                #cv2.circle(image, tuple(newPoint2), 1, (255,0,255), -1)
                shortPoint2 = newPoint2
                found2 = True                   
        if found1 and found2: 
            break     
        i += 0.2

    min_dist = [shortPoint1, shortPoint2, distance_two_points(np.asarray(shortPoint1), np.asarray(shortPoint2), image.shape, scale)]
    return min_dist

#Returns a list of all the bunker sizes in an image
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
# returns length, width, mp
# where length and width = [p1_x, p1_y], [p2_x, p2_y], dist] (the points are the two points on the contour that creates the line, and dist is the distance of the line)
# mp is the middlepoint of the green
def get_green_size(image, color='unet', scale=1000):
    _, _, _, green, _ = get_class_coords(image, color)

    if np.sum(green == 255) == 0:
        print("There is no green on this image")
        return None, None, None
    
    contours, _ = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area = 0
    contour = 0
    for cnt in contours:
        
        temp_area = cv2.contourArea(cnt)
        if temp_area > area:
            area = temp_area
            contour = cnt
            
    length = get_max_dist_cnt(contour, image.shape, scale)
    ## Draw a diagonal blue line with thickness of 5 px
    # cv2.line(image, tuple(max_dist[0]), tuple(max_dist[1]),(255,0,255),1)
    # cv2.circle(image, tuple(max_dist[0]), 1, (255,0,255), -1)
    # cv2.circle(image, tuple(max_dist[1]), 1, (255,0,255), -1)
    
    #quick maths kata
    mp = midpoint(length[0], length[1])

    #Vector for longest line
    #    B.x              A.x             B.y              A.y
    v = [length[1][0] - length[0][0], length[1][1] - length[0][1]]
    #Normalize the vector
    mag = math.sqrt(v[0]**2 + v[1]**2)
    
    if mag != 0:
        v[0] /= mag
        v[1] /= mag
    
    #Calculate the perpendicular vector
    v = [-v[1], v[0]]

    #Find the intersection between this perpendicular vector and the contour
    width = get_min_dist_cnt(contour, mp, v, image, scale)
    ## Draw a diagonal line with thickness of 5 px
#     cv2.line(image, min_dist[0], min_dist[1],(255,0,255),1)

# cv2.circle(image, mp, 1, (0,255,255), -1)

# cv2.imshow("Image with green sizes", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

    return length, width, mp



# PATH = "C:\\Users\\Vini\\Desktop\\AVS1\\data\\saved_test_images\\"
# def load_images_from_folder(path):
#     images = []
#     for filename in os.listdir(path):
#         #print("filename: ", filename)
#         img = cv2.imread(os.path.join(path,filename))
#         if img is not None:
#             images.append(img)
#         # return images
#     return images

# counter = 1
# for image in load_images_from_folder(PATH):
#     print("Checking green for image [", counter, "]")
#     max_dist, min_dist, mp = get_green_size(image, color='unet', scale=2000)
#     print(max_dist, min_dist, mp)
#     counter += 1


