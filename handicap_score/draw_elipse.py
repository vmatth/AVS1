import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from get_classes import get_class_coords
import convert


def draw_elipse_scratch_m(image, landing_point,center_point,stroke_dist_px,scale):
    if landing_point is not None:
        px_length_cm = convert.get_px_side(image.shape)    
        stroke_dist= int(convert.convert_px_to_m(px_length_cm,stroke_dist_px))
        if convert.convert_yards_to_m(230)<stroke_dist<=convert.convert_yards_to_m(250):
            width_m=convert.convert_yards_to_m(41)
            depth_m=convert.convert_yards_to_m(21)
        elif convert.convert_yards_to_m(210)<stroke_dist<=convert.convert_yards_to_m(230):
            width_m=convert.convert_yards_to_m(35)
            depth_m=convert.convert_yards_to_m(20)
        elif convert.convert_yards_to_m(190)<stroke_dist<=convert.convert_yards_to_m(210):
            width_m=convert.convert_yards_to_m(29)
            depth_m=convert.convert_yards_to_m(19)
        elif convert.convert_yards_to_m(170)<stroke_dist<=convert.convert_yards_to_m(190):
            width_m=convert.convert_yards_to_m(23)
            depth_m=convert.convert_yards_to_m(18)
        elif convert.convert_yards_to_m(150)<stroke_dist<=convert.convert_yards_to_m(170):
            width_m=convert.convert_yards_to_m(18)
            depth_m=convert.convert_yards_to_m(17)
        elif convert.convert_yards_to_m(130)<stroke_dist<=convert.convert_yards_to_m(150):
            width_m=convert.convert_yards_to_m(15)
            depth_m=convert.convert_yards_to_m(16)
        elif convert.convert_yards_to_m(110)<stroke_dist<=convert.convert_yards_to_m(130):
            width_m=convert.convert_yards_to_m(13)
            depth_m=convert.convert_yards_to_m(15)
        elif convert.convert_yards_to_m(90)<stroke_dist<=convert.convert_yards_to_m(110):
            width_m=convert.convert_yards_to_m(12)
            depth_m=convert.convert_yards_to_m(15)
        elif stroke_dist<=convert.convert_yards_to_m(90):
            width_m=convert.convert_yards_to_m(11)
            depth_m=convert.convert_yards_to_m(14)
        else :
            width_m=0
            depth_m=0
        #Calculation of width
        radius_w_px = convert.convert_m_to_px(px_length_cm, width_m, scale)/2
        #Calculation of width
        radius_d_px = convert.convert_m_to_px(px_length_cm, depth_m, scale)/2
        #Ellipse
        axes=(int(radius_d_px),int(radius_w_px))
        color = (0, 255, 255) #yellow
        point_a= (center_point[0][0],landing_point[0][1])
        c=stroke_dist_px
        a= landing_point[0][0]- point_a[0]
        angle_movement_radians= np.arccos(a/c)
        angle_movement_degrees= angle_movement_radians*57.296
        ellipse=cv2.ellipse(image,tuple(map(tuple,landing_point))[0],axes,angle_movement_degrees,0,360,color,1)
        return ellipse
    else :
        return image
def draw_elipse_scratch_f(image, landing_point,center_point,stroke_dist_px,scale):
    if landing_point is not None :
        px_length_cm = convert.get_px_side(image.shape)    
        stroke_dist= int(convert.convert_px_to_m(px_length_cm, stroke_dist_px))
        if convert.convert_yards_to_m(190)<stroke_dist<=convert.convert_yards_to_m(210):
            width_m=convert.convert_yards_to_m(34)
            depth_m=convert.convert_yards_to_m(28)
        elif convert.convert_yards_to_m(170)<stroke_dist<=convert.convert_yards_to_m(190):
            width_m=convert.convert_yards_to_m(30)
            depth_m=convert.convert_yards_to_m(24)
        elif convert.convert_yards_to_m(150)<stroke_dist<=convert.convert_yards_to_m(170):
            width_m=convert.convert_yards_to_m(26)
            depth_m=convert.convert_yards_to_m(20)
        elif convert.convert_yards_to_m(130)<stroke_dist<=convert.convert_yards_to_m(150):
            width_m=convert.convert_yards_to_m(20)
            depth_m=convert.convert_yards_to_m(18)
        elif convert.convert_yards_to_m(110)<stroke_dist<=convert.convert_yards_to_m(130):
            width_m=convert.convert_yards_to_m(17)
            depth_m=convert.convert_yards_to_m(17)
        elif convert.convert_yards_to_m(90)<stroke_dist<=convert.convert_yards_to_m(110):
            width_m=convert.convert_yards_to_m(14)
            depth_m=convert.convert_yards_to_m(16)
        elif stroke_dist<=convert.convert_yards_to_m(90):
            width_m=convert.convert_yards_to_m(12)
            depth_m=convert.convert_yards_to_m(15)
        else :
            print("hola_scract")
            width_m=0
            depth_m=0
        #Calculation of width
        radius_w_px = convert.convert_m_to_px(px_length_cm, width_m, scale)/2
        #Calculation of width
        radius_d_px = convert.convert_m_to_px(px_length_cm, depth_m, scale)/2
        #Ellipse
        axes=(int(radius_d_px),int(radius_w_px))
        color = (0, 0, 255) #red
        point_a= (center_point[1][0],landing_point[0][1])
        c=stroke_dist_px
        a= landing_point[0][0]- point_a[0]
        angle_movement_radians= np.arccos(a/c)
        angle_movement_degrees= angle_movement_radians*57.296
        ellipse=cv2.ellipse(image,tuple(map(tuple,landing_point))[0],axes,angle_movement_degrees,0,360,color,1)
        return ellipse
    else:
        return image
def draw_elipse_bogey_f(image, landing_point,center_point,stroke_dist_px,scale):
    if landing_point is not None :
        px_length_cm = convert.get_px_side(image.shape)    
        stroke_dist= int(convert.convert_px_to_m(px_length_cm, stroke_dist_px))
        if convert.convert_yards_to_m(130)<stroke_dist<=convert.convert_yards_to_m(150):
            width_m=convert.convert_yards_to_m(24)
            depth_m=convert.convert_yards_to_m(30)
        elif convert.convert_yards_to_m(110)<stroke_dist<=convert.convert_yards_to_m(130):
            width_m=convert.convert_yards_to_m(21)
            depth_m=convert.convert_yards_to_m(27)
        elif convert.convert_yards_to_m(90)<stroke_dist<=convert.convert_yards_to_m(110):
            width_m=convert.convert_yards_to_m(19)
            depth_m=convert.convert_yards_to_m(24)
        elif stroke_dist<=convert.convert_yards_to_m(90):
            width_m=convert.convert_yards_to_m(17)
            depth_m=convert.convert_yards_to_m(22)
        else :
            print("hola bogey female")
            width_m=0
            depth_m=0
        #Calculation of width
        radius_w_px = convert.convert_m_to_px(px_length_cm, width_m, scale)/2
        #Calculation of width
        radius_d_px = convert.convert_m_to_px(px_length_cm, depth_m, scale)/2
        #Ellipse
        axes=(int(radius_d_px),int(radius_w_px))
        color = (0, 0, 255) #red
        point_a= (center_point[1][0],landing_point[0][1])
        c=stroke_dist_px
        a= landing_point[0][0]- point_a[0]
        angle_movement_radians= np.arccos(a/c)
        angle_movement_degrees= angle_movement_radians*57.296
        ellipse=cv2.ellipse(image,tuple(map(tuple,landing_point))[0],axes,angle_movement_degrees,0,360,color,1)
        return ellipse
    else:
        return image    
def draw_elipse_bogey_m(image, landing_point,center_point,stroke_dist_px,scale):
    if landing_point is not None :
        px_length_cm = convert.get_px_side(image.shape)    
        stroke_dist= int(convert.convert_px_to_m(px_length_cm, stroke_dist_px))
        if convert.convert_yards_to_m(170)<stroke_dist<=convert.convert_yards_to_m(190):
            width_m=convert.convert_yards_to_m(29)
            depth_m=convert.convert_yards_to_m(34)
        elif convert.convert_yards_to_m(150)<stroke_dist<=convert.convert_yards_to_m(170):
            width_m=convert.convert_yards_to_m(24)
            depth_m=convert.convert_yards_to_m(28)
        elif convert.convert_yards_to_m(130)<stroke_dist<=convert.convert_yards_to_m(150):
            width_m=convert.convert_yards_to_m(20)
            depth_m=convert.convert_yards_to_m(25)
        elif convert.convert_yards_to_m(110)<stroke_dist<=convert.convert_yards_to_m(130):
            width_m=convert.convert_yards_to_m(18)
            depth_m=convert.convert_yards_to_m(23)
        elif convert.convert_yards_to_m(90)<stroke_dist<=convert.convert_yards_to_m(110):
            width_m=convert.convert_yards_to_m(17)
            depth_m=convert.convert_yards_to_m(21)
        elif stroke_dist<=convert.convert_yards_to_m(90):
            width_m=convert.convert_yards_to_m(16)
            depth_m=convert.convert_yards_to_m(19)
        else :
            print("hola_bogey_male")
            width_m=0
            depth_m=0
        #Calculation of width
        radius_w_px = convert.convert_m_to_px(px_length_cm, width_m, scale)/2
        #Calculation of width
        radius_d_px = convert.convert_m_to_px(px_length_cm, depth_m, scale)/2
        #Ellipse
        axes=(int(radius_d_px),int(radius_w_px))
        color = (0, 255, 255) #yellow
        point_a= (center_point[0][0],landing_point[0][1])
        c=stroke_dist_px
        a= landing_point[0][0]- point_a[0]
        angle_movement_radians= np.arccos(a/c)
        angle_movement_degrees= angle_movement_radians*57.296
        ellipse=cv2.ellipse(image,tuple(map(tuple,landing_point))[0],axes,angle_movement_degrees,0,360,color,1)
        return ellipse
    else:
        return image
