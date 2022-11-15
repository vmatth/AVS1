import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from get_classes import get_class_coords
import convert
from click import get_click_coords
import draw_elipse
import distance
from size import get_green_size
import stroke
import size



def run_all_calcs(image2, fairway_coords, point, green_centerpoint, scale, stroke_distance, player_type):
    stroke_number = 1
    total_distance = 0
    while(True):
        print(f"stroke: {stroke_number}")

        if player_type=="scratch_male" or player_type=="bogey_male":
            if stroke_number == 2:
                stroke_distance = stroke_distance-convert.convert_yards_to_m(30)

        if player_type=="scratch_female" or player_type== "bogey_female":
            if stroke_number == 2:
                stroke_distance = stroke_distance-convert.convert_yards_to_m(20)

        intersections = stroke.get_intersections(fairway_coords, point, stroke_distance)

        # landing_point = stroke.get_landing_point(intersections)
        # fairway_width = stroke.get_fairway_width(intersections, image2.shape, scale)
        # image2=draw_elipse.draw_elipse_scratch_m(image2, landing_point, point, stroke_distance ,1000)

        # Get the shortest intersections in cases where there are multiple intersections with the fairway
        if intersections:
            intersections = stroke.get_shortest_intersections(intersections, point, green_centerpoint)


        if intersections: # Make sure there are intersections with the fairway

            landing_point = stroke.get_landing_point(intersections)
            print("landing point: ", landing_point)
            distance_to_green = stroke.get_distance_landing_point_to_hole(landing_point, green_centerpoint, image2.shape, scale)
            print(f"Distance to green: {distance_to_green} [m]")
            fairway_width = stroke.get_fairway_width(intersections, image2.shape, scale)

            total_distance += distance.distance_two_points(point, landing_point, image2.shape, scale)
            print(f"Total distance for {player_type}: {total_distance} [m]")
            
            image2=draw_elipse.draw_elipse(image2, landing_point, point, player_type,stroke_distance ,1000)
            
            cv2.circle(image2, landing_point, 1, (255, 255, 0))

            bunker_dist, water_dist = distance.distance_to_objects(image2, landing_point)
            #print("bunker dist: ", bunker_dist)
            #print("water dist: ", water_dist)
            if bunker_dist is not False:
                bunker_coords = stroke.extract_list(bunker_dist)
                #print("Coordenates", bunker_coords)
                for i in bunker_coords:
                    #print("i:", i)
                    cv2.line(image2, landing_point, i, (255,255,255), 1)
            
            if water_dist is not False:
                Water_coords = stroke.extract_list(water_dist)
                #print("Coordenates", Water_coords)
                for i in Water_coords:
                    print("i:", i)
                    cv2.line(image2, landing_point, i, (255,255,255), 1)
            
            # print("Coordenates", bunker_coords)
            # print("bunker dist: ", bunker_dist)

            point = landing_point #Refresh the point to be the new landingpoint



        else:
            distance_to_green = stroke.get_distance_landing_point_to_hole(landing_point, green_centerpoint, image2.shape, scale)
            total_distance = distance_to_green + total_distance
            print(f"Total distance for {player_type}: {total_distance} [m]")

            distance_front_green, distance_back_green = size.get_distance_to_front_and_back_green(image2, landing_point, green_centerpoint, scale, color="unet")
            print(f"distance to front green {distance_front_green} [m], distance to back green {distance_back_green} [m]")

            break


        stroke_number += 1
    return image2
