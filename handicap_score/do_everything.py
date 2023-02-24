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
import csv



def run_all_calcs(original,prediction, fairway_coords, point, green_centerpoint, scale, stroke_distance, carry_distance, player_type, pixel_size, contour,hole_name):
    center_point = point
    stroke_number = 1
    total_distance = 0
    while(True):
        print(f"stroke: {stroke_number} for {player_type}")

        if player_type=="scratch_male" or player_type=="bogey_male":
            if stroke_number == 2:
                stroke_distance = stroke_distance-convert.convert_m_to_px(pixel_size, 27.43200, scale)

        if player_type=="scratch_female" or player_type== "bogey_female":
            if stroke_number == 2:
                stroke_distance = stroke_distance-convert.convert_m_to_px(pixel_size, 18.28800, scale)
                
        intersections = stroke.get_intersections(fairway_coords, point, stroke_distance)
        carry_intersections = stroke.get_intersections(fairway_coords, point, carry_distance)
        
        landing_point = (0,0)
        # Get the shortest intersections in cases where there are multiple intersections with the fairway
        if intersections and carry_intersections :
            intersections = stroke.get_shortest_intersections(intersections, point, green_centerpoint)
            carry_intersections = stroke.get_shortest_intersections(carry_intersections, point, green_centerpoint)
        if intersections and carry_intersections : 
            landing_point = stroke.get_landing_point(intersections)
        
            print("landing point: ", landing_point)
            distance_to_green = stroke.get_distance_landing_point_to_hole(landing_point, green_centerpoint, original.shape, scale)
            print(f"Distance to green: {distance_to_green} [m]")
            fairway_width_t = stroke.get_fairway_width(intersections, original.shape, scale)
            fairway_width_c = stroke.get_fairway_width(carry_intersections, original.shape, scale)
            avg_fairway_width = (fairway_width_t+fairway_width_c)/2
            # print("fairway_width total: ", fairway_width_t)
            # print("fairway width carry: ", fairway_width_c)
            print("avg fairway width :", avg_fairway_width)

            total_distance += distance.distance_two_points(point, landing_point, original.shape, scale)
            print(f"Total distance for {player_type}: {total_distance} [m]")
            with open("distance_calc_paper.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([hole_name,int(total_distance)])
                    file.close()


            original=draw_elipse.draw_elipse(original, landing_point, point, player_type, stroke_distance, scale)
            
            cv2.circle(original, landing_point, 1, (255, 255, 0))

            bunker_dist, water_dist = distance.distance_to_objects(prediction, landing_point, scale)
            #print("bunker dist: ", bunker_dist)
            #print("water dist: ", water_dist)
            if bunker_dist is not False:
                bunker_coords = stroke.extract_list(bunker_dist)
                #print("Coordenates", bunker_coords)
                for i in bunker_coords:
                    #print("i:", i)
                    cv2.line(original, landing_point, i, (255, 255, 255), 1)
            
            if water_dist is not False:
                Water_coords = stroke.extract_list(water_dist)
                #print("Coordenates", Water_coords)
                for i in Water_coords:
                    #print("i:", i)
                    cv2.line(original, landing_point, i, (255,255,255), 1)

            point = landing_point #Refresh the point to be the new landingpoint
            


        else:
            if np.sum(landing_point) == 0 and stroke_number > 1:
                distance_to_green = stroke.get_distance_landing_point_to_hole(np.array(point), green_centerpoint, original.shape, scale)
                lenght_of_hole = total_distance + distance_to_green
                print(f"Total distance for {player_type}: {lenght_of_hole} [m]")
                distance_front_green, distance_back_green = size.get_distance_to_front_and_back_green(prediction, point, green_centerpoint, contour, scale, color="unet")
                print(f"distance to front green {distance_front_green} [m], distance to back green {distance_back_green} [m]")
                
                row = []
                row.append(int(lenght_of_hole))
                with open("distance_calc_paper.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([hole_name,row])
                    file.close()


            elif np.sum(landing_point) == 0 and stroke_number == 1:
                
                distance_to_green = stroke.get_distance_landing_point_to_hole(np.array(center_point), green_centerpoint, original.shape, scale)
                print(f"Total distance for {player_type}: {distance_to_green} [m]")
                distance_front_green, distance_back_green = size.get_distance_to_front_and_back_green(prediction, np.array(center_point), green_centerpoint, contour, scale, color="unet")
                print(f"distance to front green {distance_front_green} [m], distance to back green {distance_back_green} [m]")
            
                row = []
                row.append(int(distance_to_green))
                with open("distance_calc_paper.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([hole_name, row])
                    file.close()

            break


        stroke_number += 1
    return original
