import numpy as np
from course_rating import convert
from course_rating import distance
from course_rating import stroke
from course_rating import size



def run_all_calcs(prediction, fairway_coords, point, green_centerpoint, scale, stroke_distance, carry_distance, player_type, pixel_size):
    center_point = point
    stroke_number = 1
    total_distance = 0
    fairway_widths = []
    bunker_dists = []
    water_dists = []
    landing_points = []
    distances_to_green = []
    while(True):
        print(f"stroke: {stroke_number} for {player_type}")

        if player_type=="scratch_male" or player_type=="bogey_male":
            if stroke_number == 2:
                stroke_distance = stroke_distance-convert.convert_m_to_px(pixel_size, 27.43200, scale)
                carry_distance = carry_distance-convert.convert_m_to_px(pixel_size, 27.43200, scale)

        if player_type=="scratch_female" or player_type== "bogey_female":
            if stroke_number == 2:
                stroke_distance = stroke_distance-convert.convert_m_to_px(pixel_size, 18.28800, scale)
                carry_distance = carry_distance-convert.convert_m_to_px(pixel_size, 18.28800, scale)
                
        intersections = stroke.get_intersections(fairway_coords, point, stroke_distance)
        carry_intersections = stroke.get_intersections(fairway_coords, point, carry_distance)
        
        landing_point = (0,0)
        # Get the shortest intersections in cases where there are multiple intersections with the fairway
        if intersections and carry_intersections :
            intersections = stroke.get_shortest_intersections(intersections, point, green_centerpoint)
            carry_intersections = stroke.get_shortest_intersections(carry_intersections, point, green_centerpoint)
        if intersections and carry_intersections : 
            landing_point = stroke.get_landing_point(intersections)
            landing_points.append(landing_point)
            print("landing point: ", landing_point)
            distance_to_green = stroke.get_distance_landing_point_to_hole(landing_point, green_centerpoint, prediction.shape, scale)
            print(f"Distance to green: {distance_to_green} [m]")
            fairway_width_t = stroke.get_fairway_width(intersections, prediction.shape, scale)
            fairway_width_c = stroke.get_fairway_width(carry_intersections, prediction.shape, scale)
            avg_fairway_width = (fairway_width_t+fairway_width_c)/2
            # print("fairway_width total: ", fairway_width_t)
            # print("fairway width carry: ", fairway_width_c)
            print("avg fairway width :", avg_fairway_width)
            fairway_widths.append(avg_fairway_width)

            total_distance += distance.distance_two_points(point, landing_point, prediction.shape, scale)
            print(f"Total distance for {player_type}: {total_distance} [m]")

            bunker_dist, water_dist = distance.distance_to_objects(prediction, landing_point, scale)
            bunker_dists.append(bunker_dist)
            water_dists.append(water_dist)

            #Get the distance from the landingpoint to the green for each stroke
            distance_to_green = stroke.get_distance_landing_point_to_hole(np.array(point), green_centerpoint, prediction.shape, scale)
            distance_front_green, distance_back_green = size.get_distance_to_front_and_back_green(prediction, np.array(point), green_centerpoint, scale, color="unet")
            distances_to_green.append([distance_front_green, distance_to_green, distance_back_green])

            point = landing_point #Refresh the point to be the new landingpoint
        else:
            if np.sum(landing_point) == 0 and stroke_number > 1:
                distance_to_green = stroke.get_distance_landing_point_to_hole(np.array(point), green_centerpoint, prediction.shape, scale)
                lenght_of_hole = total_distance + distance_to_green
                print(f"Total distance for {player_type}: {lenght_of_hole} [m]")
                distance_front_green, distance_back_green = size.get_distance_to_front_and_back_green(prediction, point, green_centerpoint, scale, color="unet")
                print(f"distance to front green {distance_front_green} [m], distance to back green {distance_back_green} [m]")

                distances_to_green.append([distance_front_green, distance_to_green, distance_back_green])

            elif np.sum(landing_point) == 0 and stroke_number == 1:
                distance_to_green = stroke.get_distance_landing_point_to_hole(np.array(center_point), green_centerpoint, prediction.shape, scale)
                print(f"Total distance for {player_type}: {distance_to_green} [m]")
                distance_front_green, distance_back_green = size.get_distance_to_front_and_back_green(prediction, np.array(center_point), green_centerpoint, scale, color="unet")
                print(f"distance to front green {distance_front_green} [m], distance to back green {distance_back_green} [m]")
                lenght_of_hole = distance_to_green
                distances_to_green.append([distance_front_green, distance_to_green, distance_back_green])
            break


        stroke_number += 1
    print("------------------------")
    print("DISTANCES TO GREEN", distances_to_green)
    return fairway_widths, bunker_dists, water_dists, landing_points, stroke_number, lenght_of_hole, distances_to_green
