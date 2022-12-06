import numpy as np
import cv2
from course_rating import stroke
from course_rating import do_everything
from course_rating import get_classes
from course_rating import size
from course_rating import distance
from course_rating import convert

# Returns all relevant distance measurements
#Parameters: prediction mask cv image and the scale
def return_everything(image, scale, male_point, female_point):
    print("img", image.shape)
    pixel_size = convert.get_px_side(image.shape)

    #scale = get_scale(image_path)
    print("scale: ", scale)
    # Get stroke lenghts in meters
    total_s_m, carry_s_m = stroke.get_stroke_lengths(image.shape, 250, 230, scale)
    total_s_f, carry_s_f = stroke.get_stroke_lengths(image.shape, 210, 190, scale)
    total_b_m, carry_b_m = stroke.get_stroke_lengths(image.shape, 200, 180, scale)
    total_b_f, carry_b_f = stroke.get_stroke_lengths(image.shape, 150, 130, scale)

    # Get Class coordinates
    _, _, fairway, _, _ = get_classes.get_class_coords(image)
    if np.sum(np.array(fairway)) > 0:
        fairway_coords = cv2.findNonZero(fairway)
    else:
        fairway_coords = 0
        
    
    
    #Different players
    player_type=["scratch_male","scratch_female","bogey_male","bogey_female"]

    #Get green sizes
    green_length, green_width, green_centerpoint = size.get_green_size(image, color='unet', scale=scale)

    bunker_to_obstacles_male_tee, water_to_obstacles_male_tee = distance.distance_to_objects(image, male_point, scale, max_distance=convert.convert_px_to_m(pixel_size, total_s_m, scale))
    # print(f"Distance to bunkers - male tee: {bunker_to_obstacles_male_tee} [m]")
    # print(f"Distance to water - male tee: {water_to_obstacles_male_tee} [m]")
    bunker_to_obstacles_female_tee, water_to_obstacles_female_tee  = distance.distance_to_objects(image, female_point, scale, max_distance=convert.convert_px_to_m(pixel_size, total_s_f, scale))
    # print(f"Distance to bunkers - female tee: {bunker_to_obstacles_female_tee} [m]")
    # print(f"Distance to water - female tee: {water_to_obstacles_female_tee} [m]")

    # actual good code : )
    if green_centerpoint:
        fairway_width_s_m, bunker_dist_s_m, water_dist_s_m, landing_points_s_m, stroke_number_s_m, length_of_hole_s_m, distances_to_green_s_m = do_everything.run_all_calcs(image, fairway_coords, male_point, green_centerpoint, scale, total_s_m, carry_s_m, player_type[0], pixel_size)
        fairway_width_s_f, bunker_dist_s_f, water_dist_s_f, landing_points_s_f, stroke_number_s_f, length_of_hole_s_f, distances_to_green_s_f = do_everything.run_all_calcs(image, fairway_coords, female_point, green_centerpoint, scale, total_s_f, carry_s_f, player_type[1], pixel_size)
        fairway_width_b_m, bunker_dist_b_m, water_dist_b_m, landing_points_b_m, stroke_number_b_m, length_of_hole_b_m, distances_to_green_b_m = do_everything.run_all_calcs(image, fairway_coords, male_point, green_centerpoint, scale, total_b_m, carry_b_m, player_type[2], pixel_size)
        fairway_width_b_f, bunker_dist_b_f, water_dist_b_f, landing_points_b_f, stroke_number_b_f, length_of_hole_b_f, distances_to_green_b_f = do_everything.run_all_calcs(image, fairway_coords, female_point, green_centerpoint, scale, total_b_f, carry_b_f, player_type[3], pixel_size)
        all_fairway_widths = [fairway_width_s_m, fairway_width_s_f, fairway_width_b_m, fairway_width_b_f]
        all_bunker_dists = [bunker_dist_s_m, bunker_dist_s_f, bunker_dist_b_m, bunker_dist_b_f]
        all_water_dists = [water_dist_s_m, water_dist_s_f, water_dist_b_m, water_dist_b_f]
        all_landing_points = [landing_points_s_m, landing_points_s_f, landing_points_b_m, landing_points_b_f]
        all_stroke_numbers = [stroke_number_s_m, stroke_number_s_f, stroke_number_b_m, stroke_number_b_f]
        all_length_of_holes = [length_of_hole_s_m, length_of_hole_s_f, length_of_hole_b_m, length_of_hole_b_f]
        all_bunker_dists_from_tees = [bunker_to_obstacles_male_tee, bunker_to_obstacles_female_tee]
        all_water_dists_from_tees = [water_to_obstacles_male_tee, water_to_obstacles_male_tee]
        all_distances_to_green = [distances_to_green_s_m, distances_to_green_s_f, distances_to_green_b_m, distances_to_green_b_f]
        return all_fairway_widths, all_bunker_dists, all_water_dists, all_landing_points, all_stroke_numbers, all_length_of_holes, all_bunker_dists_from_tees, all_water_dists_from_tees, all_distances_to_green
    else:
        raise EnvironmentError("No green was detected in the selected image.")
###
# For every player type
    # For every stroke
        # Fairway width - check
        # Length to obstacles from landing point - check
        # Landing zones - check
    # Length of the hole - check   
    # Length to obstacles from tee - check
    # Distance from final landing point to front and back of green - check
    
