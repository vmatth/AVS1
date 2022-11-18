from scale import get_scale
import numpy as np
import cv2
import stroke
import do_everything
import get_classes
import size
import click
import distance
import convert
import glob
from natsort import natsorted



def main():
    path='C:\\Users\\jespe\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\saved_test_images\\*.png'
    cv_img = []
    for file in natsorted(glob.glob(path)):
        if "__" in file:
            cv_img.append(file)
            
        elif "prediction" in file:
            cv_img.append(file)
    
    for i in range(0, len(cv_img), 2):
        original = cv2.imread(cv_img[i])
        prediction = cv2.imread(cv_img[i+1])
    
    
    # image_path='C:\\Users\\jespe\\OneDrive\\Skrivebord\\AVS1\\11_figure_1500_001.png'
    # image_path1='C:\\Users\\jespe\\OneDrive\\Skrivebord\\AVS1\\11_prediction_1500_001.png'
    # prediction = cv2.imread(image_path1)
    # original = cv2.imread(image_path)
    
        pixel_size = convert.get_px_side(original.shape)

        #Get scale from the name of the image
        scale = get_scale(cv_img[i])
        print("scale: ", scale)
        # Get stroke lenghts in meters
        total_s_m, carry_s_m = stroke.get_stroke_lengths(original.shape, 250, 230, scale)
        total_s_f, carry_s_f = stroke.get_stroke_lengths(original.shape, 210, 190, scale)
        total_b_m, carry_b_m = stroke.get_stroke_lengths(original.shape, 200, 180, scale)
        total_b_f, carry_b_f = stroke.get_stroke_lengths(original.shape, 150, 130, scale)

        # Get Class coordinates
        _, _, fairway, _, _ = get_classes.get_class_coords(prediction)
        if np.sum(np.array(fairway)) > 0:
            fairway_coords = cv2.findNonZero(fairway)
        else:
            fairway_coords = 0
            
            
    
        
        
        #Different players
        player_type=["scratch_male","scratch_female","bogey_male","bogey_female"]

        #Click the tee
        center_point = []
        center_point = click.get_click_coords(original, center_point)
        #print("centerpoint: ", center_point)
    
        #Get green sizes
        green_length, green_width, green_centerpoint = size.get_green_size(prediction, color='unet', scale=scale)
        #print("green centerpoint: ", green_centerpoint)
        male_point = center_point[0]
        female_point = center_point[1]
    
        bunker_to_obstacles_male_tee, water_to_obstacles_male_tee = distance.distance_to_objects(prediction, male_point, scale, max_distance=convert.convert_px_to_m(pixel_size, total_s_m, scale))
        print(f"Distance to bunkers - male tee: {bunker_to_obstacles_male_tee} [m]")
        print(f"Distance to water - male tee: {water_to_obstacles_male_tee} [m]")
        bunker_to_obstacles_female_tee, water_to_obstacles_female_tee  = distance.distance_to_objects(prediction, female_point, scale, max_distance=convert.convert_px_to_m(pixel_size, total_s_f, scale))
        print(f"Distance to bunkers - female tee: {bunker_to_obstacles_female_tee} [m]")
        print(f"Distance to water - female tee: {water_to_obstacles_female_tee} [m]")

        original = do_everything.run_all_calcs(original, prediction, fairway_coords, male_point, green_centerpoint, scale, total_s_m, player_type[0], pixel_size)
        original1 = do_everything.run_all_calcs(original, prediction, fairway_coords, female_point, green_centerpoint, scale, total_s_f, player_type[1], pixel_size)
        original2 = do_everything.run_all_calcs(original1, prediction, fairway_coords, male_point, green_centerpoint, scale, total_b_m, player_type[2], pixel_size)
        original3 = do_everything.run_all_calcs(original2, prediction, fairway_coords, female_point, green_centerpoint, scale, total_b_f, player_type[3], pixel_size)
        
        print(f"Green length: {green_length[-1]} [m]")
        print(f"Green width: {green_width[-1]} [m]")


        cv2.imshow("image", original3)
        cv2.waitKey(0)
    cv2.destroyWindow()


    



if __name__ == "__main__":
    main()