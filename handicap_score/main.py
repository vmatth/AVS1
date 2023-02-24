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
import csv



def main():

    header = ["Circularity"]

    with open("circularity.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        file.close()


    path='D:\\Users\\jacob\\Master\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Images_data_collection\\Kaggle\\\Testing_accuracies\\test_predict_3\\*.png'
    #path='C:\\Users\\jespe\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\saved_test_images_best_model\\*.png'
    #path='C:\\Users\\jespe\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\Course_dist_testing\\*.png'
    cv_img = []
    for file in natsorted(glob.glob(path)):
        if "__" in file:
            cv_img.append(file)
            
        elif "prediction" in file:
           cv_img.append(file)
    
    for i in range(0, len(cv_img), 2):
        print("--------------------------------------------------------------")
        original = cv2.imread(cv_img[i])
        prediction = cv2.imread(cv_img[i+1])
        #original = prediction
    
    # image_path='C:\\Users\\jespe\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\saved_test_images\\32__Soelleroed Golfklub_2000_08.png'
    # image_path1='C:\\Users\\jespe\\Aalborg Universitet\\AVS1 - Golf Project - General\\1. Project\\3. Data\\saved_test_images\\32_prediction.png'
    # prediction = cv2.imread(image_path1)
    # original = cv2.imread(image_path)
    
        pixel_size = convert.get_px_side(original.shape)

        #Get scale from the name of the image
        scale, img_path = get_scale(cv_img[i])
        #scale = get_scale(image_path)
        print("scale: ", scale)
        # Get stroke lenghts in meters
        total_s_m, carry_s_m = stroke.get_stroke_lengths(original.shape, 250, 230, scale)
        total_s_f, carry_s_f = stroke.get_stroke_lengths(original.shape, 210, 190, scale)
        total_b_m, carry_b_m = stroke.get_stroke_lengths(original.shape, 200, 180, scale)
        total_b_f, carry_b_f = stroke.get_stroke_lengths(original.shape, 150, 130, scale)
        
        # print("total sm: ", total_s_m)
        # print("total sf: ", total_s_f)
        # print("total bm: ", total_b_m)
        # print("total bf: ", total_b_f)

        # print("carry sm: ", carry_s_m)
        # print("carry sf: ", carry_s_f)
        # print("carry bm: ", carry_b_m)
        # print("carry bf: ", carry_b_f)



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
        green_length, green_width, green_centerpoint, contour = size.get_green_size(prediction, color='unet', scale=scale)
        #print("green centerpoint: ", green_centerpoint)
        male_point = center_point[0]
        female_point = center_point[1]

        with open("Green_things.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([img_path[-1],green_length,green_width])
            file.close() 




       # bunker_to_obstacles_male_tee, water_to_obstacles_male_tee = distance.distance_to_objects(prediction, male_point, scale, max_distance=convert.convert_px_to_m(pixel_size, total_s_m, scale))
       # print(f"Distance to bunkers - male tee: {bunker_to_obstacles_male_tee} [m]")
       # print(f"Distance to water - male tee: {water_to_obstacles_male_tee} [m]")
       # bunker_to_obstacles_female_tee, water_to_obstacles_female_tee  = distance.distance_to_objects(prediction, female_point, scale, max_distance=convert.convert_px_to_m(pixel_size, total_s_f, scale))
       # print(f"Distance to bunkers - female tee: {bunker_to_obstacles_female_tee} [m]")
       # print(f"Distance to water - female tee: {water_to_obstacles_female_tee} [m]")

        
        if green_centerpoint:
            original = do_everything.run_all_calcs(original, prediction, fairway_coords, male_point, green_centerpoint, scale, total_s_m, carry_s_m, player_type[0], pixel_size, contour,img_path[-1])
            original1 = do_everything.run_all_calcs(original, prediction, fairway_coords, female_point, green_centerpoint, scale, total_s_f, carry_s_f, player_type[1], pixel_size, contour,img_path[-1])
            #original2 = do_everything.run_all_calcs(original1, prediction, fairway_coords, male_point, green_centerpoint, scale, total_b_m, carry_b_m, player_type[2], pixel_size)
            #original3 = do_everything.run_all_calcs(original2, prediction, fairway_coords, female_point, green_centerpoint, scale, total_b_f, carry_b_f, player_type[3], pixel_size)

        
            

            print(f"Green length: {green_length[-1]} [m]")
            print(f"Green width: {green_width[-1]} [m]")
            #cv2.imshow("image", original1)
            #cv2.waitKey(0)
            #cv2.destroyWindow()
        else:
             print("No green was detected")
    
    

     #cv2.destroyWindow()


    



if __name__ == "__main__":
    main()