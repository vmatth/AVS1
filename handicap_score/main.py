from scale import get_scale
import cv2
import stroke
import do_everything
import get_classes
import size
import click





def main():
    #image = cv2.imread('C:\\Users\\jacob\\Project\\for_jacobo.png')
    image_path='C:\\Users\\jespe\\Desktop\\AVS1\\11_figure_1500_001.png'
    image_path1='C:\\Users\\jespe\\Desktop\\AVS1\\11_prediction_1500_001.png'
    prediction = cv2.imread(image_path1)
    original = cv2.imread(image_path)

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print("img shape: ", image.shape) # (256,416,3)
    #ori_image=cv2.resize(ori_image,(800,450))
    #image2=cv2.resize(image,(800,450))
    #scale = 1250
    #Get scale from the name of the image
    scale= get_scale(image_path)
    print("scale: ", scale)
    # Get stroke lenghts in meters
    total_s_m, carry_s_m = stroke.get_stroke_lengths(original.shape, 250, 230, scale)
    total_s_f, carry_s_f = stroke.get_stroke_lengths(original.shape, 210, 190, scale)
    total_b_m, carry_b_m = stroke.get_stroke_lengths(original.shape, 200, 180, scale)
    total_b_f, carry_b_f = stroke.get_stroke_lengths(original.shape, 150, 130, scale)


    # Get Class coordinates
    _, _, fairway, _, _ = get_classes.get_class_coords(prediction)
    fairway_coords = cv2.findNonZero(fairway)

    #Different players
    player_type=["scratch_male","scratch_female","bogey_male","bogey_female"]

    #Click the tee
    center_point=[]
    center_point= click.get_click_coords(original,center_point)
    #print("centerpoint: ", center_point)
    
    #Get green sizes
    green_length, green_width, green_centerpoint = size.get_green_size(prediction, color='unet', scale=scale)
    #print("green centerpoint: ", green_centerpoint)
    male_point = center_point[0]
    female_point = center_point[1]

    original = do_everything.run_all_calcs(original, prediction, fairway_coords, male_point, green_centerpoint, scale, total_s_m, player_type[0])
    original1 = do_everything.run_all_calcs(original, prediction, fairway_coords, female_point, green_centerpoint, scale, total_s_f, player_type[1])
    original2 = do_everything.run_all_calcs(original1, prediction, fairway_coords, male_point, green_centerpoint, scale, total_b_m, player_type[2])
    original3 = do_everything.run_all_calcs(original2, prediction, fairway_coords, female_point, green_centerpoint, scale, total_b_f, player_type[3])
    
    print(f"Green length: {green_length[-1]} [m]")
    print(f"Green width: {green_width[-1]} [m]")

    cv2.imshow("image", original3)
    cv2.waitKey(0)
    cv2.destroyWindow()


    



if __name__ == "__main__":
    main()