
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from get_classes import get_class_coords
import convert
from click import get_click_coords
import draw_elipse
import distance


def get_stroke_lengths(image_shape):
    # scratch_m: 250y, scratch_F: 210y, bogey_m: 200y, bogey_f: 150y
    stroke_lenghts = [250, 210, 190, 150]
    stroke_dists_px = []
    px_length_cm = convert.get_px_side(image_shape)

    for lenghts in stroke_lenghts:
        lenghts = convert.convert_yards_to_m(lenghts)
        stroke_dists_px.append(int(convert.convert_m_to_px(px_length_cm, lenghts)))

    # scratch_m: 230y, scratch_F: 190y, bogey_m: 180y, bogey_f: 130y
    carry_lenghts = [230, 190, 180, 130]
    carry_dists_px = []

    for lenghts in carry_lenghts:
        lenghts = convert.convert_yards_to_m(lenghts)
        carry_dists_px.append(int(convert.convert_m_to_px(px_length_cm, lenghts)))
    


    return stroke_dists_px, carry_dists_px
    

def calc_fairway_width(class_, centerpoint, stroke_dist, image_shape):
    # add try except function
    try:
        intersection = []
        edge_points = []
        for point in class_:
            # Check if distance from the tee to the fairway is the same as the stroke lenght
            if (int(math.sqrt((centerpoint[0]-point[0][0])**2+(centerpoint[1]-point[0][1])**2))) == int(stroke_dist):
                intersection.append(point)

        # Get landing zone coordinates
        landing_zone = intersection[int(len(intersection)/2)]
        
        # Calculate width of the fairway
        edge_points.append(intersection[0])
        edge_points.append(intersection[-1])

        fairway_width = math.sqrt((edge_points[1][0][0]- edge_points[0][0][0])**2+(edge_points[1][0][1]-edge_points[0][0][1])**2)
        
        px_length_cm = convert.get_px_side(image_shape)

        fairway_width = convert.convert_px_to_m(px_length_cm, fairway_width)

        print(f"Fairway width: {int(fairway_width)} m")

        return landing_zone, fairway_width

    except:
        if len(intersection) == 0:
            print("No intersections with fairway found")
            return None, None

def extract_list(lst):
    return [item[0] for item in lst]

def main():
    #image = cv2.imread('C:\\Users\\jacob\\Project\\for_jacobo.png')
    image = cv2.imread('C:\\Users\\jespe\\Desktop\\AVS1\\for_jacobo.png')
    #image = cv2.imread('C:\\Users\\jespe\\Desktop\\AVS1\\mask.png')
    #ori_image = cv2.imread('C:\\Users\\jespe\\Desktop\\AVS1\\original_img.jpg')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print("img shape: ", image.shape) # (256,416,3)
    #ori_image=cv2.resize(ori_image,(800,450))
    image2=cv2.resize(image,(800,450))

    # Get stroke lenghts in meters
    total_lenghts, carry_lenghts = get_stroke_lengths(image2.shape)
    print("stroke_lenghts: ", total_lenghts)

    # Get Class coordinates
    _, _, fairway, _, _ = get_class_coords(image2)
    fairway_coords = cv2.findNonZero(fairway)

    #Click the tee
    center_point=[]
    center_point=get_click_coords(image2,center_point)
    print("centerpoint: ", center_point)
    #centerpoint = (798,277), (x,y)

    landing_t_point_list = []
    landing_c_point_list = []
    j=0
    for i in range(len(total_lenghts)):
        landing_point_t, _ = calc_fairway_width(fairway_coords, center_point[j%2], total_lenghts[i], image2.shape)
        landing_point_c, _ = calc_fairway_width(fairway_coords, center_point[j%2], carry_lenghts[i], image2.shape)
        landing_t_point_list.append(landing_point_t)
        landing_c_point_list.append(landing_point_c)
        j+=1
        # print("The total landing point is: ", landing_point_t)
        # print("The carry landing point is: ", landing_point_c)

    #px_length_cm = convert.get_px_side(image2.shape)  
    print("Landing ploints before e:",landing_t_point_list)
    image2=draw_elipse.draw_elipse_scratch_m(image2, landing_t_point_list[0],center_point, total_lenghts[0] ,1000)
    image2=draw_elipse.draw_elipse_scratch_f(image2, landing_t_point_list[1],center_point, total_lenghts[1] ,1000)
    image2=draw_elipse.draw_elipse_bogey_m(image2, landing_t_point_list[2],center_point, total_lenghts[2] ,1000)
    image2=draw_elipse.draw_elipse_bogey_f(image2, landing_t_point_list[3],center_point, total_lenghts[3] ,1000)

    for point in landing_t_point_list:
        print("points: ", point)
        bunker_dist, water_dist = distance.distance_to_objects(image2, point)
        if bunker_dist is not False:
            bunker_coords = extract_list(bunker_dist)
            for i in bunker_coords:
                print("i:", i)
                cv2.line(image2, point[0], i, (255,255,255), 1)
            print("Coordenates",bunker_coords)
            print("bunker dist: ", bunker_dist)
        #cv2.line(image2, point, )

    #bunker_coords = extract_list(bunker_distances)
    
    # for i in bunker_coords:
    #     cv2.line()


    figsize=(15,8)
    fig,ax=plt.subplots(figsize=figsize)
    #plt.plot(landing_point[0][0], landing_point[0][1], marker='v', color="white")

    cv2.imshow("image", image2)
    cv2.waitKey(0)
    cv2.destroyWindow()
if __name__ == "__main__":
    main()
            




