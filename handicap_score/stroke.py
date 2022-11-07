
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def scracth_m(scale):
    dpi_scale = 1200
    stroke_dist = 250
    stroke_dist_cm=stroke_dist*100
    stroke_descale=stroke_dist_cm/(scale*2)
    stroke_dist_s_m_px=stroke_descale*dpi_scale/25.4
    print("stroke: ", stroke_dist_s_m_px)

    return stroke_dist_s_m_px
    

def scracth_f(scale):
    dpi_scale = 1200
    stroke_dist = 210
    stroke_dist_cm=stroke_dist*100
    stroke_descale=stroke_dist_cm/(scale*2)
    stroke_dist_s_f_px=stroke_descale*dpi_scale/25.4
    print("stroke: ", stroke_dist_s_f_px)

def bogey_m(scale):
    dpi_scale = 1200
    stroke_dist = 200
    stroke_dist_cm=stroke_dist*100
    stroke_descale=stroke_dist_cm/(scale*2)
    stroke_dist_b_m_px=stroke_descale*dpi_scale/25.4
    print("stroke: ", stroke_dist_b_m_px)


def bogey_f(scale):
    dpi_scale = 1200
    stroke_dist = 150
    stroke_dist_cm=stroke_dist*100
    stroke_descale=stroke_dist_cm/(scale*2)
    stroke_dist_b_f_px=stroke_descale*dpi_scale/25.4
    print("stroke: ", stroke_dist_b_f_px)

def get_class_cords(image):
    #fairway_list, green_list, tee_list, bunker_list, water_list = ([] for _ in range(5))
    
    #r g b -> b g r
    #colors of each class
    colors_rgb= {
        'tee': np.array([0, 36, 250]), #red
        'water': np.array([231, 200, 46]), #blue
        'bunker': np.array([158, 246, 246]), #yellow
        'fairway': np.array([77, 156, 77]), #dark green
        'green': np.array([122, 243, 142]) #light green
    }
    
    #mask for each class
    mask = {}
    for item in colors_rgb:
        mask.update({item : cv2.inRange(image, colors_rgb[item], colors_rgb[item])})
    
    '''
    for y , value_y in enumerate(image):
        for x, value_x in enumerate(value_y):
            if value_x[0]==0 and value_x[1]==140 and value_x[2]==0:
                fairway_list.append((x,y))
            if value_x[0]==0 and value_x[1]==255 and value_x[2]==0:
                green_list.append((x,y))
            if value_x[0]==255 and value_x[1]==0 and value_x[2]==0:
                tee_list.append((x,y))
            if value_x[0]==217 and value_x[1]==230 and value_x[2]==122:
                bunker_list.append((x,y))
            if value_x[0]==7 and value_x[1]==15 and value_x[2]==247:
                water_list.append((x,y))
    '''
    # print("fair: ", len(fairway_list))
    # print("green: ", len(green_list))
    # print("tee: ", len(tee_list))
    # print("bunker: ", len(bunker_list))
    # print("water: ", len(water_list))
    return mask['tee'], mask['water'], mask['fairway'], mask['green'], mask['bunker']


def draw_elipse(image, laning_zone, stroke_dist):
    
    
    
    
    pass

def calc_intersection(class_, centerpoint, stroke_dist):
    intersection = []
    print("centerpoint: ", centerpoint[0])
    print("storke dist: ", stroke_dist)
    for point in class_:
        #print("points:", point)
        if (math.sqrt((centerpoint[0]- point[0])**2+(centerpoint[1]-point[1])**2)) == int(stroke_dist):
            intersection.append(point)
            print("HHEHEHEHHEHE")
    print("intersection: ", intersection)
    return intersection 

def main():
    image = cv2.imread('for_jacobo.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)


    # (256,416,3)
    image2=cv2.resize(image,(800,450))
    plt.imshow(image2)


    #transform meters to pixels     
    stroke_dist_s_m= scracth_m(1000)
    print(stroke_dist_s_m)
    
    #c1= plt.Circle((1400,555),scratch_s_px,color='w',fill=False )

    fairway,green, tees, bunkers, waters = get_class_cords(image2)
    print("fairway: ", fairway)

    #fairway2 = [(x,y) for y , value_y in enumerate(image2) for x, value in enumerate(value_y) if  value[0]==0 and value[1]==140 and value[2]==0]
    # intersection_fairway=[ point for point in fairway if  int(math.sqrt((700- point[0])**2+(277-point[1])**2)) == int(distance)]
    # print(len(intersection_fairway))
    centerpoint = (798,277)
    intersection_fairway = calc_intersection(fairway, centerpoint, stroke_dist_s_m)

    edge_points=[]
    edge_points.append(intersection_fairway[0])
    edge_points.append(intersection_fairway[-1])
    landing_point= intersection_fairway[int(len(intersection_fairway)/2)]
    print("The edge points are: ",edge_points)
    print("The landing point is: ",landing_point)

    

if __name__ == "__main__":
    main()
            




