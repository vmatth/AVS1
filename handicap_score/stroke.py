
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

# Inputs the total length and carry length of a stroke in yards
# Returns them in pixels
# Total Length in Yards: scratch_m: 250y, scratch_F: 210y, bogey_m: 200y, bogey_f: 150y
# Carry Length in Yards: scratch_m: 230y, scratch_F: 190y, bogey_m: 180y, bogey_f: 130y
def get_stroke_lengths(image_shape, total_length, carry_length, scale):

    px_length_cm = convert.get_px_side(image_shape)

    total_length_m = convert.convert_yards_to_m(total_length) #Convert from yard to meter
    total_length_px = int(convert.convert_m_to_px(px_length_cm, total_length_m, scale)) #Convert from meter to px


    carry_length_m = convert.convert_yards_to_m(carry_length) 
    carry_length_px = int(convert.convert_m_to_px(px_length_cm, carry_length_m, scale))

    return total_length_px, carry_length_px

def get_stroke_lengths_old(image_shape):
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

# returns a list of points  in a class that the stroke intersects (e.g intersection between fairway class and the stroke distance)
def get_intersections(class_, starting_coordinates_, stroke_dist):
    intersection = []
    for point in class_:
        # Check if distance from the tee to the fairway is the same as the stroke length
        if int(np.linalg.norm(starting_coordinates_-point[0])) == int(stroke_dist):
            intersection.append(point)

    if len(intersection) == 0:
        print("Could not calculate a landing point")
        return None

    return intersection

# Returns the intersections that are closer to the endpoint (e.g the green)
# This is because get_intersections can find intersections on both sides of the fairway (as it draws a circle)
def get_shortest_intersections(intersections, start_point, end_point):
    shortest_intersection = []
    distance = np.linalg.norm(np.array(start_point) - np.array(end_point))
    for coords in intersections:
        distance_from_points = np.linalg.norm(coords - end_point)
        if distance > distance_from_points:
            shortest_intersection.append(coords)
    return shortest_intersection
    
# Returns the landing point which lies at the middle of the intersection
def get_landing_point(intersections):
    return intersections[int(len(intersections)/2)][0]
    
# Calculates the fairway width at a given intersection
def get_fairway_width(intersections, image_shape, scale):
    # Get the first and last point in intersections
    edge_points1 = intersections[0][0]
    edge_points2 = intersections[-1][0]

    # Use the edgepoints to calculate the distance between them (that is calculating the fairway width)
    fairway_width = np.linalg.norm(edge_points1 - edge_points2)
    px_length_cm = convert.get_px_side(image_shape)
    print("fairway px: ", fairway_width)
    fairway_width = convert.convert_px_to_m(px_length_cm, fairway_width, scale)
    
    print(f"Fairway width: {fairway_width} m")
    return fairway_width

# returns the distance from landing point to hole (in metres)
def get_distance_landing_point_to_hole(starting_point, ending_point, image_shape, scale):
    distance = np.linalg.norm(starting_point-ending_point)
    px_length_cm = convert.get_px_side(image_shape)
    return convert.convert_px_to_m(px_length_cm, distance, scale) 

# returns a list of points where the ball will land for each stroke distance
# with the corresponding fairway width at that landing point.
def calc_fairway_width_old(class_, centerpoint, stroke_dist, image_shape):
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
    image = cv2.imread('C:\\Users\\jespe\\Desktop\\AVS1\\for_jacobo1.png')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print("img shape: ", image.shape) # (256,416,3)
    #ori_image=cv2.resize(ori_image,(800,450))
    image2=cv2.resize(image,(800,450))
    scale = 1250

    # Get stroke lenghts in meters
    total_s_m, carry_s_m = get_stroke_lengths(image2.shape, 250, 230, scale)
    #print("stroke_lenghts: ", total_lenghts)

    # Get Class coordinates
    _, _, fairway, _, _ = get_class_coords(image2)
    fairway_coords = cv2.findNonZero(fairway)

    #Click the tee
    center_point=[]
    center_point=get_click_coords(image2,center_point)
    print("centerpoint: ", center_point)
    #centerpoint = (798,277), (x,y)
    
    #Get green sizes
    green_length, green_width, green_centerpoint = get_green_size(image, color='unet', scale=scale)

    point = center_point[0]
    counter = 1
    previous_distance_to_green = 999
    total_distance = 0
    while(True):
        print(f"stroke: {counter}")
        intersections = get_intersections(fairway_coords, point, total_s_m)

        if intersections:
            for stupidPoint in intersections:
                cv2.circle(image2, stupidPoint[0], 1, (255, 255, 255))

        # Get the shortest intersections in cases where there are multiple intersections with the fairway
        if intersections:
            intersections = get_shortest_intersections(intersections, point, green_centerpoint)

        if intersections:
            for stupidPoint in intersections:
                cv2.circle(image2, stupidPoint[0], 3, (0, 255, 255))

        if intersections: # Make sure there are intersections with the fairway

            landing_point = get_landing_point(intersections)
            distance_to_green = get_distance_landing_point_to_hole(landing_point, green_centerpoint, image2.shape, scale)
            print("Distance to green: ", distance_to_green)
            fairway_width = get_fairway_width(intersections, image2.shape, scale)

            total_distance += distance.distance_two_points(point, landing_point, image2.shape, scale)
            print("Total distance [m]: ", total_distance)
            
            cv2.circle(image2, landing_point, 1, (255, 255, 0))

            point = landing_point #Refresh the point to be the new landingpoint

        else:
            distance_to_green = get_distance_landing_point_to_hole(landing_point, green_centerpoint, image2.shape, scale)
            total_distance = distance_to_green + total_distance
            print(f"Total distance of the hole: {total_distance} [m]")
            break


        counter += 1

        cv2.imshow("image", image2)
        cv2.waitKey(0)

    # landing_t_point_list = []
    # landing_c_point_list = []
    # j=0
    # for i in range(len(total_lenghts)):
    #     landing_point_t, _ = calc_fairway_width(fairway_coords, center_point[j%2], total_lenghts[i], image2.shape)
    #     landing_point_c, _ = calc_fairway_width(fairway_coords, center_point[j%2], carry_lenghts[i], image2.shape)
    #     landing_t_point_list.append(landing_point_t)
    #     landing_c_point_list.append(landing_point_c)
    #     j+=1
    #     # print("The total landing point is: ", landing_point_t)
    #     # print("The carry landing point is: ", landing_point_c)

    # #px_length_cm = convert.get_px_side(image2.shape)  
    # print("Landing ploints before e:",landing_t_point_list)
    # image2=draw_elipse.draw_elipse_scratch_m(image2, landing_t_point_list[0],center_point, total_lenghts[0] ,1000)
    # image2=draw_elipse.draw_elipse_scratch_f(image2, landing_t_point_list[1],center_point, total_lenghts[1] ,1000)
    # image2=draw_elipse.draw_elipse_bogey_m(image2, landing_t_point_list[2],center_point, total_lenghts[2] ,1000)
    # image2=draw_elipse.draw_elipse_bogey_f(image2, landing_t_point_list[3],center_point, total_lenghts[3] ,1000)

    # for point in landing_t_point_list:
    #     print("points: ", point)
    #     bunker_dist, water_dist = distance.distance_to_objects(image2, point)
        
    #     if bunker_dist is not False:
    #         bunker_coords = extract_list(bunker_dist)
    #         for i in bunker_coords:
    #             print("i:", i)
    #             cv2.line(image2, point[0], i, (255,255,255), 1)
    #         print("Coordenates",bunker_coords)
    #         print("bunker dist: ", bunker_dist)
    #     #cv2.line(image2, point, )


    #Get green sizes
    #green_length, green_width, green_centerpoint = get_green_size(image, color='unet', scale=1000)

    print("hello im here")


    figsize=(15,8)
    fig,ax=plt.subplots(figsize=figsize)
    #plt.plot(landing_point[0][0], landing_point[0][1], marker='v', color="white")

    cv2.imshow("image", image2)
    cv2.waitKey(0)
    cv2.destroyWindow()
if __name__ == "__main__":
    main()
            




