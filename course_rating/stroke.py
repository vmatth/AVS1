
import numpy as np
import math
from course_rating import convert

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
    if np.sum(np.array(class_)) == 0:
        return None
    else:
        intersection = []
        for point in class_:
            # Check if distance from the tee to the fairway is the same as the stroke length
            if int(np.linalg.norm(starting_coordinates_-point[0])) == int(stroke_dist):
                intersection.append(point)

        if len(intersection) == 0:
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
    if intersections == None:
        return np.array([0,0])
    else:
        return intersections[int(len(intersections)/2)][0]
    
# Calculates the fairway width at a given intersection
def get_fairway_width(intersections, image_shape, scale):
    # Get the first and last point in intersections
    edge_points1 = intersections[0][0]
    edge_points2 = intersections[-1][0]

    # Use the edgepoints to calculate the distance between them (that is calculating the fairway width)
    fairway_width = np.linalg.norm(edge_points1 - edge_points2)
    px_length_cm = convert.get_px_side(image_shape)
    #print("fairway px: ", fairway_width)
    fairway_width = convert.convert_px_to_m(px_length_cm, fairway_width, scale)
    
    #print(f"Fairway width: {fairway_width} m")
    return fairway_width, edge_points1, edge_points2

# returns the distance from landing point to hole (in metres)
def get_distance_landing_point_to_hole(starting_point, ending_point, image_shape, scale):
    distance = np.linalg.norm(starting_point-ending_point)
    px_length_cm = convert.get_px_side(image_shape)
    return int(convert.convert_px_to_m(px_length_cm, distance, scale))

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

        return landing_zone, fairway_width

    except:
        if len(intersection) == 0:
            return None, None

def extract_list(lst):
    return [item[0] for item in lst]




