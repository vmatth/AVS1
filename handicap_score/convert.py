import numpy as np
import math

def get_px_side(image_shape):
    one_px = 2.54 / 120 # dpi: 120 px on 1 inch (2.54 cm)
    og_area = one_px ** 2 * 1600 * 900 # original width: 1600, original height: 900
    new_area_px = image_shape[0] * image_shape[1]
    new_one_px = math.sqrt(og_area / new_area_px) # The length of the pixel side in cm
    return new_one_px

# Converts pixels to meters.
def convert_px_to_m(px_size, px_num, scale=1000):
    return px_size * px_num * scale / 100

def convert_to_m2(px_size, px_num, scale=1000):
    return px_size**2 * px_num * scale**2 / (100**2)

# Calculates the distanconvert_to_m2ce between 
def calculate_distance_between_two_points(a, b, image_shape):
    return convert_px_to_m(np.linalg.norm(a - b), image_shape)
