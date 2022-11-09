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

# Converting px numbers to m2 
def convert_to_m2(px_size, px_num, scale=1000):
    return px_size**2 * px_num * scale**2 / (100**2)

# Converting meters to px number
def convert_m_to_px(px_size, distance, scale=1000):
    return distance * 100 / scale / px_size

# Converts yards to meters.
def convert_yards_to_m(yards):
    return yards * 0.9144