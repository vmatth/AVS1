import numpy as np

# Calculates the dpi for the new image (as the new image is resized from the original 1600x900)
def get_dpi(image_shape):
    #print("-----Calculating dpi-----")
    #print("image shape", image_shape)
    ratio = 1600 / image_shape[1] #Calculates the ratio between this image's size and the original image's size which is (1600 x 900).
    dpi = 1200 / ratio # Calculate the dpi for the image.
    print("dpi: ", dpi)
    return dpi

# Converts pixels to meters.
def convert_px_to_m(pixels, image_shape): #axis=0 is width, axis=1 is height
    dpi_pixels = get_dpi(image_shape) #calculates the dpi using the new image shape
    scale = 1000
    ratio = pixels/dpi_pixels
    #print("ratio ", ratio)
    ten_inches_in_cm = 25.4
    cm_per_px = ratio * ten_inches_in_cm 
    cm = cm_per_px * scale
    #print("cm ", cm)
    m = cm / 100
    print("Converted from: ", pixels, " pixels to meters: ", m)
    return m

# Calculates the distance between 
def calculate_distance_between_two_points(a, b, image_shape):
    return convert_px_to_m(np.linalg.norm(a - b), image_shape)


