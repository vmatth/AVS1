#This python file calculates the size of a given class in an image.
#This is done by calculating the amount of pixels in the blob
import cv2
import numpy as np
from get_classes import get_class_coords
import convert

def get_green_size(image):
    _, _, _, green, _ = get_class_coords(image)
    size = np.sum(green == 255)
    print("shape", green.shape)
    cv2.imshow("green", green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Green size: ", size)
    return size


test_path = "C:\\Users\\Vini\\Desktop\\for_jacobo.png"
#416, 256
i = cv2.imread(test_path)
resized = cv2.resize(i, (400,225), interpolation = cv2.INTER_AREA)
get_green_size(resized)

convert.convert_px_to_m(400, resized.shape)
convert.calculate_distance_between_two_points(np.array([0,0]), np.array([400,400]), resized.shape)





