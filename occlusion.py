# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
#              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Constants used
import constants

from distance_image_generator import *
from histograms import *
from reader import *
from cost_calculator import *
from thresholding import *

tranforms_to_apply = [generate_distance_image, get_histogram_prob_images, 
    perform_cost_calculation, perform_thresholding, perform_filtering]
keys_to_ignore = ["virt_center_coordinates"]

def visualize_output(images):
    for key, value in images.items():
        if key in keys_to_ignore:
            continue
            
        cv2.imshow(key, value)

if __name__ == "__main__":
    # Get the reader
    reader = None
    if constants.read_format == "dummy":
        reader = DummyReader()
    else:
        reader = CameraReader()
    
    print("Finished setting up reader of type", constants.read_format)

    try:
        while reader.has_next():
            images = reader.get_next()

            if images is None:
                print("Got none from get_next")
                break

            for tranformation in tranforms_to_apply:
                tranformation(images)
            
            visualize_output(images)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

            '''
            # Resize the mask to fit in the color image
            virt_image, virt_image_mask = resize_img(color_image)

            # Crop out the central part of the color image
            a_h, a_w, a_d = color_image.shape
            virt_h, virt_w, virt_d = virt_image.shape
            start_x = int((a_w - virt_w) / 2)
            start_y = int((a_h - virt_h) / 2)
            end_x = int(start_x + virt_w)
            end_y = int(start_y + virt_h)

            # See where we need to borrow from color image
            blended = np.copy(color_image)
            blended[start_y:end_y, start_x:end_x, :] = virt_image[:, :, :]
            height, width, _ = blended.shape
            for h in range(start_y, end_y):
                for w in range(start_x, end_x):
                    r, g, b = blended[h, w]
                    depth = depth_info_in_meters[h, w]
                    v_o_depth = constants.virtual_obj_depth_in_meters
                    if (r == 255 & g == 255 & b == 255) | (depth < v_o_depth):
                        blended[h, w] = color_image[h, w]
            '''
    finally:
        reader.finish()
    
    cv2.destroyAllWindows()
