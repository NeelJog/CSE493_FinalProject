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
import time

from distance_image_generator import *
from histograms import *
from reader import *
from cost_calculator import *
from thresholding import *
from trimap_generator import *
from alpha_matting import *
from composition import *

tranforms_to_apply = [generate_distance_image, get_histogram_prob_images, 
    perform_cost_calculation, perform_thresholding, perform_filtering, perform_trimap,
    perform_alpha_matting, perform_composition]
keys_to_ignore = ["virt_center_coordinates"]

def visualize_output(images):
    for key, value in images.items():
        print(key, value.shape, np.mean(value), np.std(value))
        '''
        if key not in keys_to_ignore:
            cv2.imshow(key, value)
        '''

    exit = False
    cv2.imshow("Output Image", images["combined_image"])
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit = True
    
    return exit

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
                start_time = time.time()
                tranformation(images)
                time_taken = (time.time() - start_time)
                print("Transformation", str(tranformation), "takes", time_taken, "secs")
            
            if visualize_output(images):
                break

    finally:
        reader.finish()
    
    cv2.destroyAllWindows()
