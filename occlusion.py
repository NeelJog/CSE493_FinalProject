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
keys_to_show = ["combined_image"]

def visualize_output(images):
    save_dir = "images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key, value in images.items():
        if key in keys_to_show:
            cv2.imshow(key, value)

    exit = False
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
    
    try:
        timing_values = {}

        while reader.has_next():
            start_time = time.time()
            images = reader.get_next()

            if images is None:
                print("Got none from get_next")
                break
            
            for tranformation in tranforms_to_apply:
                trans_name = str(tranformation)

                start_time = time.time()
                tranformation(images)
                trans_time = time.time() - start_time

                # Record the time
                if trans_name not in timing_values:
                    timing_values[trans_name] = []
                timing_values[trans_name].append(trans_time)
            
            if visualize_output(images):
                break
        
        cv2.destroyAllWindows()
        for key in timing_values:
            values = np.array(timing_values[key])
            print(key, np.mean(values), np.std(values))

    finally:
        reader.finish()
    
    
