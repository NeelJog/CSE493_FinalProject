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

if __name__ == "__main__":
    # Get the reader
    reader = None
    if constants.read_format == "dummy":
        reader = DummyReader()
    else:
        reader = CameraReader()

    print("Created reader and are going to start rendering")

    while True:
        try:

            while reader.has_next():
                start_time = time.time()
                images = dict(reader.get_next())
                # print("Images of type", type(images))

                if images is None:
                    print("Got none from get_next")
                    break

                for tranformation in tranforms_to_apply:
                    tranformation(images)

                cv2.imshow("Result", images["combined_image"])
                value = cv2.waitKey(1)

                if value == 113:
                    cv2.destroyAllWindows()
                    break
        
        except:
            continue
