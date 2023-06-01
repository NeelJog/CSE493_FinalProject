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
# Get the distance image
from distance_image_generator import *
# Import the histagram stuff
from histograms import *


def prepare_images(virt_img):
    virt_img_mask = np.copy(virt_img)
    ht, wdth, dpth = virt_img.shape

    coords = [-1, 1000, -1, 1000]
    for height in range(ht):
        for width in range(wdth):
            r, g, b = virt_img[height, width]
            if not(r == 255 and g == 255 and b == 255):
                virt_img_mask[height, width] = (0, 0, 0)
                coords[0] = max(height, coords[0])
                coords[1] = min(height, coords[1])
                coords[2] = max(width, coords[2])
                coords[3] = min(width, coords[3])

    return virt_img_mask[coords[1]: coords[0], coords[3]: coords[2]]


def setup():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    return pipeline, config


def resize_img(color_img):
    # resize the virtual object to fit in the color image
    virt_image = cv2.imread("virtual_image.png")
    side = int(color_image.shape[0] * constants.virt_obj_scale_factor)

    virt_image = cv2.resize(virt_image, (side, side))
    virt_image_mask = prepare_images(virt_image)
    cv2.imwrite("virtual_image.png", virt_image)
    cv2.imwrite("virtual_image_mask.png", virt_image_mask)
    return virt_image, virt_image_mask


if __name__ == "__main__":
    # Get the configuration and pipeline
    pipeline, config = setup()

    # Start streaming
    profile = pipeline.start(config)

    # Get the depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    mask = load_mask()
    mask_loc_in_image = None

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            # a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            # Get the images inside the frame
            depth_info = np.asanyarray(aligned_depth_frame.get_data())
            depth_info_in_meters = depth_info / 1000
            color_image = np.asanyarray(color_frame.get_data())

            if mask_loc_in_image is None:
                mask_loc_in_image = get_mask_loc_in_image(color_image, mask)

            image_mask_depth = depth_info_in_meters[ mask_loc_in_image[0] : mask_loc_in_image[1], mask_loc_in_image[2] : mask_loc_in_image[3] ]
            distance_image = generate_distance_image(image_mask_depth)
            
            color_image_needed = color_image[mask_loc_in_image[0]: mask_loc_in_image[1], mask_loc_in_image[2]:mask_loc_in_image[3]]
            get_Rf_Rb(color_image_needed, distance_image)

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

            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()
