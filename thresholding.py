import cv2
import numpy as np
import constants
from guided_filter_implementation import *

def generate_means(cost_image, region_mask, threshold):
    # Get the background and foreground masks
    background_mask = (255.0 * (cost_image < threshold)).astype(np.uint8)
    background_mask = cv2.bitwise_and(background_mask, region_mask)/255.0

    foreground_mask = (255.0 * (cost_image >= threshold)).astype(np.uint8)
    foreground_mask = cv2.bitwise_and(foreground_mask, region_mask)/255.0

    # Get the updated mean values
    return np.mean(foreground_mask), np.mean(background_mask)

def perform_thresholding(images):
    cost_image = images["cost_image"]
    region_mask = images["virt_mask"]

    # Get the initial difference
    mean_val = np.mean(cost_image)
    t_old, t_new = generate_means(cost_image, region_mask, mean_val)
    while ((t_old - t_new) ** 2) > (constants.automatic_threshold) ** 2:
        # Update the values
        updated_fore, updated_back = generate_means(cost_image, region_mask, t_new)
        t_old = t_new
        t_new = (updated_fore + updated_back)/2.0

    _, threshold_image = cv2.threshold(cost_image, (t_old + t_new)/2.0, 255, cv2.THRESH_BINARY)
    images["thresholded_image"] = threshold_image

def perform_filtering(images):
    threshold_image = images["thresholded_image"]

    # Perform a closing operation
    kernel = np.ones(constants.closing_kernel_size, np.uint8)
    closing_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
    images["filtered_image"] = closing_image

    '''
    # Apply the guided filtering
    images["filtered_image"] = guided_filter(images["image_center"], closing_image, 
        constants.guided_window_radius, constants.guided_regulization)
    '''