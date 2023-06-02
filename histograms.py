import cv2
import numpy as np
import constants


def get_probability_image(color_image, distance_image, region_mask):
    # Get the hue channel and the depth mask
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    image_hue = hsv_image[ : , : , 0]

    # Make the depth mask
    depth_mask = (255.0 * distance_image).astype(np.uint8)
    depth_mask = cv2.bitwise_and(depth_mask, region_mask)
    depth_mask = depth_mask/255.0

    # Get the normalized histogram
    total_pixels = cv2.countNonZero(region_mask)
    color_probs = cv2.calcHist([image_hue], [0], region_mask, 
        [constants.color_histogram_bins], [0, constants.max_hue_value])
    color_probs /= 1.0 * total_pixels

    # Get the probability using color
    def get_probability_value(color_value):
        nonlocal color_probs
        if color_value == 0:
            return 0.0

        bin_num = int((constants.color_histogram_bins * color_value)/constants.max_hue_value)
        return color_probs[bin_num][0]
    
    # Get the color mask
    color_probability = np.vectorize(get_probability_value)(image_hue)
    color_mask = (255.0 * color_probability).astype(np.uint8)
    color_mask = cv2.bitwise_and(color_mask, region_mask)
    color_mask = color_mask/255.0

    # Perform the merging
    return constants.color_weight * color_mask + (1.0 - constants.color_weight) * depth_mask

def get_histogram_prob_images(images):
    distance_image = images["distance_image"]
    color_image = images["image_center"]
    
    # Make the foreground and background mask
    foreground_mask = (distance_image == 1.0)
    foreground_mask = foreground_mask.astype(np.uint8) * 255

    background_mask = np.logical_and(distance_image > 0, distance_image < 1.0)
    background_mask = background_mask.astype(np.uint8) * 255

    # Make the probability image
    images["foreground_prob"] = get_probability_image(color_image, distance_image, foreground_mask)
    images["background_prob"] = get_probability_image(color_image, 1.0 - distance_image, background_mask)
