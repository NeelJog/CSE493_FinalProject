import cv2
import numpy as np
import constants


def get_probability_image(color_image, distance_image, region_mask):
    # Get the hue channel
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    image_hue = hsv_image[ : , : , 0]

    # Apply the mask to the distance
    depth_int_mask = cv2.bitwise_and((255 * distance_image).astype(np.uint8), region_mask)
    depth_mask = depth_int_mask/255.0

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
        return color_probs[bin_num]
    
    color_probability = np.vectorize(get_probability_value)(image_hue)
    color_probability_int = (255 * color_probability).astype(np.uint8)
    color_mask = cv2.bitwise_and(color_probability_int, region_mask)
    color_mask = color_mask/255.0

    return constants.color_weight * color_mask + constants.depth_weight * depth_mask

def get_histogram_prob_images(color_image, distance_image):

    foreground_mask = (distance_image == 1.0)
    foreground_mask = foreground_mask.astype(np.uint8) * 255

    background_mask = np.logical_and(distance_image > 0, distance_image < 1.0)
    background_mask = background_mask.astype(np.uint8) * 255

    cv2.imshow("Color image", color_image)
    cv2.imshow("Distance image", distance_image)
    cv2.imshow("Foreground mask", foreground_mask)
    cv2.imshow("Background mask", background_mask)

    foreground_prob = get_probability_image(color_image, distance_image, foreground_mask)
    background_prob = get_probability_image(color_image, 1.0 - distance_image, background_mask)
    print("Foreground", np.mean(foreground_prob), np.std(foreground_prob))
    print("Background", np.mean(background_prob), np.std(background_prob))
    cv2.imshow("Foreground probability", foreground_prob)
    cv2.imshow("Background probability", background_prob)

    return None, None
