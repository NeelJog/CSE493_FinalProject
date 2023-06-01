import cv2
import numpy as np


def get_color_histogram(color_image):
    histogram = cv2.calcHist([color_image], [0], None, [180], [0, 180])
    return histogram


def get_histograms(color_image, distance_image):
    # First convert to hsv
    as_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    color_img_hue = as_hsv[:, :, 0]
    print(color_img_hue.shape)
    foreground = (distance_image == 1.0).astype(np.uint8) * 255
    background = (distance_image < 1.0).astype(np.uint8) * 255
    print(foreground.shape)
    foreground_pixels = cv2.bitwise_and(color_img_hue, foreground)
    background_pixels = cv2.bitwise_and(color_img_hue, background)
    foreground_hist = get_color_histogram(foreground_pixels)
    background_hist = get_color_histogram(background_pixels)
    return foreground_hist, background_hist
