import cv2
import numpy as np
import constants

def single_frame_cost(foreground_prop, background_prop):
    return (foreground_prop)/(foreground_prop + background_prop + constants.epsilon)

def multi_frame_cost(foreground_prop, background_prop):
    return None

def perform_cost_calculation(images):
    foreground_prop = images["foreground_prob"]
    background_prop = images["background_prob"]

    # Get the cost image
    cost_image = None
    if constants.cost_method == "single":
        cost_image = single_frame_cost(foreground_prop, background_prop)
    else:
        cost_image = multi_frame_cost(foreground_prop, background_prop)
        
    images["cost_image"] = cost_image