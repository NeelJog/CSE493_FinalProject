import constants
import cv2
import numpy as np

def perform_composition(images):
    virt_image = images["virt_image"]
    virt_mask = images["virt_mask"]
    image_center = images["image_center"]
    trimap = images["trimap_image"]
    alpha_mat = images["alpha_mat"]

    rows, cols, _ = image_center.shape
    composed_image = np.zeros((rows, cols, 3), dtype = np.uint8)
    for y_val in range(rows):
        for x_val in range(cols):
            image_val = None

            # Determine the image val
            real_val, virtual_val = image_center[y_val, x_val, : ], virt_image[y_val, x_val, : ]
            if virt_mask[y_val, x_val] <= 5.0 or trimap[y_val, x_val] == 1.0:
                image_val = real_val
            elif trimap[y_val, x_val] == 0.0:
                image_val = virtual_val
            else:
                alpha_val = alpha_mat[y_val, x_val]
                image_val = alpha_val * real_val + (1 - alpha_val) * virtual_val
            
            composed_image[y_val, x_val, : ] = image_val

    # Combine this with the whole image
    center_location = images["virt_center_coordinates"]
    combined_image = images["real_image"].copy()
    combined_image[ center_location[0] : center_location[1], center_location[2] : center_location[3] ] = composed_image

    images["combined_image"] = combined_image