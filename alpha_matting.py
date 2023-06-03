from pymatting import *
import constants
import numpy as np
import cv2

def perform_alpha_matting(images):
    image = images["image_center"].astype(np.float64)/255.0
    trimap = images["trimap_image"].astype(np.float64)

    # Get the alpha mat
    alpha_mat = estimate_alpha_knn(image, trimap, 
        laplacian_kwargs={"n_neighbors": constants.alpha_neighbors},
        cg_kwargs={"maxiter": constants.alpha_max_iter})
    
    # Smooth the mat
    images["alpha_mat"] = cv2.GaussianBlur(alpha_mat.astype(np.float32), constants.gaussian_blur_kernel, 0)