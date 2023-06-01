import cv2
import constants
import numpy as np
import os

def load_mask():
    virtual_mask_path = "virtual_image_mask.png"
    binary_mask = cv2.imread(virtual_mask_path)

    red_channel = binary_mask[ : , : , 0]
    blue_channel = binary_mask[ : , : , 1]
    green_channel = binary_mask[ : , : , 2]

    red_black = red_channel < constants.color_black_threshold
    blue_black = blue_channel < constants.color_black_threshold
    green_black = green_channel < constants.color_black_threshold

    binary_mask = np.logical_and(red_black, blue_black)
    binary_mask = np.logical_and(blue_black, green_black)
    binary_mask = binary_mask.astype(np.uint8) * 255
    
    return binary_mask

def get_mask_loc_in_image(image, binary_mask):
    image_height, image_width, _ = image.shape
    mask_height, mask_width = binary_mask.shape

    start_y = int((image_height - mask_height)/2)
    end_y = start_y + mask_height
    start_x = int((image_width - mask_width)/2)
    end_x = start_x + mask_width

    return [start_y, end_y, start_x, end_x]

def generate_distance_image(real_depth_data):

    def get_distance_val(real_depth_val):
        return_val = 0.0

        if real_depth_val == 0.0:
            return_val =  0.0
        elif real_depth_val < constants.virtual_obj_depth_in_meters:
            return_val = 255.0
        else:
            return_val = 125.0
            x_p = (constants.virtual_obj_depth_in_meters)/(real_depth_val + constants.epsilon)
            return_val = ((constants.distance_constant_val ** x_p) - 1)/(constants.distance_constant_val - 1)
        
        return return_val

    distance_image = np.vectorize(get_distance_val)(real_depth_data)

    return distance_image

def runner():
    frames_dir = "sample_frames"
    depth_dir = "sample_depth_frames"

    for index in range(3, 10):
        image_file_path = os.path.join(frames_dir, str(index) + ".png")
        depth_file_path = os.path.join(depth_dir, str(index) + ".txt")

        curr_frame = cv2.imread(image_file_path)
        depth_data = np.loadtxt(depth_file_path, delimiter=',')
        depth_data = depth_data
        print("Mean depth", np.mean(depth_data), "Std dev", np.std(depth_data))

        mask = load_mask()
        mask_loc_in_image = get_mask_loc_in_image(curr_frame, mask)
        image_mask_depth = depth_data[ mask_loc_in_image[0] : mask_loc_in_image[1], mask_loc_in_image[2] : mask_loc_in_image[3] ]
        
        print("Mask depth data - Mean:", np.mean(image_mask_depth), "Std dev:", np.std(image_mask_depth))
        print("Virtual object depth: ", constants.virtual_obj_depth_in_meters)
        
        distance_image = generate_distance_image(image_mask_depth)

        cv2.imshow("Current Frame", curr_frame)
        cv2.imshow("Binary Mask", mask)
        cv2.imshow("Distance Image", distance_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    runner()