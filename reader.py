import pyrealsense2 as rs
import numpy as np
import cv2
import constants
import os

class ImageReader:

    def __init__(self):
        self.read_virt_image()
        self.virt_center_coordinates = None
    
    def read_virt_image(self):
        # Read in the files
        original_image = cv2.imread("virtual_image.png")
        virt_mask = cv2.imread("virtual_image_mask.png")
        virt_mask = 255 - cv2.cvtColor(virt_mask, cv2.COLOR_BGR2GRAY)
        print("Virtual values", np.mean(virt_mask), np.std(virt_mask))
        
        # Crop the original image to create the virtual image
        image_height, image_width, _ = original_image.shape
        mask_height, mask_width = virt_mask.shape
        start_y, start_x = int((image_height - mask_height)/2), int((image_width - mask_width)/2)
        end_y, end_x = start_y + mask_height, start_x + mask_width
        
        # Save the virtual mask
        self.virt_image = original_image[start_y : end_y, start_x : end_x , : ]
        self.virt_mask = virt_mask
    
    def has_next(self):
        return True
    
    def get_next(self):
        return None
    
    def finish(self):
        pass
    
    def add_in_virtual_data(self, images):
        real_image = images["real_image"]
        depth_image = images["depth_image"]

        if self.virt_center_coordinates is None:
            # Get the coordinates to place the ball in the video
            real_height, real_width, _ = real_image.shape
            virt_height, virt_width = self.virt_mask.shape
            start_y, start_x = int((real_height - virt_height)/2), int((real_width - virt_width)/2)
            end_y, end_x = start_y + virt_height, start_x + virt_width
            self.virt_center_coordinates = np.array([start_y, end_y, start_x, end_x])
        
        images["virt_center_coordinates"] = self.virt_center_coordinates
        images["virt_image"] = self.virt_image
        images["virt_mask"] = self.virt_mask

        coords = self.virt_center_coordinates
        images["image_center"] = real_image[  coords[0] : coords[1], coords[2] : coords[3]]
        images["depth_center"] = depth_image[ coords[0] : coords[1], coords[2] : coords[3] ]

class CameraReader(ImageReader):

    def __init__(self):
        super().__init__()
        self.setup()
    
    def setup(self):
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
        
        # Get the depth information
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        align_to = rs.stream.color
        align = rs.align(align_to)

        self.pipeline = pipeline
        self.config = config
        self.align = align
        self.depth_scale = depth_sensor.get_depth_scale()

    def has_next(self):
        return True
    
    def get_next(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return None
        
        # Get the images inside the frame
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data()) * self.depth_scale

        # Store the depth and image data
        images = {}
        images["real_image"] = color_image
        images["depth_image"] = depth_image
        self.add_in_virtual_data(images)

        return images
    
    def finish(self):
        self.pipeline.stop()


class DummyReader(ImageReader):

    def __init__(self):
        super().__init__()
        self.counter = 3
        self.frames_dir = "sample_frames"
        self.depth_dir = "sample_depth_frames"
    
    def has_next(self):
        return self.counter < 10
    
    def get_next(self):
        if not self.has_next():
            return None
        
        # Determine the file paths
        image_file_path = os.path.join(self.frames_dir, str(self.counter) + ".png")
        depth_file_path = os.path.join(self.depth_dir, str(self.counter) + ".txt")

        # Store the depth and image data
        images = {}
        images["real_image"] = cv2.imread(image_file_path)
        images["depth_image"] = np.loadtxt(depth_file_path, delimiter=',')
        self.add_in_virtual_data(images)
        
        self.counter += 1
        return images



if __name__ == "__main__":
    reader = DummyReader()