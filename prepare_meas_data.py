import os
import cv2
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MEAS_DIR = os.path.join(ROOT_DIR, "real_data")
DATA_DIR = os.path.join(MEAS_DIR, "data_sets")
IMAGE_DIR = os.path.join(MEAS_DIR, "photos")
LASER_DIR = os.path.join(MEAS_DIR, "laser")

def r(x,y):
    return np.sqrt(x**2 + y**2)

def preproces_image(file_loc):
    image = cv2.imread(file_loc)
    h,w,_ = image.shape
    w_m = int(12*w/(12+20))

    # Make image square w.r.t. the center
    image_croped = image[:,w_m-int(h/2):w_m+int(h/2),:]

    # Define points in the perspective image
    pts_src = np.array([[400, 97], [578, 97], [400, 250], [578, 250]], dtype='float32')

    # Define where these points should map to in the orthographic view
    pts_dst = np.array([[402, 97], [575, 97], [402, 248], [575, 248]], dtype='float32')

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Apply the perspective transformation
    height, width = 720, 720  # size of the output image
    orthographic_image = cv2.warpPerspective(image_croped, M, (width, height))

    # Center deformation
    orthographic_image = orthographic_image[0:-145, 72:647, :]

    scaled_image = cv2.resize(orthographic_image, (128, 128))
    return scaled_image

def create_image_data(save_location = DATA_DIR, image_directory = IMAGE_DIR):
    for pressure in os.listdir(image_directory):
        pressure_directory = os.path.join(image_directory, pressure)
        
        for img_name in os.listdir(pressure_directory):
            img_loc = os.path.join(pressure_directory, img_name)
            img = preproces_image(img_loc)
            
            save_loc = os.path.join(save_location, pressure)
            os.makedirs(save_loc, exist_ok=True)
            
            file_name = os.path.join(save_loc, img_name)
            cv2.imwrite(file_name, img) 

def read_csv_laser_measurements(pressure_directory):
    depth_array = [pd.read_csv(os.path.join(LASER_DIR, pressure, file), sep = ';', header=2).to_numpy() for file in pressure_directory]
        
    # Equal shape
    shapes = [depth.shape[0] for depth in depth_array]
    depth_array = [depth[:np.min(shapes),:] for depth in depth_array]
    
    # Average measurements
    depth_array = np.mean(np.stack(depth_array, axis = -1), axis=-1)
    return depth_array

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    create_image_data()
    
    for pressure in os.listdir(LASER_DIR):
        pressure_directory = os.listdir(os.path.join(LASER_DIR, pressure))
        depth_array = read_csv_laser_measurements(pressure_directory)
        
        
            
            


    # preproces_image()