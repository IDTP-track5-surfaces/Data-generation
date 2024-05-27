import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from scipy.optimize import curve_fit

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MEAS_DIR = os.path.join(ROOT_DIR, "real_data")
DATA_DIR = os.path.join(MEAS_DIR, "data_sets")
IMAGE_DIR = os.path.join(MEAS_DIR, "photos")

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

    orthographic_image = orthographic_image[0:-145, 72:647, :]

    scaled_image = cv2.resize(orthographic_image, (128, 128))
    return scaled_image

def create_image_data():
    for pressure in os.listdir(IMAGE_DIR):
        for img_name in os.listdir(os.path.join(IMAGE_DIR, pressure)):
            img_loc = os.path.join(IMAGE_DIR, pressure, img_name)
            img = preproces_image(img_loc)
            
            save_loc = os.path.join(DATA_DIR, pressure)
            os.makedirs(save_loc, exist_ok=True)
            
            file_name = os.path.join(save_loc, img_name)
            cv2.imwrite(file_name, img) 
    
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    create_image_data()
    

    # preproces_image()