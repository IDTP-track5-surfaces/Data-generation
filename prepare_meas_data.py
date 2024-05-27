import os
import cv2
import warnings
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MEAS_DIR = os.path.join(ROOT_DIR, "real_data")
DATA_DIR = os.path.join(MEAS_DIR, "data_sets")
IMAGE_DIR = os.path.join(MEAS_DIR, "photos")
LASER_DIR = os.path.join(MEAS_DIR, "laser")

def r(x,y):
    return np.sqrt(x**2 + y**2)

def puff_fit(r, a,b,c,d):
    # scaled logistic function describing surface deformation
    return d + (c) / (1 + np.exp(a - b * r**2))

def preprocess_image(file_loc):
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
            img = preprocess_image(img_loc)
            
            save_loc = os.path.join(save_location, pressure)
            os.makedirs(save_loc, exist_ok=True)
            
            file_name = os.path.join(save_loc, img_name)
            cv2.imwrite(file_name, img) 

def read_csv_laser_measurements(pressure_directory):
    depth_array = [pd.read_csv(os.path.join(pressure_directory, file), sep = ';', header=2).to_numpy() for file in os.listdir(pressure_directory)]
        
    # Equal shape
    shapes = [depth.shape[0] for depth in depth_array]
    depth_array = [depth[:np.min(shapes),:] for depth in depth_array]
    
    # Average measurements
    depth_array = np.mean(np.stack(depth_array, axis = -1), axis=-1)
    return depth_array

def preprocess_depth_array(depth_array, bounds = (-22, 28)):
    lb, ub = bounds
    max_radius = np.abs((lb-ub)/np.sqrt(2))
    
    data = depth_array[np.where(depth_array[:,0]>lb)]
    data = depth_array[np.where(depth_array[:,0]<ub)]

    data[:,0] = np.flip(np.abs(data[:,0] - ub))
    data[:,1] = np.flip(data[:,1])
    
    x_data = np.concatenate([-np.flip(data[:,0]), data[:,0]])
    y_data = np.concatenate([np.flip(data[:,1]), data[:,1]])
    
    # convert mm to m
    x_data /= 1000
    y_data /= 1000
    max_radius /= 1000
    
    return (x_data, y_data), max_radius

def parameter_fit(data, func=puff_fit, maxfev=50000):
    x_data, y_data = data
    parameters, covariance = curve_fit(func, x_data, y_data, maxfev=maxfev)
    return parameters, func

def create_depth(width, parameters, func=puff_fit):
    xy_range = np.linspace(-width, width, 128)
    X, Y = np.meshgrid(xy_range, xy_range)
    mesh = r(X, Y)
    # mesh /= 1000
    
    a,b,c,d = parameters
    depth = puff_fit(mesh,a,b,c,d)
    return depth
    
def make_depth_maps(width, save_location = DATA_DIR, laser_directory = LASER_DIR):

    
    for pressure in os.listdir(laser_directory):
        pressure_directory = os.path.join(laser_directory, pressure)
        depth_array = read_csv_laser_measurements(pressure_directory)
        data, max_radius = preprocess_depth_array(depth_array)
        parameters, func = parameter_fit(data)
        
        if width > max_radius:
            warnings.warn('width is larger than the maximum radius. Therefore, width=max_radius.') 
            width = max_radius
            
        depth = create_depth(width, parameters, func=func)
        file_loc = os.path.join(save_location, pressure, "depth.npy")
        np.save(file_loc, depth)
    
    
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    create_image_data()
    make_depth_maps(20e-3)