import os
import numpy as np
from utils import *
from scipy.interpolate import RectBivariateSpline
from imageio.v2 import imwrite, imread

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATTERN_DIR = os.path.join(ROOT_DIR, "reference_patterns")
DATA_SETS_DIR = os.path.join(ROOT_DIR, "data_sets")

def create_directory_structure():
    if not os.path.exists(DATA_SETS_DIR):
        # for i in ["train", "test", "validation"]:
        os.makedirs(os.path.join(DATA_SETS_DIR, "depth"))
        os.makedirs(os.path.join(DATA_SETS_DIR, "normal"))
        for j in ["reflection", "refraction"]:
            os.makedirs(os.path.join(DATA_SETS_DIR, "warp_map", j))
            os.makedirs(os.path.join(DATA_SETS_DIR, "warped_image", j))
            
            
def create_depth_and_normal_maps(w=128, fps = 24):
    x = np.linspace(-52e-3,52e-3, w); # in meter
    y = np.linspace(-52e-3,52e-3, w); # in meter
    X, Y = np.meshgrid(x, y)
    seq = 0
    
    # for dx in np.linspace(-13e-3,13-3, 10):
    #     for dy in np.linspace(-13e-3,13-3, 10):
    for wave_width in np.linspace(0.5, 3, 45):
        for i, ta in enumerate(np.linspace(0.5, 10, fps*10)):
            depth_map = Puff_profile(X, Y, ta, width=wave_width) 
            
            # Apply 2D cubic spline interpolation
            depth_map_smooth = RectBivariateSpline(x, y, depth_map)
            
            # Compute the gradients of depth_map_smooth with respect to x and y
            Gx = depth_map_smooth.partial_derivative(1,0)(x,y) 
            Gy = depth_map_smooth.partial_derivative(0,1)(x,y)
                
            # Create normal vectors
            normal_ori = np.ones((w, w, 3)) # Default reference image size
            normal_ori[:, :, 0] = -Gx
            normal_ori[:, :, 1] = -Gy
            
            # Normalize normal vectors
            norm = np.sqrt(Gx**2 + Gy**2 + 1)
            normal = normal_ori / norm[..., None]

            # Adjust height
            depth_map = depth_map + 2e-2  
            
            # Save normal and depth map
            file_name_depth = f"depth_map_seq{seq}_f{i}" 
            file_name_normal = f"normal_map_seq{seq}_f{i}" 
            np.save(os.path.join(DATA_SETS_DIR, "depth", file_name_depth), depth_map)
            np.save(os.path.join(DATA_SETS_DIR, "normal", file_name_normal), normal)
        # Increase sequence number for different simulation
        seq +=1
    return

def create_warp_maps(n1 = 1, n2 = 1.33):
    """This function creates a warp map corresponding to a depth map and normalmap
    
    Args:
        phase {"train", "test", "validation"}: Select wether the data is used for training / testing or validation.
        n1 (float, optional): Refractive index of the incident medium. Defaults to 1 (air).
        n2 (float, optional): Refractive index of refractive medium. Defaults to 1.33 (water).
    """
    normal_dir = os.path.join(DATA_SETS_DIR, "normal")
    depth_dir = os.path.join(DATA_SETS_DIR, "depth")
    
    for normal_file, depth_file in zip(os.listdir(normal_dir), os.listdir(depth_dir)):
        file_index = os.path.splitext(depth_file[9:])[0]
        normal = np.load(os.path.join(normal_dir, normal_file))
        depth_map = np.load(os.path.join(depth_dir, depth_file))
        
        for transparent in ["reflection", "refraction"]:
            warp_map = raytracing_im_generator_ST(normal, depth_map, transparent, n1=n1, n2=n2)
            file_name = "warp_map" + file_index
            np.save(os.path.join(DATA_SETS_DIR, "warp_map", transparent, file_name), warp_map)
            
                    
def create_warped_images(image, image_name, gray_scale=False):
    """This function warps a image according to a warp map.
    
    Args:
        image (ndarray): image with 3 channels for the RGB values.
        image_name (string): name of the image.
        train_or_test {"train", "test"}: Select wether the data is used for training or testing. Defaults to "train".
        refract_or_reflect {"refraction", "reflection"}: Select if refraction or reflection model is used. Defaults to "refraction".
    """
    if gray_scale:
        # not true gray scale
        image = image[:,:,0]
    
    # Normalization
    rgb = image / 255.0
    
    warp_dir = os.path.join(DATA_SETS_DIR, "warp_map")
        
    for transparent in ["refraction"]: #["reflection", "refraction"]
        warp_dir = os.path.join(DATA_SETS_DIR, "warp_map", transparent)
        for file in os.listdir(warp_dir):
            warp_map = np.load(os.path.join(warp_dir, file))
            image_name_save = image_name + file[8:-3] + "png" # check this
        
            # Deform image
            rgb_deformation = deform_image(rgb, warp_map)
            image_deformation = np.array(rgb_deformation * 255, dtype=np.uint8)

            imwrite(os.path.join(DATA_SETS_DIR, "warped_image", transparent, image_name_save), image_deformation)
            
            
if __name__ == "__main__":
    create_directory_structure()
    create_depth_and_normal_maps(w=128, fps=4.5)
    create_warp_maps(n1=1, n2=1.33) 
    
    # Create deformed image for each reference pattern
    for file in os.listdir('reference_patterns'):
        file_name = os.path.splitext(file)[0]
        file_path = os.path.join('reference_patterns', file)
        image  = imread(file_path)

        create_warped_images(image, file_name, gray_scale=True)