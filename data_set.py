import os
import numpy as np
from utils import *
from imageio.v2 import imwrite, imread

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATTERN_DIR = os.path.join(ROOT_DIR, "reference_patterns")
DATA_SETS_DIR = os.path.join(ROOT_DIR, "data_sets")

def create_directory_structure():
    if not os.path.exists(DATA_SETS_DIR):
        for i in ["train", "test", "validation"]:
            os.makedirs(os.path.join(DATA_SETS_DIR, i, "depth"))
            os.makedirs(os.path.join(DATA_SETS_DIR, i, "normal"))
            for j in ["reflection", "refraction"]:
                os.makedirs(os.path.join(DATA_SETS_DIR, i, "warp_map", j))
                os.makedirs(os.path.join(DATA_SETS_DIR, i, "warped_image", j))
            
            
def create_depth_maps(phase, w=265, fps = 24):
    x = np.linspace(-52e-3,52e-3, w); # in meter
    y = np.linspace(-52e-3,52e-3, w); # in meter
    x2, y2 = np.meshgrid(x, y)
    
    for i, ta in enumerate(np.linspace(0,10, fps*10)):
        depth_map = Puff_profile(x2, y2, ta) # TD: change for multiple profiles
        
        # Adjust height
        depth_map = depth_map + 2e-2  
        
        file_name = f"depth_map_seq0_f{i}" # change sequence number
        np.save(os.path.join(DATA_SETS_DIR, phase, "depth", file_name), depth_map)
    return


def create_normal_maps(phase):
    depth_dir = os.path.join(DATA_SETS_DIR, phase, "depth")
    
    for file in os.listdir(depth_dir):
        file_index = os.path.splitext(file[9:])[0]
        # print(file_index)
    #Frisos function
    return


def create_warp_maps(phase, n1 = 1, n2 = 1.33):
    """This function creates a warp map corresponding to a depth map and normalmap
    
    Args:
        phase {"train", "test", "validation"}: Select wether the data is used for training / testing or validation.
        n1 (float, optional): Refractive index of the incident medium. Defaults to 1 (air).
        n2 (float, optional): Refractive index of refractive medium. Defaults to 1.33 (water).
    """
    normal_dir = os.path.join(DATA_SETS_DIR, phase, "normal")
    depth_dir = os.path.join(DATA_SETS_DIR, phase, "depth")
    
    for normal_file, depth_file in zip(os.listdir(normal_dir), os.listdir(depth_dir)):
        file_index = os.path.splitext(depth_file[9:])[0]
        normal = np.load(os.path.join(normal_dir, normal_file))
        depth_map = np.load(os.path.join(depth_dir, depth_file))
        
        for transparent in ["reflection", "refraction"]:
            warp_map = raytracing_im_generator_ST(normal, depth_map, transparent, n1=n1, n2=n2)
            file_name = "warp_map" + file_index
            np.save(os.path.join(DATA_SETS_DIR, phase, "warp_map", file_name), warp_map)
            
                    
def create_warped_images(image, image_name, phase):
    """This function warps a image according to a warp map.
    
    Args:
        image (ndarray): image with 3 channels for the RGB values.
        image_name (string): name of the image.
        train_or_test {"train", "test"}: Select wether the data is used for training or testing. Defaults to "train".
        refract_or_reflect {"refraction", "reflection"}: Select if refraction or reflection model is used. Defaults to "refraction".
    """
    # Normalization
    rgb = image / 255.0
    
    # save_dir = os.path.join(DATA_SETS_DIR, phase,  train_or_test)
    warp_dir = os.path.join(DATA_SETS_DIR, phase, "warp_map")
        
    for transparent in ["reflection", "refraction"]:
        warp_dir = os.path.join(DATA_SETS_DIR, phase, "warp_map", transparent)
        for file in os.listdir(warp_dir):
            warp_map = np.load(os.path.join(warp_dir, file))
            image_name_save = image_name + file[8:-3] + "jpg" # check this
        
            # Deform image
            rgb_deformation = deform_image(rgb, warp_map)
            image_deformation = np.array(rgb_deformation * 255, dtype=np.uint8)

            imwrite(os.path.join(DATA_SETS_DIR, phase, "warped_image", transparent, image_name_save), image_deformation)
            
            
if __name__ == "__main__":
    phase = "train" #"test"/validation
    
    create_directory_structure()
    create_depth_maps(phase)
    create_normal_maps(phase)    
    create_warp_maps(phase, n1=1, n2=1.33) 
    
    # Create deformed image for each reference pattern
    for file in os.listdir('reference_patterns'):
        file_name = os.path.splitext(file)[0]
        file_path = os.path.join('reference_patterns', file)
        image  = imread(file_path)

        create_warped_images(image, file_name, phase)