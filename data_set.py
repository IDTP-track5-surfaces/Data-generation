import os
import numpy as np
from utils import deform_image, raytracing_im_generator_ST
from imageio.v2 import imwrite, imread

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATTERN_DIR = os.path.join(ROOT_DIR, "reference_patterns")
WAVE_SEQUENCE_DIR = os.path.join(ROOT_DIR, "wave_sequences")
DATA_SETS_DIR = os.path.join(ROOT_DIR, "data_sets")

def create_directory_structure():
    if not os.path.exists(DATA_SETS_DIR):
        for i in ["reflection", "refraction"]:
            for j in ["train", "test"]:
                os.makedirs(os.path.join(DATA_SETS_DIR, i, j, "depth"))
                os.makedirs(os.path.join(DATA_SETS_DIR, i, j, "image"))
                os.makedirs(os.path.join(DATA_SETS_DIR, i, j, "warp_map"))
                os.makedirs(os.path.join(DATA_SETS_DIR, i, j, "warped_image"))
                

def create_warp_maps(train_or_test, refract_or_reflect, n1 = 1, n2 = 1.33):
    """This function creates a warp map corresponding to a depth map
    
    Args:
        train_or_test {"train", "test"}: Select wether the data is used for training or testing. Defaults to "train".
        refract_or_reflect {"refraction", "reflection"}: Select if refraction or reflection model is used. Defaults to "refraction".
        n1 (float, optional): Refractive index of the incident medium. Defaults to 1 (air).
        n2 (float, optional): Refractive index of refractive medium. Defaults to 1.33 (water).
    """
    save_dir = os.path.join(DATA_SETS_DIR, refract_or_reflect, train_or_test)
    
    for wave_index, waves in enumerate(os.listdir(WAVE_SEQUENCE_DIR)):
        wave_path = os.path.join(WAVE_SEQUENCE_DIR, waves)
        
        for seq in os.listdir(wave_path):
            seq_path = os.path.join(wave_path, seq)
            seq_index = seq[4:]
                
            for frame_idx, file in enumerate(os.listdir(seq_path)):
                depth_map = np.load(os.path.join(seq_path, file))
                
                # Save depth map
                file_name = "depth_" + str(wave_index) + "_"  + str(seq_index) + "_"  + str(frame_idx) + ".npy"
                np.save(os.path.join(save_dir, "depth", file_name), depth_map) 
                
                warp_map = raytracing_im_generator_ST(depth_map, n1, n2, refract_or_reflect)
                
                # Save warp map
                file_name = "warp_" + str(wave_index) + "_"  + str(seq_index) + "_"  + str(frame_idx) + ".npy"
                np.save(os.path.join(save_dir, "warp_map", file_name), warp_map)
            
                    
def warp_image(image, image_name, train_or_test, refract_or_reflect):
    """This function warps a image according to a warp map.
    
    Args:
        image (ndarray): image with 3 channels for the RGB values.
        image_name (string): name of the image.
        train_or_test {"train", "test"}: Select wether the data is used for training or testing. Defaults to "train".
        refract_or_reflect {"refraction", "reflection"}: Select if refraction or reflection model is used. Defaults to "refraction".
    """
    # Normalization
    rgb = image / 255.0
    
    save_dir = os.path.join(DATA_SETS_DIR, refract_or_reflect, train_or_test)
    warp_dir = os.path.join(save_dir, "warp_map")
    
    # Save original image
    image_name_save = image_name + ".jpg"
    imwrite(os.path.join(save_dir, "image", image_name_save), image)
    
    for file in os.listdir(warp_dir):
        warp_map = np.load(os.path.join(warp_dir, file))
        image_name_save = image_name + file[4:-3] + "jpg"
        
        # Deform image
        rgb_deformation = deform_image(rgb, warp_map)
        image_deformation = np.array(rgb_deformation * 255, dtype=np.uint8)

        imwrite(os.path.join(save_dir, "warped_image", image_name_save), image_deformation)
            
            
if __name__ == "__main__":
    phase = "train" #"test"
    refract_or_reflect = "refraction"
    
    create_directory_structure()
    create_warp_maps(train_or_test=phase, refract_or_reflect=refract_or_reflect, n1=1, n2=1.33)
    
    # Create deformed image for each reference pattern
    for file in os.listdir('reference_patterns'):
        file_name = os.path.splitext(file)[0]
        file_path = os.path.join('reference_patterns', file)
        image  = imread(file_path)

        warp_image(image, file_name, train_or_test=phase, refract_or_reflect=refract_or_reflect)