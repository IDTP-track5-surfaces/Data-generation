import os
import numpy as np
from utils import deform_image, raytracing_im_generator_ST
from imageio.v2 import imwrite, imread

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATTERN_DIR = os.path.join(ROOT_DIR, "reference_patterns")
WAVE_SEQUENCE_DIR = os.path.join(ROOT_DIR, "wave_sequences")
DATA_SETS_DIR = os.path.join(ROOT_DIR, "data_sets")

# TRAINING_IMAGE_DIR = os.path.join(WAVE_SEQUENCE_DIR, "data_sets", "training_images")
# TEST_IMAGE_DIR = os.path.join(WAVE_SEQUENCE_DIR, "data_sets", "test_images")
# DATA_DIR = {'train': TRAINING_IMAGE_DIR, 'test': TEST_IMAGE_DIR}


def create_directory_structure():
    if not os.path.exists(DATA_SETS_DIR):
        for i in ["reflection", "refraction"]:
            for j in ["train", "test"]:
                os.makedirs(os.path.join(DATA_SETS_DIR, i, j, "depth"))
                os.makedirs(os.path.join(DATA_SETS_DIR, i, j, "image"))
                os.makedirs(os.path.join(DATA_SETS_DIR, i, j, "warp_map"))
                os.makedirs(os.path.join(DATA_SETS_DIR, i, j, "warped_image"))
                

def combine_sequence_depthmap(image, image_name, train_or_test = "train", 
                              refract_or_reflect = "refraction", 
                              save_warped = True, n1 = 1, n2 = 1.33):
    """_summary_

    Args:
        image (ndarray): image with 3 channels for the RGB values.
        image_name (string): name of the image.
        train_or_test ({"train", "test"}, optional): Select wether the data is used for training or testing. Defaults to "train".
        refract_or_reflect ({"refraction", "reflection"}, optional): Select if refraction or reflection model is used. Defaults to "refraction".
        save_warped (boolean, optional): Select if warped images should be saved. Defaults to True.
        n1 (float, optional): Refractive index of the incident medium. Defaults to 1 (air).
        n2 (float, optional): Refractive index of refractive medium. Defaults to 1.33 (water).
    """
    rgb = image / 255.0
    save_dir = os.path.join(DATA_SETS_DIR, refract_or_reflect, train_or_test)
    
    image_name_save = image_name + ".jpg"
    imwrite(os.path.join(save_dir, "image", image_name_save), image)
    
    for wave_index, waves in enumerate(os.listdir(WAVE_SEQUENCE_DIR)):
        wave_path = os.path.join(WAVE_SEQUENCE_DIR, waves)
        
        for seq in os.listdir(wave_path):
            seq_path = os.path.join(wave_path, seq)
            seq_index = seq[4:]
                
            for frame_idx, file in enumerate(os.listdir(os.path.join(seq_path, "depth"))):
                depth_map = np.load(os.path.join(seq_path, "depth", file))
                file_name = "depth_" + str(wave_index) + "_"  + str(seq_index) + "_"  + str(frame_idx) + ".npy"
                np.save(os.path.join(save_dir, "depth", file_name), depth_map) 
                
                # Save warp maps for refraction and reflection in seperate folders
                # for i in ["refraction", "reflection"]:
                
                warp_map = raytracing_im_generator_ST(depth_map, n1, n2, refract_or_reflect)
                file_name = "warp_" + str(wave_index) + "_"  + str(seq_index) + "_"  + str(frame_idx) + ".npy"
                np.save(os.path.join(save_dir, "warp_map", file_name), warp_map)
                
                if save_warped:
                    rgb_deformation = deform_image(rgb, warp_map)
                    image_name = image_name + "_" + str(wave_index) + "_"  + str(seq_index) + "_"  + str(frame_idx) + ".jpg"
                    imwrite(os.path.join(save_dir, "warped_image", image_name), rgb_deformation)
                    

            

def generate_warp_map(train_or_test='train', n1 = 1, n2 = 1.33):
    """Function that creates warp maps for corresponding depth map 
    and saves it to a npy-file. 

    Args:
        train_or_test ({'train', 'test'}, optional): Generate warp maps for testing or training. Default is 'train'.
        n1 (float, optional): Refractive index of the incident medium. Defaults to 1 (air).
        n2 (float, optional): Refractive index of refractive medium. Defaults to 1.33 (water).
    """
    
    for waves in os.listdir(WAVE_SEQUENCE_DIR):
        wave_path = os.path.join(WAVE_SEQUENCE_DIR, waves)
        
        for seq in os.listdir(wave_path):
            seq_path = os.path.join(wave_path, seq)
            
            # Create directories for reflection and refraction
            if not os.path.exists(os.path.join(seq_path, "warp")):
                os.makedirs(os.path.join(seq_path, "warp", "reflection"))
                os.makedirs(os.path.join(seq_path, "warp", "refraction"))
                
            for file in os.listdir(os.path.join(seq_path, "depth")):
                depth_map = np.load(os.path.join(seq_path, "depth", file))
                
                # Save warp maps for refraction and reflection in seperate folders
                for i in ["refraction", "reflection"]:
                    warp_map = raytracing_im_generator_ST(depth_map, n1, n2, i)
                    np.save(os.path.join(seq_path, "warp", i, file), warp_map)


def generate_warped_image(image, name):
    """Function that warps the image for all warp maps. 

    Args:
        image (ndarray): rgb image
        name (string): name of the image
    """
    rgb = image / 255.0
        
    for waves in os.listdir(WAVE_SEQUENCE_DIR):
        wave_path = os.path.join(WAVE_SEQUENCE_DIR, waves)
        
        for seq in os.listdir(wave_path):
            seq_path = os.path.join(wave_path, seq)
            
            for file in os.listdir(os.path.join(seq_path, "warp")):
                warp_map = np.load(os.path.join(seq_path, "warp", file))
                
                for i in ["refraction", "reflection"]:                    
                    # Create directory
                    if not os.path.exists(os.path.join(TRAINING_IMAGE_DIR, i, waves, seq, name)):
                        os.makedirs(os.path.join(TRAINING_IMAGE_DIR, i, waves, seq, name))
                    
                    # Save deformed rgb image    
                    rgb_deformation = deform_image(rgb, warp_map)
                    np.save(os.path.join(TRAINING_IMAGE_DIR, i, waves, seq, name, file), rgb_deformation)


if __name__ == "__main__":
    create_directory_structure()
    img = imread("reference_patterns/tex1.jpg")
    combine_sequence_depthmap(img, "tex1", 1, 1.33, "train", "refraction")