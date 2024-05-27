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
        os.makedirs(os.path.join(DATA_SETS_DIR, "depth"))
        os.makedirs(os.path.join(DATA_SETS_DIR, "normal"))
        for j in ["reflection", "refraction"]:
            os.makedirs(os.path.join(DATA_SETS_DIR,  j))
            os.makedirs(os.path.join(DATA_SETS_DIR, "warp_map", j))
            
            
def create_depth_and_normal_maps(w=128, fps = 24):
    # Create physical domain
    x = np.linspace(-52e-3,52e-3, w); # in meter
    y = np.linspace(-52e-3,52e-3, w); # in meter
    X, Y = np.meshgrid(x, y)
    seq = 0
    
    for wave_width in np.linspace(0.5, 3, 9):
        for wave_depth in np.linspace(-0.003, -0.008, num=6):
            for i, ta in enumerate(np.arange(start=0.5, stop=10, step=1/fps)):
                
                # Create depth profile and gradients
                depth_map = Puff_profile(X, Y, ta, depth=wave_depth, width=wave_width)                 
                Gx, Gy = grad_puff_profile(X, Y, ta, depth=wave_depth, width=wave_width)
                    
                # Create normal vectors
                normal_ori = np.ones((w, w, 3)) # Default reference image size
                normal_ori[:, :, 0] = -Gx
                normal_ori[:, :, 1] = -Gy
                
                # Normalize normal vectors
                norm = np.sqrt(Gx**2 + Gy**2 + 1)
                normal = normal_ori / norm[..., None]

                # Adjust height
                depth_map = depth_map + 1e-2  # in meters
            
                # Save normal and depth map
                file_name_depth = f"depth_seq{seq}-f{i}" 
                file_name_normal = f"normal_seq{seq}-f{i}" 
                np.save(os.path.join(DATA_SETS_DIR, "depth", file_name_depth), depth_map)
                np.save(os.path.join(DATA_SETS_DIR, "normal", file_name_normal), normal)
            # Increase sequence number for different simulation
            seq +=1
    return

def create_warp_maps(n1 = 1, n2 = 1.33):
    """This function creates a warp map corresponding to a depth map and normalmap
    
    Args:
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
            file_name = "warp_seq" + file_index
            np.save(os.path.join(DATA_SETS_DIR, "warp_map", transparent, file_name), warp_map)
            
                    
def create_warped_images(image, gray_scale=False):
    """This function warps a image according to a warp map.
    
    Args:
        image (ndarray): image with 3 channels for the RGB values.
        image_name (string): name of the image.
        refract_or_reflect {"refraction", "reflection"}: Select if refraction or reflection model is used. Defaults to "refraction".
    """
    # Add axis for compatibility with RGB images
    if gray_scale:
        image = image[:,:,0][..., np.newaxis]
    
    # Normalization
    rgb = image / 255.0
    
    warp_dir = os.path.join(DATA_SETS_DIR, "warp_map")    
    for transparent in ["refraction"]: #["reflection", "refraction"]
        warp_dir = os.path.join(DATA_SETS_DIR, "warp_map", transparent)
        N = len(os.listdir(warp_dir))
        for i, file in enumerate(os.listdir(warp_dir)):
            warp_map = np.load(os.path.join(warp_dir, file))
            image_name = file[5:-4] + ".png" # check this
        
            # Deform image
            rgb_deformation = deform_image(rgb, warp_map)
            
            # Copy gray scale deformation for RGB channels
            if gray_scale:
                rgb_deformation = np.concatenate(3*[rgb_deformation], axis=-1)
            
            # Save image
            image_deformation = np.array(rgb_deformation * 255, dtype=np.uint8)
            imwrite(os.path.join(DATA_SETS_DIR, transparent, image_name), image_deformation)
            
            if i%100 == 0:
                print(f"{i/N*100} %")
            
if __name__ == "__main__":
    create_directory_structure()  
    create_depth_and_normal_maps(fps=12)
    create_warp_maps() 
    
    # Create deformed image for each reference pattern  
    file_path = os.path.join('reference_patterns', "ref_seq_24.png")
    image  = imread(file_path)
    create_warped_images(image)
    
    print("Data generation is finished")