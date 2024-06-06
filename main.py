import os
import numpy as np
import argparse
from tqdm import tqdm

from utils import *
from imageio.v2 import imwrite, imread

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_directory_structure(data_dir, doReflection):
    """
    Creates directory structure for saving data.

    Args:
        data_dir (str): Root directory location where the data will be stored.
        doReflection (bool): Specify if reflection directory is created.
    """
    # Check if the root data directory does not exist
    if not os.path.exists(data_dir):
        # Create primary subdirectories: 'depth' and 'normal'
        os.makedirs(os.path.join(data_dir, "depth"))
        os.makedirs(os.path.join(data_dir, "normal"))
        
        dirs = ["refraction"]
        # If reflection data is needed, add 'reflection' to the list of directories
        if doReflection:
            dirs.append("reflection")

        # Create specified subdirectories and their corresponding 'warp_map' subdirectories
        for subdir in dirs:
            os.makedirs(os.path.join(data_dir, subdir))
            os.makedirs(os.path.join(data_dir, "warp_map", subdir))

def create_depth_and_normal_maps(data_dir, w=128, fps=24):
    """
    Creates depth maps according to the puff rheometer data,
    and calculates the corresponding normal map.

    Args:
        data_dir (str): Root directory location where the data will be stored.
        w (int, optional): Width in number of pixels. Defaults to 128.
        fps (int, optional): Frames per second. Defaults to 24.
    """
    # Create physical domain coordinates
    x = np.linspace(-52e-3, 52e-3, w)  # in meters
    y = np.linspace(-52e-3, 52e-3, w)  # in meters
    X, Y = np.meshgrid(x, y) 
    
    seq = 0  # Initialize sequence counter
    # Loop over different wave parameters
    for wave_width in np.linspace(0.5, 3, 9):
        for wave_depth in np.linspace(-0.003, -0.008, num=6):
            # Loop over time steps
            for i, ta in enumerate(np.arange(start=0.5, stop=10, step=1/fps)):
                
                # Create depth profile and gradients
                depth_map = Puff_profile(X, Y, ta, depth=wave_depth, width=wave_width)                 
                Gx, Gy = grad_puff_profile(X, Y, ta, depth=wave_depth, width=wave_width)
                    
                # Create normal vectors
                normal_ori = np.ones((w, w, 3))  # Initialize normal map
                normal_ori[:, :, 0] = -Gx  # X gradient
                normal_ori[:, :, 1] = -Gy  # Y gradient
                
                # Normalize normal vectors
                norm = np.sqrt(Gx**2 + Gy**2 + 1)
                normal = normal_ori / norm[..., None]

                # Adjust height to undeformed surface height
                depth_map += 1e-2  # in meters
            
                # Save normal and depth maps
                file_name_depth = f"depth_seq{seq}-f{i}" 
                file_name_normal = f"normal_seq{seq}-f{i}" 
                np.save(os.path.join(data_dir, "depth", file_name_depth), depth_map)
                np.save(os.path.join(data_dir, "normal", file_name_normal), normal)
            
            # Increase sequence number for different simulation parameters
            seq += 1
    return

def create_warp_maps(data_dir, doReflection, n1=1, n2=1.33):
    """
    This function creates a warp map corresponding to a depth map and normal map.

    Args:
        data_dir (str): Root directory location where the data will be stored.
        doReflection (bool): Specify if reflection warp maps are created.
        n1 (float, optional): Refractive index of the incident medium. Defaults to 1 (air).
        n2 (float, optional): Refractive index of refractive medium. Defaults to 1.33 (water).
    """
    normal_dir = os.path.join(data_dir, "normal")
    depth_dir = os.path.join(data_dir, "depth")
    
    # Iterate through files in normal and depth directories
    for normal_file, depth_file in zip(os.listdir(normal_dir), os.listdir(depth_dir)):
        # Extract the file index from the depth file name
        file_index = os.path.splitext(depth_file[9:])[0]
        
        # Load normal and depth maps
        normal = np.load(os.path.join(normal_dir, normal_file))
        depth_map = np.load(os.path.join(depth_dir, depth_file))
        
        # Initialize list of directories for warp maps
        dirs = ["refraction"]
        if doReflection:
            dirs.append("reflection")
        
        for transparent in dirs:
            warp_map = raytracing_im_generator_ST(normal, depth_map, transparent, n1=n1, n2=n2)
            
            # Construct file name and save warp map
            file_name = f"warp_seq{file_index}"
            np.save(os.path.join(data_dir, "warp_map", transparent, file_name), warp_map)
            
                    
def create_warped_images(image, data_dir, doReflection, gray_scale=False):
    """This function warps a image according to a warp map.
    
    Args:
        image (ndarray): image with 3 channels for the RGB values.
        data_dir (string): root directory location where the data will be stored.
        doReflection (boolean): Specify if reflection images are created.
        gray_scale (boolean, optional): Specify if a grayscale image is used.
    """
    # Add axis for compatibility with RGB images
    if gray_scale:
        image = image[:,:,0][..., np.newaxis]
    
    # Normalization
    rgb = image / 255.0
    
    warp_dir = os.path.join(data_dir, "warp_map") 

    dirs = ["refraction"]
    if doReflection:
        dirs.append("reflection")   
    for transparent in dirs:
        warp_dir = os.path.join(data_dir, "warp_map", transparent)
        N = len(os.listdir(warp_dir))
        for i, file in tqdm(enumerate(os.listdir(warp_dir))):
            warp_map = np.load(os.path.join(warp_dir, file))
            image_name = file[5:-4] + ".png"
        
            # Deform image
            rgb_deformation = deform_image(rgb, warp_map)
            
            # Copy gray scale deformation for RGB channels
            if gray_scale:
                rgb_deformation = np.concatenate(3*[rgb_deformation], axis=-1)
            
            # Save image
            image_deformation = np.array(rgb_deformation * 255, dtype=np.uint8)
            imwrite(os.path.join(data_dir, transparent, image_name), image_deformation)

def main(data_dir, ref_file_path, doCreateMaps=False, doCreateImages=False, doReflection=False):
    if not doCreateMaps and not doCreateImages:
        print("Neither maps, nor images are created.")
        return
    create_directory_structure(data_dir, doReflection) 
    print("Directory structure is created.") 
    if doCreateMaps:
        create_depth_and_normal_maps(data_dir, fps=12)
        print("Depth and normal maps are created.") 
        create_warp_maps(data_dir, doReflection) 
        print("Warp maps are created.")
    
    if doCreateImages:
        if ref_file_path == '':
            print("Reference image should be given.")
            return
        image  = imread(ref_file_path)
        create_warped_images(image, data_dir, doReflection)
        print("Image creation is finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create depth, normal and corresponding warp maps to create warped images of a reference image.")
    parser.add_argument('--root_dir', type=str, default=ROOT_DIR, help='Root directory.')
    parser.add_argument('--data_dir', type=str, default='', help='Directory name for saving the data.')
    parser.add_argument('--ref_image', type=str, default='', help='File location of reference image.')
    parser.add_argument('--create_maps', action='store_true', help='Specify if depth/normal and warp maps should be created.')
    parser.add_argument('--create_images', action='store_true', help='Specify if images should be created.')
    parser.add_argument('--reflection', action='store_true', help='Specify if data generation should also be done for reflection.')

    args = parser.parse_args()

    ROOT_DIR = args.root_dir
    REF_IMAGE = args.ref_image
    CREATE_MAPS = args.create_maps
    CREATE_IMAGES = args.create_images
    REF_IMAGE = args.ref_image
    REFLECTION = args.reflection
    
    data_loc = args.data_dir
    if args.data_dir == '':
        data_loc = "data_sets" 
    DATA_SETS_DIR = os.path.join(ROOT_DIR, data_loc)
             
    main(DATA_SETS_DIR, REF_IMAGE, doCreateMaps=CREATE_MAPS, doCreateImages=CREATE_IMAGES, doReflection=REFLECTION)