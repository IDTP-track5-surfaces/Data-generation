import os
import numpy as np
import argparse
import tqdm

from utils import *
from imageio.v2 import imwrite, imread

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_directory_structure(data_dir, doReflection):
    """_summary_

    Args:
        data_dir (string): root directory location where the data will be stored.
    """
    if not os.path.exists(data_dir):
        os.makedirs(os.path.join(data_dir, "depth"))
        os.makedirs(os.path.join(data_dir, "normal"))
        
        dirs = ["refraction"]
        if doReflection:
            dirs.append("reflection")

        for j in dirs:
            os.makedirs(os.path.join(data_dir, j))
            os.makedirs(os.path.join(data_dir, "warp_map", j))

                
def create_depth_and_normal_maps(data_dir, w=128, fps = 24):
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
                np.save(os.path.join(data_dir, "depth", file_name_depth), depth_map)
                np.save(os.path.join(data_dir, "normal", file_name_normal), normal)
            # Increase sequence number for different simulation
            seq +=1
    return

def create_warp_maps(data_dir, doReflection, n1 = 1, n2 = 1.33):
    """This function creates a warp map corresponding to a depth map and normalmap
    
    Args:
        data_dir (string): root directory location where the data will be stored.
        n1 (float, optional): Refractive index of the incident medium. Defaults to 1 (air).
        n2 (float, optional): Refractive index of refractive medium. Defaults to 1.33 (water).
    """
    normal_dir = os.path.join(data_dir, "normal")
    depth_dir = os.path.join(data_dir, "depth")
    
    for normal_file, depth_file in zip(os.listdir(normal_dir), os.listdir(depth_dir)):
        file_index = os.path.splitext(depth_file[9:])[0]
        normal = np.load(os.path.join(normal_dir, normal_file))
        depth_map = np.load(os.path.join(depth_dir, depth_file))
        
        dirs = ["refraction"]
        if doReflection:
            dirs.append("reflection")
            
        for transparent in dirs:
            warp_map = raytracing_im_generator_ST(normal, depth_map, transparent, n1=n1, n2=n2)
            file_name = "warp_seq" + file_index
            np.save(os.path.join(data_dir, "warp_map", transparent, file_name), warp_map)
            
                    
def create_warped_images(image, data_dir, doReflection, gray_scale=False):
    """This function warps a image according to a warp map.
    
    Args:
        image (ndarray): image with 3 channels for the RGB values.
        data_dir (string): root directory location where the data will be stored.
        image_name (string): name of the image.
        refract_or_reflect {"refraction", "reflection"}: Select if refraction or reflection model is used. Defaults to "refraction".
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
            
            # if i%100 == 0:
            #     print(f"{i/N*100} %")

def main(data_dir, ref_file_path, doCreateMaps=False, doCreateImages=False):
    if not doCreateMaps and not doCreateImages:
        print("Neither maps, nor images are created.")
        return
    
    create_directory_structure(data_dir) 
    print("Directory structure is created.") 
    if doCreateMaps:
        create_depth_and_normal_maps(data_dir, fps=12)
        print("Depth and normal maps are created.") 
        create_warp_maps(data_dir) 
        print("Warp maps are created.")
    
    if doCreateMaps:
        # file_path = os.path.join(REFERENCE_PATTERN_DIR, "ref_seq_24.png")
        if ref_file_path == '':
            print("Reference image should be given.")
            return
        image  = imread(ref_file_path)
        create_warped_images(image, data_dir)
        print("Image creation is finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create depth, normal and corresponding warp maps to create warped images of a reference image.")
    parser.add_argument('--root_dir', type=str, default=ROOT_DIR, help='Root directory.')
    parser.add_argument('--data_dir', type=str, default='', help='Directory name for saving the data.')
    parser.add_argument('--ref_image', type=str, default='', help='File location of reference image.')
    parser.add_argument('--create_maps', action='store_true', help='Specify if depth/normal and warp maps should be created.')
    parser.add_argument('--create_images', action='store_true', help='Specify if images should be created.')

    args = parser.parse_args()

    ROOT_DIR = args.root_dir
    REF_IMAGE = args.ref_image
    CREATE_MAPS = args.create_maps
    CREATE_IMAGES = args.create_images
    REF_IMAGE = args.ref_image
    
    data_loc = args.data_dir
    if args.data_dir == '':
        data_loc = "data_sets" 
    DATA_SETS_DIR = os.path.join(ROOT_DIR, data_loc)
             
    main(DATA_SETS_DIR, REF_IMAGE, doCreateMaps=CREATE_MAPS, doCreateImages=CREATE_IMAGES)