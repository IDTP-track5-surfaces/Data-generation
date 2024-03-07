from utils import *

import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread

def __main__():
    n1 = 1.0 # air
    n2 = 1.33 # water
    
    # Load reference pattern
    rgb = imread('tex1.jpg') / 255.0

    depth_map = generate_example_depth_map() # Change to depth map from simulation
    warp_map = raytracing_im_generator_ST(rgb, depth_map, n1, n2)
    
    rgb_refraction = deform_image(rgb, warp_map)
    
    plt.imshow(rgb_refraction)
    plt.show()

    # fig, ax = plt.subplots(ncols = 2)
    # ax[0].imshow(warp_map[:,:,0])
    # ax[1].imshow(warp_map[:,:,1])
    # ax[0].set_title("Warp x")
    # ax[1].set_title("Warp y")
    # plt.show()
    
    

if __name__ == "__main__":
    __main__()
