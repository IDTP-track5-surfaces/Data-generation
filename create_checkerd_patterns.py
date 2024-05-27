import os
import numpy as np
from imageio.v2 import imwrite
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERNCE_PATTERNS_DIR = os.path.join(ROOT_DIR, "reference_patterns")

def create_checkerboard_array(size, tile_size):
    array = np.zeros([size, size], dtype=np.uint8)

    # Iterate through each cell
    for i in range(size):
        for j in range(size):
            # Calculate the tile indices
            tile_i = i // tile_size
            tile_j = j // tile_size

            # Set value based on tile position
            if (tile_i + tile_j) % 2 == 0:
                array[i, j] = 255
    
    return np.array(3*[array]).T # return RGB array

if __name__ == "__main__":
    # Define array size and tile size
    array_size = 128
    tile_sizes = [2,8,32]
    
    fig, ax = plt.subplots(ncols=3, figsize=(15,5))

    for i, tile_size in enumerate(tile_sizes):
        # Create checkerboard array
        checkerboard_array = create_checkerboard_array(array_size, tile_size)
        ax[i].imshow(checkerboard_array, cmap='gray')
        
        file_loc = os.path.join(REFERNCE_PATTERNS_DIR, f"grid{tile_size}.png")
        imwrite(file_loc, checkerboard_array)

    plt.show()