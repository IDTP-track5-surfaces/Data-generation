import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR =  os.path.join(ROOT_DIR, "data_sets")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")

def find_max_incident_angle(normal):
    nz = normal[:,:,2]
    ny = normal[:,:,1]
    
    theta_max = 0
    theta_list = np.deg2rad(np.linspace(0, 90, num=91))
    for i, theta in enumerate(theta_list):
        sz = -np.cos(theta)+2*(nz*np.cos(theta)-ny*np.sin(theta))*nz
        if sz.min() > 0:
            theta_max = i
    return theta_max


if __name__ == "__main__":
    depth_dir = os.path.join(VALIDATION_DIR, "depth")
    
    max_angels = np.zeros(0)
    for i in os.listdir(depth_dir):
        file = os.path.join(depth_dir, i)
        depth_map = np.load(file)
        
        
        h, w = depth_map.shape

        # Generate normal map from depth map
        Gx = sobel(depth_map, axis=0)
        Gy = sobel(depth_map, axis=1)

        # Create normal vectors
        normal_ori = np.ones((h, w, 3)) # Default reference image size
        normal_ori[:, :, 0] = -Gx
        normal_ori[:, :, 1] = -Gy

        # Normalize normal vectors
        norm = np.sqrt(Gx**2 + Gy**2 + 1)
        normal = normal_ori / norm[..., None]

        max_angel = find_max_incident_angle(normal)
        max_angels = np.append(max_angels, max_angel)

    unique, counts = np.unique(max_angels, return_counts=True)
    print(f"mean: {max_angels.mean()}")
    print(f"median: {np.median(max_angels)}")
    print(f"mode: {unique[counts.argmax()]}")
    print(f"min: {max_angels.min()}")
    print(f"max: {max_angels.max()}")

    plt.hist(max_angels)
    plt.xlabel(r"Incident angle $\theta$")
    plt.title("Histogram of maximum allowed angle")
    plt.suptitle(f"mean: {max_angels.mean()}, median: {np.median(max_angels)}, mode: {unique[counts.argmax()]}, min: {max_angels.min()}")
    plt.show()
