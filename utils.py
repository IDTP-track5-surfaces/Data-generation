import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import sobel

   
# Function to perform ray tracing
def raytracing_im_generator_ST(im_rgb, depth_map, n1, n2, selector = True):
    step = 1
    h, w, channels = im_rgb.shape

    # Generate normal map from depth map
    Gx = sobel(depth_map, axis=0)
    Gy = sobel(depth_map, axis=1)

    # Create normal vectors
    normal_ori = np.ones((h,w, 3))
    normal_ori[:, :, 0] = -Gx
    normal_ori[:, :, 1] = -Gy
    
    # Normalize normal vectors
    norm = np.sqrt(Gx**2 + Gy**2 + 1)
    normal = normal_ori / norm[..., None]

    # Create incident vectors
    s1 = np.zeros_like(normal)
    s1[:, :, 2] = -1
    
    if selector: # refraction
        s2 = refraction_wikipedia(normal, s1, n1, n2)
    else: # reflection 
        print("Not implemented")
        
        # Rotation angle
        incident_angle = np.radians(20)
        theta = np.pi/2 - incident_angle

        # Rotation matrix
        R = np.array([[1, 0, 0],[0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

        # Rotated incident vectors
        s1 = np.einsum('ij,pqj->pqi', R, s1)
        
        s2 = reflection(normal, s1)
    

    a = depth_map / s2[:, :, 2]
    x_c = np.round(a * s2[:, :, 0] / step, 2)
    y_c = np.round(a * s2[:, :, 1] / step, 2)

    warp_map = np.dstack((x_c, y_c))

    return warp_map


def refraction(normal, s1, n1, n2):
    this_normal = normal
    s1_normalized = s1 / np.sqrt(s1[:,:,0]**2 + s1[:,:,1]**2 + s1[:,:,2]**2)[..., None]
    
    term_1 = np.cross(this_normal, np.cross(-this_normal, s1_normalized, axis=2), axis=2)
    term_2 = np.sqrt(1 - (n1 / n2)**2 * np.sum(np.cross(this_normal, s1_normalized, axis=2) * np.cross(this_normal, s1_normalized, axis=2), axis=2))
    
    s2 = (n1 / n2) * term_1 - this_normal * term_2[...,None]
    s2_normalized = s2 / np.sqrt(s2[:,:,0]**2 + s2[:,:,1]**2 + s2[:,:,2]**2)[...,None]

    return s2_normalized

def refraction_wikipedia(normal, s1, n1, n2):
    n = n1/n2
    
    cos1 = np.einsum('ijk, ijk->ij', -normal, s1)
    cos2 = np.sqrt(1-n**2*(1-cos1**2))
    
    s2 = n*s1 + (n*cos1-cos2)[...,None]*normal
    
    s2_normalized = s2/np.linalg.norm(s2, axis =2)[..., None]
    
    return s2_normalized

def reflection(normal, s1):
    cos1 = np.einsum('ijk, ijk->ij', -normal, s1)

    s2 = s1 + 2*cos1[...,None]*normal

    s2_normalized = s2/np.linalg.norm(s2, axis =2)[..., None]
    
    return s2_normalized

def deform_image(img, warp_map):
    """_summary_

    Args:
        img (_type_): Original reference pattern
        warp_map (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    h, w, nChannel = img.shape
    # Create meshgrid for the original coordinates
    X, Y = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))

    x_c = warp_map[:,:,0]
    y_c = warp_map[:,:,1]

    # Mapping coordinates to warped coordinates
    Xnew = X + x_c
    Ynew = Y + y_c

    # Flattening the original coordinates and values
    Xnew = Xnew.flatten()
    Ynew = Ynew.flatten()

    # Selecting only the valid points
    valid = np.logical_and.reduce([Xnew >= 1, Xnew <= w, Ynew >= 1, Ynew <= h])
    x_valid = Xnew[valid]
    y_valid = Ynew[valid]

    imgCurr = np.zeros((h, w, nChannel))

    # Interpolate all channels of the image
    for k in range(nChannel):
        currFrame = np.zeros(h * w)
        currFrame[valid] = griddata((X.flatten(), Y.flatten()), img[:,:,k].flatten(), (x_valid, y_valid), method='linear', fill_value = 0)
        imgCurr[:,:, k] = np.reshape(currFrame, (h, w), order='F')
    
    return imgCurr

def generate_example_depth_map():
    xymax = 10
    depth_map = np.zeros((900,900))
    for i,x in enumerate(np.linspace(-xymax, xymax, num=900)):
        for j,y in enumerate(np.linspace(-xymax, xymax, num=900)):
            depth_map[i,j] = 20*np.sin(-(x**2 + y**2)/10)
    return depth_map
