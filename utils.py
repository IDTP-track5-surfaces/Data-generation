import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter #,sobel

   
# Function to perform ray tracing
def raytracing_im_generator_ST(normal, depth_map, transparent, n1=1, n2=1.33):
    width, height = depth_map.shape
    scaling = width/(2*52e-3)
    
    # Create incident vectors
    s1 = np.zeros_like(normal)
    s1[:, :, 2] = -1
    
    if transparent == "refraction":
        s2 = refraction(normal, s1, n1, n2)
               
        a = depth_map / s2[:, :, 2]
        x_c = (a * s2[:, :, 0])*scaling
        y_c = (a * s2[:, :, 1])*scaling
    elif transparent == "reflection":         
        # Rotation angle
        theta = np.deg2rad(80) # incident angle

        # Rotation matrix
        R = np.array([[1, 0, 0],[0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

        # Rotated incident vectors
        s1 = np.einsum('ij,pqj->pqi', R, s1)
        
        s2 = reflection(normal, s1)
                
        gamma = 256*.2

        r = np.abs(s1)
        w = -gamma*(s2 - r*np.einsum('ijk, ijk->ij', s2, r)[..., None])

        x_c = w[:,:,0]
        y_c = np.sqrt(w[:,:,1]**2 + w[:,:,2]**2)
    else:
        print("Not implemented")


    warp_map = np.dstack((x_c, y_c))

    return warp_map

# This is equivalent formula from wikipedia: https://en.wikipedia.org/wiki/Snell%27s_law
def refraction(normal, s1, n1, n2):
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
    h, w, nChannel = img.shape
    h_map, w_map, _ = warp_map.shape
    
    # Create meshgrid for the original coordinates
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    x_c = warp_map[:,:,0]
    y_c = warp_map[:,:,1]

    # Select middle of image 
    start_row = (h - h_map)//2
    end_row =  start_row + h_map
    start_col = (w - w_map)//2
    end_col = start_col + w_map

    # Mapping coordinates to warped coordinates
    Xnew = X[start_row:end_row, start_col:end_col] + x_c
    Ynew = Y[start_row:end_row, start_col:end_col] + y_c

    # Flattening the original coordinates and values
    Xnew = Xnew.flatten()
    Ynew = Ynew.flatten()

    imgCurr = np.zeros((h_map, w_map, nChannel))

    # Interpolate all channels of the image
    for k in range(nChannel):
        currFrame_valid = np.zeros(h * w)
        currFrame = griddata((X.flatten(), Y.flatten()), img[:,:,k].flatten(), (Xnew, Ynew), method='linear', fill_value = 0)
        imgCurr[:,:, k] = np.reshape(currFrame, (h_map, w_map))
    
    return imgCurr

# Profile funtions
def exp_f(x,a,b,c):
    return a*np.exp(-(x/b)) + c;
 
def pol2_f(x,a,b,c):
    return a*x + b*x**2 + c;
 
def Puff_profile(x, y, ta, dx=0, dy=0, width=1):
    # evaluation of surface deformation extracted from EXP_ID=142
    upper = exp_f(ta,-6.17761515e-05,  1.78422769e+00,  7.40139158e-05)
    lower = exp_f(ta,0.00377156,  1.45234773, -0.00326456)
    a     = pol2_f(ta,0.31294945, -0.00963803,  2.6743828)
    b     = pol2_f(ta,38.56906702,  -1.6278976 , 453.87937763)
    r = np.sqrt((x-dx)**2 + (y-dy)**2)/width
    
    # scaled logistic function describing surface deformation
    return lower + (upper - lower) / (1 + np.exp(a - b * r))