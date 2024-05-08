import numpy as np
from scipy.interpolate import griddata

   
def raytracing_im_generator_ST(normal, depth_map, transparent = "refraction", n1=1.0, n2=1.33):
    """A raytracing model that creates a warp map that can be used to generate a warped image based on either reflection or refraction. 

    Args:
        normal (array_like): (n,n,3)-array which contains the normal vectors for each point on a n x n grid.
        depth_map (array_like): (n,n,1)-array which contains the height of the fluid for each point on a n x n grid.
        transparent {"refraction", "reflection"}: Selector for wether the raytracing model should use refraction or reflection. Defaults to "refraction".
        n1 (float, optional): Refractive index of incident medium. Not necesarry for reflection. Defaults to 1.0 for air.
        n2 (float, optional): Refractive index of transmitted medium. Not necesarry for reflection. Defaults to 1.33 for water.

    Returns:
        warp_map (array_like): (n,n,2)-array that contains how much each pixel should be shifted for both the x and y coordinates.
    """
    width, height = depth_map.shape
    
    # Scaling factor to scale from real distances to pixels
    scaling = width/(2*52e-3) 
    
    # Create incident vectors
    s1 = np.zeros_like(normal)
    s1[:, :, 2] = -1
    
    if transparent == "refraction":
        s2 = refraction(normal, s1, n1, n2)
        
        # Scale unit vectors to pixels    
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
        
        # Scaling factor for setup   
        X,_ = np.meshgrid(np.arange(width),np.arange(width))
        d = .5*52e-3 
        l = d/np.sin(theta) + X*np.sin(theta)/scaling

        # Orthogonal projection for shift vector
        r = np.abs(s1)
        w = -l[...,None]*(s2 - r*np.einsum('ijk, ijk->ij', s2, r)[..., None])*scaling

        # Map shift vector to shift in pixels
        x_c = w[:,:,0]
        y_c = np.sqrt(w[:,:,1]**2 + w[:,:,2]**2)
    else:
        print("Not implemented")


    warp_map = np.dstack((x_c, y_c))

    return warp_map


def refraction(normal, s1, n1, n2):
    """Refraction model that calculates the refracted unit vectors for a n x n  grid with the use of Snell's law.

    Args:
        normal (array_like): (n,n,3)-array which contains the normal vectors for each point on a n x n grid.
        s1 (array_like): (n,n,3)-array which contains the incident vectors for each point on a n x n grid.
        n1 (float): Refractive index of incident medium.
        n2 (float): Refractive index of transmitted medium.

    Returns:
        s2_normalized (array_like): (n,n,3)-array which contains the normalized refrected vectors for each point on a n x n grid.
        
    Notes:
        See https://en.wikipedia.org/wiki/Snell%27s_law for detailed discription of Snell's law and its vector form.
    """
    n = n1/n2
    
    # dot product of -normal and s1
    cos1 = np.einsum('ijk, ijk->ij', -normal, s1)
    
    cos2 = np.sqrt(1-n**2*(1-cos1**2))
    s2 = n*s1 + (n*cos1-cos2)[...,None]*normal
    
    s2_normalized = s2/np.linalg.norm(s2, axis =2)[..., None]
    
    return s2_normalized


def reflection(normal, s1):
    """Reflection model that calculates the reflected unit vectors for a n x n  grid by specular reflection.

    Args:
        normal (array_like): (n,n,3)-array which contains the normal vectors for each point on a n x n grid.
        s1 (array_like): (n,n,3)-array which contains the incident vectors for each point on a n x n grid.

    Returns:
       s2_normalized (array_like): (n,n,3)-array which contains the normalized reflected vectors for each point on a n x n grid.
    """
    # Dot product of -normal and s1
    cos1 = np.einsum('ijk, ijk->ij', -normal, s1)

    s2 = s1 + 2*cos1[...,None]*normal

    s2_normalized = s2/np.linalg.norm(s2, axis =2)[..., None]
    
    return s2_normalized


def deform_image(img, warp_map):
    """Function to deform a reference image according to a warp map.

    Args:
        img (array_like): A 3D array containing float values of an image. The last axis are the channels of the images
        warp_map (array_like): (n,n,2)-array that contains how much each pixel should be shifted for both the x and y coordinates.

    Returns:
        imgCurr (array_like): A 3D array containing float values of the deformed image. 
    """
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
        # Interpolate to align shifted pixel values to correct pixels. 
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