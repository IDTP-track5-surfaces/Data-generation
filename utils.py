import numpy as np
from scipy.interpolate import interp2d, griddata
from scipy.fft import ifft2
from scipy.ndimage import sobel, convolve, laplace
from scipy.signal import convolve2d
from tqdm import tqdm


# Function to write NPY file
def write_npy(data, filename):
    np.save(filename, data)
    
# Function to simulate distortion
def simulate(img, warp_x, warp_y, is_shown=False, n_frame=None):
    h, w, n_channel = img.shape

    X, Y = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))

    x_c = warp_x
    y_c = warp_y

    Xnew = np.reshape(X + x_c, (h * w, 1))
    Ynew = np.reshape(Y + y_c, (h * w, 1))

    valid = (Xnew >= 1) & (Xnew <= w) & (Ynew >= 1) & (Ynew <= h)

    img_curr = np.zeros((h, w, n_channel))
    curr_frame = np.zeros(h * w)

    for k in range(n_channel):
        curr_frame[valid] = interp2d(img[:, :, k], Xnew[valid], Ynew[valid])
        img_curr[:, :, k] = np.reshape(curr_frame, (h, w))

    return img_curr

# Function to perform ray tracing
def raytracing_im_generator_ST(im_rgb, this_depth, n1, n2, x, y):
    step = 1
    h, w, dim = im_rgb.shape
    # h, w = im_rgb.shape

    Gx = sobel(this_depth, axis=0)
    Gy = sobel(this_depth, axis=1)

    normal_ori = np.ones_like(im_rgb)
    normal_ori[:, :, 0] = -Gx
    normal_ori[:, :, 1] = -Gy
    
    norm = np.sqrt(Gx**2 + Gy**2 + 1)
    normal = normal_ori / norm[..., None]

    s1 = np.zeros_like(normal)
    s1[:, :, 2] = -1
    s2 = refraction(normal, s1, n1, n2)

    a = this_depth / s2[:, :, 2]
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


# def refraction(normal, s, n1, n2):
#     n = n1 / n2
#     dot_n_s = normal[:, :, 0] * s[:, :, 0] + normal[:, :, 1] * s[:, :, 1] + normal[:, :, 2] * s[:, :, 2]

#     cos_theta_t = np.sqrt(1 - n**2 * (1 - dot_n_s**2))
#     refraction_vec = n * s + (n * dot_n_s - cos_theta_t) * normal

#     return refraction_vec

# def reflection(normal, s, n1, n2):
#     dot_n_s = normal[:, :, 0] * s[:, :, 0] + normal[:, :, 1] * s[:, :, 1] + normal[:, :, 2] * s[:, :, 2]

#     reflection_vec = s + 2*-dot_n_s*normal

#     return reflection_vec

# Example usage
# im_rgb = np.random.rand(128, 128, 3)
# this_depth = np.random.rand(128, 128)
# n1 = 1.0
# n2 = 1.33
# x, y = np.meshgrid(np.arange(1, 129), np.arange(1, 129))
# warp_map = raytracing_im_generator_ST(im_rgb, this_depth, n1, n2, x, y)
