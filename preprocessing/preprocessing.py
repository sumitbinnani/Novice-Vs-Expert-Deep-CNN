
# coding: utf-8

# ## Importing Required Libraries

# In[1]:

import os
import glob
import scipy.io as sio
import numpy as np
from skimage.transform import *
from skimage.io import *
import tqdm
import warnings
warnings.filterwarnings("ignore")


# # Setting Variables

# In[2]:

IMAGE_PATH = "../images/*/*.jpg"
ANNOTATION_PATH = "../annotations-mat"

output_shape = (224, 224)
mode = {'mode': 'constant'}
OUTPUT_PATH = "../resized_images_padded_0"
try:
    os.stat(OUTPUT_PATH)
except:
    os.makedirs(OUTPUT_PATH)


# # Looping Through the Images

# In[3]:

import matplotlib.pyplot as plt

for x in tqdm.tqdm(glob.glob(IMAGE_PATH)):
    img_name = os.path.basename(x)
    mat_name = img_name.replace(".jpg", ".mat")
    dir_name = os.path.basename(os.path.dirname(x))
    mat_path = os.path.join(ANNOTATION_PATH, dir_name, mat_name)
    
    image = imread(x)
    mat = sio.loadmat(mat_path)
    left, top, right, bottom = [mat['bbox'][0][0][i][0][0] for i in range(4)]    
    
    max_pad = max(int(right - left), int(bottom - top))
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    r = np.pad(r, max_pad, **mode)
    g = np.pad(g, max_pad, **mode)
    b = np.pad(b, max_pad, **mode)
    padded_image = np.dstack([r, g, b])
    
    center_hor = left + max_pad + int((right - left) / 2 + 0.5)
    center_ver = top + max_pad + int((bottom - top) / 2 + 0.5)
    max_pad = int(max_pad / 2 + 0.5)
    cropped_image = padded_image[center_ver - max_pad: center_ver + max_pad, center_hor - max_pad: center_hor + max_pad]
    
    resized_image = resize(cropped_image, output_shape)
    
    try:
        imsave(os.path.join(OUTPUT_PATH, dir_name, img_name), resized_image)
    except:
        os.makedirs(os.path.join(OUTPUT_PATH, dir_name))
        imsave(os.path.join(OUTPUT_PATH, dir_name, img_name), resized_image)

