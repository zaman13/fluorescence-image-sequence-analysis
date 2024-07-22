#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:25:07 2023

@author: Mohammad Asif Zaman

Sept, 22, 2023:
    - analyze_images(path) returns brightess values of each individual bead found in
      the images files within the folder specified by path

Sept. 25-26, 2023:
    - after adaptive thresholding, the th_factor_set is just a place holder. The thresholding algorithm
      does not use these values. However, the parameter is stil kept for back compatibility (simple 
      thresholding can still be used if desired)    
    
    - Color cycling in subplots
    
    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import os
import cv2
import time
import datetime


from scipy.optimize import curve_fit
# from ascii_magic import AsciiArt, Back

st = time.time()

# import custom functions
from func_fl_single_img import *







    
    




# =============================================================================
# Definitions
# =============================================================================



th_factor = 0.2
block_size = 81

path_main = ''
folder = ['Test img']


scale_factor = [1]
# scale_factor = [4,8]
# scale_factor = [1,4,8]
# scale_factor = [1,4,8,1,1,1]
# scale_factor = [1,1,1]
# scale_factor = [1,1,1,1,1,1]






# # Make output folders if they don't exist
# path_out = 'Test_out'
# isExist = os.path.exists(path_out)
# if not isExist:
#    os.makedirs(path_out)

# for fld in folder:
#     path_out_folder = 'Data_out/'  + fld
#     isExist = os.path.exists(path_out_folder)
#     if not isExist:
#        os.makedirs(path_out_folder)

# =============================================================================

py.close('all')
cmp = 'magma'








# path_full = path_main + folder[m] + '/*.tif'        # full path
m = 0
# path_full = path_main + folder[m] + '/1.tif'        # full path
path_full = path_main + folder[m] + '/2.tif'        # full path
path_out = path_main + folder[m]


img = cv2.imread(path_full, cv2.IMREAD_UNCHANGED)       # Assume the image is 16bit

if img.dtype == 'uint8':                            # if the image is 8 bit
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)   # then read it differently
    


   
# Display original image 
py.figure()
py.imshow(img, cmap = cmp)
# py.clim(bmn,bmx)
py.clim(0,2000)
py.axis('off')

# py.suptitle(fpath + '\n' + 'Image (B) min, max = ' + str(bmn) + ', ' +  str(bmx))   # print sourcce file and image min-max in the title


# analyze_images() will process all images in the folder specified by path_full
# it will return a list that is equal to the total number of beads found in all the images
# in that folder. Note that, this will be >= number of images in that folder

o_mean, o_size, b_mean, x_store, y_store, iB_store = single_img_analysis(path_full, path_out, block_size, th_factor)

py.show()

en = time.time()


print('Run time = %1.2f sec' %(en-st))
# my_art = AsciiArt.from_image(path_out + '/out_crop.png')
# my_output = my_art.to_ascii(columns=200, monochrome = True)
# print(my_output)



