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


# import custom functions
from func_fl_single_img_t3 import *



st_t = time.time()



# =============================================================================
# Function for logging console output to file
def print_log( *args):
    file = open('log.txt','a')
    toprint = ' '.join([str(arg) for arg in args]) 
    print(toprint)
    toprint = toprint + '\n'
    file.write(toprint)
    file.close()


# =============================================================================
# Fitting function
# =============================================================================
def fit_func(x, m, c):
    x = np.array(x)
    return m*x + c
# =============================================================================

def single_img_analysis(fpath):
    temp =  np.array(analyze_images(path_full,path_out + '/' + folder[m], th_factor_set[m]))*scale_factor[m]           # analyze the data set images and multiply with the scale factor
    
    print_log('Finsihed analysis. Back to main.')
    py.figure()
    py.axis('off')
    py.axis('tight')
    # py.xticks(np.linspace(1,len(temp)))
    
    
    # py.bar(np.linspace(0,len(temp)-1, len(temp)), temp)
    
    x = np.linspace(0,len(temp)-1, len(temp))
    print(len(x))
    print(len(temp))
    df = pd.DataFrame({'index': x, 'mean ': temp})
    py.table(cellText=np.round(df.values,1),  colLabels = df.columns, loc='center')
    return temp

# Start log with current date time
now = datetime.datetime.now()
print_log('******************************************************************')
print_log('******************************************************************')
print_log(now)
print_log('******************************************************************')
print_log('******************************************************************')
# =============================================================================
# Control parameters
# =============================================================================

export_data = 'n'  # whether to export data or not
export_file_name = 'dataset_x_fl_out.csv'
fit_line = 'n'     # whether or not to fit a line through the data and plot

# =============================================================================



# =============================================================================
# Definitions
# =============================================================================



th_factor_set = [0.6]


path_main = ''
folder = ['Test img']


scale_factor = [1]
# scale_factor = [4,8]
# scale_factor = [1,4,8]
# scale_factor = [1,4,8,1,1,1]
# scale_factor = [1,1,1]
# scale_factor = [1,1,1,1,1,1]






# Make output folders if they don't exist
path_out = 'Test_out'
isExist = os.path.exists(path_out)
if not isExist:
   os.makedirs(path_out)

for fld in folder:
    path_out_folder = 'Data_out/'  + fld
    isExist = os.path.exists(path_out_folder)
    if not isExist:
       os.makedirs(path_out_folder)

# =============================================================================

py.close('all')






# Initialize empty lists to store data
fl_data_store = []       # store avergae brightness of each individual bead
fl_data_mean = []        # mean bead brightness in each image (which may contain many beads)



# path_full = path_main + folder[m] + '/*.tif'        # full path
m = 0
path_full = path_main + folder[m] + '/1.tif'        # full path


# analyze_images() will process all images in the folder specified by path_full
# it will return a list that is equal to the total number of beads found in all the images
# in that folder. Note that, this will be >= number of images in that folder

temp = single_img_analysis(path_full)

fl_data_store.append(temp)  # store bead brightness values              
fl_data_mean.append(np.mean(fl_data_store[m]))   # store average brightness per image

d_min = 0
d_max = 1e8
d_max = max(d_max, np.max(fl_data_store[m]))
d_min = min(d_min, np.min(fl_data_store[m]))

el_t = time.time()- st_t
print_log('\nRun time = %.1f sec \n' % el_t)

print_log('============================================================\n\n\n')






