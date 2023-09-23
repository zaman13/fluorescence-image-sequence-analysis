#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:25:07 2023

@author: Mohammad Asif Zaman

Sept, 22, 2023:
    - analyze_images(path) returns brightess values of each individual bead found in
      the images files within the folder specified by path
    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import os

from scipy.optimize import curve_fit


# import custom functions
from func_fl_v2_2 import *


# =============================================================================
# Fitting function
# =============================================================================
def fit_func(x, m, c):
    x = np.array(x)
    return m*x + c
# =============================================================================




# =============================================================================
# Definitions
# =============================================================================
x_data =  [100.0,200.0,300.0]    # x-axis variable (sampele types/datasets)
xp_data = [50,100,400]           # x-axis data for plotting the fit function

path_main = 'Data/'

# folder = ['100_100','100_200', '100_300','100_100c','100_200c', '100_300c']
# scale_factor = [1,4,8,1,4,8]

folder = ['test']
scale_factor = [1]



# Make output folders if they don't exist
path_out = 'Data_out'
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



# folder = ['100_100','100_200', '100_300']
# folder = ['100_100c','100_200c', '100_300c']
# folder = ['100_100']



# Initialize empty lists to store data
fl_data_store = []       # store avergae brightness of each individual bead
fl_data_mean = []        # mean bead brightness in each image (which may contain many beads)


# loop over all the image folders
for m in range(len(folder)):
    path_full = path_main + folder[m] + '/*.tif'        # full path
    
    # analyze_images() will process all images in the folder specified by path_full
    # it will return a list that is equal to the total number of beads found in all the images
    # in that folder. Note that, this will be >= number of images in that folder
    
    temp = np.array(analyze_images(path_full,path_out + '/' + folder[m]))*scale_factor[m]           # analyze the data set images and multiply with the scale factor
    
    fl_data_store.append(temp)  # store bead brightness values              
    fl_data_mean.append(np.mean(fl_data_store[m]))   # store average brightness per image
    

print('============================================================\n\n\n')




# =============================================================================
# Print information
# =============================================================================
for m in range(len(folder)):
    print('Number of beads found in %s dataset = %i' %(folder[m], len(fl_data_store[m])))
   
# =============================================================================



# =============================================================================
# Split data (full images and cropped datasets) and separate curve fittings for each split
# =============================================================================
ydata1 = fl_data_mean[0:3]
ydata2 = fl_data_mean[3:6]


popt1, pcov1 = curve_fit(fit_func, x_data, ydata1)
popt2, pcov2 = curve_fit(fit_func, x_data, ydata2)


# =============================================================================


# =============================================================================
# Plotting mean values and fit lines
# =============================================================================

py.figure()


py.plot(xp_data, fit_func(xp_data, *popt1), color = '#ff9001')
py.plot(xp_data, fit_func(xp_data, *popt2), color = '#7d98a1')

py.plot(x_data, ydata1,'o',color = '#d35100', markeredgecolor = '#151515', label = 'Full image sets')
py.plot(x_data, ydata2,'s',color = '#3F6385', markeredgecolor = '#151515', label = 'Cropped image sets')
py.xlim([0,320])
py.ylim([0,680])
py.legend()

py.xlabel('Strand length')
py.ylabel('Intensity (a.u)')
py.title('Benchtop reaction with 100 mers')

# =============================================================================


# =============================================================================
# Plotting histograms
# =============================================================================
py.figure()
for m in range(3):
    data = fl_data_store[m]
    binwidth = 20
    py.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth), edgecolor = 'k',alpha = 0.5,label = folder[m],orientation = 'horizontal')
    py.xlabel('Count')
    py.ylabel('Intensity (a.u.)')
    py.legend()


# =============================================================================