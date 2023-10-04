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
from func_fl import *



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

export_data = 'y'  # whether to export data or not
export_file_name = 'dataset_x_fl_out.csv'
fit_line = 'y'     # whether or not to fit a line through the data and plot

# =============================================================================



# =============================================================================
# Definitions
# =============================================================================
# x_data =  [100.0,200.0,300.0]    # x-axis variable (sampele types/datasets)
# xp_data = [50,100,400]           # x-axis data for plotting the fit function

# x_data2 =  [50.0, 100.0, 150.0]  # x-axis variable (sampele types/datasets)

# x_data =  [50.0, 100.0, 150.0]    # x-axis variable (sampele types/datasets)
# xp_data = [25,100,200]           # x-axis data for plotting the fit function



x_data =  [1,2,3]    # x-axis variable (sampele types/datasets)
xp_data = [0,2,4]           # x-axis data for plotting the fit function
x_data2 = x_data

# th_factor_set = [0.6, 0.6, 0.5]
# th_factor_set = [0.4, 0.4, 0.4]
th_factor_set = [0.4, 0.4, 0.4, 0.6, 0.6, 0.5]


path_main = 'Data/'

# folder = ['50x3']
# folder = ['100_100','100_200', '100_300']
# folder = ['50_50','50_100', '50_150']
# folder = ['100_100','100_200', '100_300','100_100c','100_200c', '100_300c']
# folder = ['100_100','100_200', '100_300','50_50','50_100', '50_150']
# folder = ['50x1','50x2', '50x3']
# folder = ['100x1','100x2', '100x3']
# folder = ['100x2']

folder = ['100x1','100x2', '100x3','50x1','50x2', '50x3']
# scale_factor = [1]
# scale_factor = [4,8]
# scale_factor = [1,4,8]
# scale_factor = [1,4,8,1,1,1]
# scale_factor = [1,1,1]
scale_factor = [1,1,1,1,1,1]




# ttl = 'Benchtop reaction with 50mers'
# ttl = 'Benchtop reaction with 100mers'
ttl = 'Benchtop reactions'
lb1 = '100mer'
lb2 = '50mer'
# lb1 = '50mer'



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
d_max = 0
d_min = 1e8
for m in range(len(folder)):
    path_full = path_main + folder[m] + '/*.tif'        # full path
    
    # analyze_images() will process all images in the folder specified by path_full
    # it will return a list that is equal to the total number of beads found in all the images
    # in that folder. Note that, this will be >= number of images in that folder
    
    temp = np.array(analyze_images(path_full,path_out + '/' + folder[m], th_factor_set[m]))*scale_factor[m]           # analyze the data set images and multiply with the scale factor
    
    fl_data_store.append(temp)  # store bead brightness values              
    fl_data_mean.append(np.mean(fl_data_store[m]))   # store average brightness per image
    
    d_max = max(d_max, np.max(fl_data_store[m]))
    d_min = min(d_min, np.min(fl_data_store[m]))
    

print_log('============================================================\n\n\n')




# =============================================================================
# Print information
# =============================================================================
for m in range(len(folder)):
    print_log('Number of beads found in %s dataset = %i' %(folder[m], len(fl_data_store[m])))
   
# =============================================================================



# =============================================================================
# Plotting mean values and fit lines
# =============================================================================

if fit_line == 'y':
    # =============================================================================
    # Split data (full images and cropped datasets) and separate curve fittings for each split
    # =============================================================================
    ydata1 = fl_data_mean[0:3]
    ydata2 = fl_data_mean[3:6]
    # ydata1 = fl_data_mean
    
    
    popt1, pcov1 = curve_fit(fit_func, x_data, ydata1)
    popt2, pcov2 = curve_fit(fit_func, x_data2, ydata2)
    
    # =============================================================================
    
    
    
    py.figure()
    
    
    py.plot(xp_data, fit_func(xp_data, *popt1), color = '#ff9001')
    py.plot(xp_data, fit_func(xp_data, *popt2), color = '#7d98a1')
    
    py.plot(x_data, ydata1,'o',color = '#d35100', markeredgecolor = '#151515', label = lb1)
    py.plot(x_data2, ydata2,'s',color = '#3F6385', markeredgecolor = '#151515', label = lb2)
    # py.xlim([0,320])
    # py.ylim([0,680])
    py.legend()
    
    py.xlabel('Cycle number')
    py.ylabel('Intensity (a.u)')
    py.title(ttl)
    
    
# =============================================================================





# =============================================================================
# Plotting histograms
# =============================================================================
clr = py.cm.Pastel2.colors

binwidth_set = [40,80,80, 40,80,80]
py.figure()
for m in range(len(folder)):
    data = fl_data_store[m]
    binwidth = binwidth_set[m]
    py.subplot(len(folder),1,m+1)
    py.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth), edgecolor = 'k',alpha = 0.5,label = folder[m],orientation = 'vertical', color = clr[m])
    py.ylabel('Count')
    py.xlabel('Intensity (a.u.)')
    py.legend()
    py.xlim([d_min, d_max])

# =============================================================================


# =============================================================================
# Export data
# =============================================================================
if export_data == 'y':
    df = pd.DataFrame(fl_data_store).T
    df.columns = folder
    df.to_csv(export_file_name)
# =============================================================================

el_t = time.time()- st_t
print_log('\nRun time = %.1f sec \n' % el_t)




