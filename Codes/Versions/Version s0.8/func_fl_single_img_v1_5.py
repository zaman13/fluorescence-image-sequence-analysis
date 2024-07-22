# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 21:47:28 2023

@author: Mohammad Asif Zaman

July 13, 2023
    - Added mask_fragment() function to identify objects in a frame individually. The function
      creates submasks for each identified object. 
    - Parameterized the median blur filter order.


Sept. 25, 2023
    - Included adaptive thresholding

Sept. 26, 2023
    - Fixed cmap range issues when plotting and exporting images
    - Ehnanced low light detection sensitivity

Sept. 28, 2023
    - Added labels on the submasks    
    
Sept. 29, 2023
    - Added skew rejection to remvoe noise   
    
Oct. 11, 2023
    - Auto square grid when subplotting individual beads
    - Auto distinguish between 8bit and 16bit images

March 19, 2024
    - Convert code to analyze a single image. Loop over images in the wrap (if needed)

April 2024
    - Added statistics function
    
    
May 2, 2024
    - Added object size (area) as an output
    - Added object location coordinate (x,y) as an output
    - Print data in a table format with object label, mean intensity, size and coordinate. Also tabulated image mean and background mean.

May 7, 2024
    - Exported coordinates of bounding box (iB) for annotation in the main program
    - Note that export of (x,y) coordinates and object size is redundant now as they can be calculated from iB values. However, they are still kept as output for now.

July 8, 2024
    - Added data export and histogram image export
    - Included median and mode in the export data matrix


July 20, 2024
    - Auto invert of adaptive threshold for analyzing bright field images. This is based upon the th_factor parameter.
      th_factor > 0 for fluorescence images, and th_factor < 0 for bright field images.
    - Now using sparse matrix to represent the sub-masks. This should save a lot of memory
    
    
"""

# Load libraries
import numpy as np
import matplotlib.pyplot as py
import cv2
import pandas as pd 


from PIL import Image as PIL_Image

from skimage import measure
from scipy.stats import kurtosis, skew
from scipy.sparse import csr_matrix

import glob

import random


import os





# =============================================================================
# Function for logging console output to file
# =============================================================================
def print_log( *args):
    file = open('log.txt','a')
    toprint = ' '.join([str(arg) for arg in args]) 
    print(toprint)
    toprint = toprint + '\n'
    file.write(toprint)
    file.close()

# =============================================================================

# =============================================================================
# Functions to define image mask
# =============================================================================

# =============================================================================
# simple threshold
def object_mask_find(img,blur_order,th_factor):
    im_temp = img                             # input image matrix
    im_blur = cv2.medianBlur(im_temp, blur_order)      # Blurring to reduce noise during boundary/mask creation
    mx = np.max(im_blur)
    mn = np.min(im_blur)
    th = mx*th_factor                         # Threshold value for finding bead mask
    im_mask = (im_blur > th)                  # bead mask
    return im_mask
# =============================================================================


# =============================================================================
# Adaptive thresholding
def adaptive_threshold_16bit(img, blur_order, block_size, mean_th_fct):
    imb = cv2.medianBlur(img, blur_order)  # blur image to get rid of noise and hot pixels
    imgM = np.array(imb)
    # =========================================================================
    # Calculate block average
    # =========================================================================
    # The cv2.blur() funciton is basically a filter with averaging kernel.
    # https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html#gsc.tab=0
    blur_avg = cv2.blur(imb,(block_size,block_size)) 
    # =========================================================================
    # mask = imgM > (blur_avg - const_C)
    
    mask = imgM > ((1+mean_th_fct)*blur_avg) 
    
    # July 20, 2024: Smart masking.Invert mask automatically depending on whether most of the region of the mask is bright or dark.
    # In this way, the algorithm can automatically detect between brightfield or fluorescence image and adjust the mask accordingly.
    # This "smart" method will probably cause issues. Much better to use a dumber approach.
    
    # if sum(sum(mask)) > np.size(mask)/2: # if there are more ones than zeros
        # mask = np.invert(mask)
    
    # July 20, 2024: More simple version
    if mean_th_fct < 0:
        mask = np.invert(mask)
        
    return mask
# =============================================================================


# =============================================================================
# Function to split/fragment the image mask into individual submasks, each submask 
# corresponding to one object 
# =============================================================================
def mask_fragment(img,obj_size_th_factor ):
    # note, function input is the binary image mask. img has to be binary valued matrix.
    
    #https://stackoverflow.com/questions/49771746/numpy-2d-array-get-indices-of-all-entries-that-are-connected-and-share-the-same

    img_labeled = measure.label(img, connectivity=1)
    
    idx = [np.where(img_labeled == label)
            for label in np.unique(img_labeled)
            if label]
  
    N_obj = len(idx)
    print_log('Number of objects identified, N_obj = %i' %(N_obj))   
    
    if N_obj == 0:
        print_log('================ Mask fragmant error =============\n')
        return [], [], []
    
    # bboxes = [area.bbox for area in measure.regionprops(img_labeled)]
    
    
    uu = np.zeros(np.shape(img)) # numpy image array
 
    
    
    
    # sub_masks = np.zeros([np.shape(img)[0],np.shape(img)[1],N_obj])  # array of image arrays. A mask for each detected object.
    sub_masks = []
    obj_size_sum = np.zeros(N_obj)     # initialization. The array will store size of each submask.
    
    for m in range(N_obj):             # loop through each object
        uu = uu*0                      # reset uu array for reuse
        uu[idx[m]] = 1                 # populate uu   
        obj_size_sum[m] = np.sum(uu)   # size of the object. Defined by the number of ones.
        # sub_masks[:,:,m] = uu          # store the submask
        
        sub_masks.append(csr_matrix(uu))
        
        # py.subplot(3,int(np.ceil(N_obj/3)),m+1)
        # py.imshow(sub_masks[:,:,m])
        # py.axis('off')
    
    # Once we have splitted the mask into submasks for individual objects, we can
    # start checking the size of each object independently. If a detected object
    # is too small, we can neglect it. We start by setting a threshold object size.
    # Then we refine our submask set and only pick the submasks that are larger in size
    # than the threshould value. Once that is done, we recombine the submasks to find
    # a new refined composite mask. 
    
    
    
    # Size thresholding
    # =============================================================================

    # obj_size_sum_th = np.max(obj_size_sum)*obj_size_th_factor        # threshold value for object size (i.e. threshold size)
    # Use mean size instead of max size for thresholding
    obj_size_sum_th = np.mean(obj_size_sum)*obj_size_th_factor        # threshold value for object size (i.e. threshold size)
    
    N_obj_refined = sum(obj_size_sum > obj_size_sum_th)              # find the number of objects having length larger than threshold size

    print_log('Number of objects refined, N_obj_refined = %i' %(N_obj_refined))
    
  
    
    # sub_masks_refined = np.zeros([np.shape(img)[0],np.shape(img)[1],N_obj_refined])  # initialize array of image arrays where submasks will be stored
    sub_masks_refined = []
    mask_refined = np.zeros(np.shape(img))   # initialize composite mask from refined sub-masks 
    # py.figure()
    
    counter = 0
    for m in range(N_obj):   # loop over all objects
        
        if obj_size_sum[m] > obj_size_sum_th:   # for objects that meet threshold size
            # sub_masks_refined[:,:,counter] = sub_masks[:,:,m]             # store refined sub-mask
            sub_masks_refined.append(sub_masks[m])
            
            mask_refined = mask_refined + sub_masks_refined[counter]  # superimpose on the refined mask
             
            # note that mask refined is still a binary matrix as the sub_masks do not overlap (no element will be larger than one)
        
            # py.subplot(3,int(np.ceil(N_obj_refined/3)),counter+1)
            # py.imshow(sub_masks_refined[:,:,counter])
            # py.axis('off')
                         
            counter = counter + 1
    
    return sub_masks, sub_masks_refined, mask_refined

# =============================================================================

def skew_refine(smasks):
    
    sk_th = 0.1
    N = np.shape(smasks)[2]
    N_refined = 0
    ind = []
    for m in range(N):
        ims = smasks[:,:,m]                # select a submask
        
        iB = bbox2(ims)                     # calculate bounding box around the detected bead
                                           # calculate bounding box around the detected bead
        
        im_crop = ims[iB[0]:iB[1],iB[2]:iB[3]]  
        
        sk = skew(im_crop,axis = None) 
        if sk < sk_th:
            N_refined = N_refined + 1 
            ind.append(m)
        else:
            print('Skew rejection, s = %1.2f' % sk)
            
    smask_r = np.zeros([np.shape(ims)[0],np.shape(ims)[1],N_refined])  # initialize array of image arrays where submasks will be stored
    mask_r = np.zeros(np.shape(ims))
    for m in range(N_refined):
        smask_r[:,:,m] = smasks[:,:,ind[m]]
        mask_r = mask_r + smask_r[:,:,m]
   
    print('Number of objects refined (K), N_obj_refined = %i' %(N_refined))

    return smask_r, mask_r
# =============================================================================



# =============================================================================

def bbox1(img):
    im_box_border = 4                 # Amount of offset pixesl at each side when displaying image of a single bead 
    
    a = np.where(img != 0)
    bbox = np.min(a[0])-im_box_border, np.max(a[0])+im_box_border, np.min(a[1])-im_box_border, np.max(a[1])+im_box_border
    return bbox
# =============================================================================

def bbox2(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox
# =============================================================================




# =============================================================================
def calc_stats(img1):
    
    t_vals, t_counts = np.unique(img1, return_counts=True)   # count and list the number of unique entries. Needed for calculating mode.
    
    # i_min = np.min(img1)                        # min of the masked region
    # i_max = np.max(img1)                        # max of the masked region
    i_mean = np.mean(img1)                      # mean of the masked region
    i_median = np.median(img1)                  # median of the masked region
    i_mode = t_vals[np.argwhere(t_counts == np.max(t_counts))]  # mode of the masked region
        
    # Make sure i_mode retruns only one value. i.e., i_mode[0]
    # return i_mean, i_median, i_mode[0], i_min, i_max

    return i_mean, i_median, i_mode[0]


# =============================================================================


# =============================================================================
# Main analysis function
# =============================================================================

def analyze_images(fpath, path_out, block_size, th_factor):
    
    
    # =============================================================================
    # Parameters of the tiff image
    # =============================================================================
    # path = '/home/asif/Dropbox/Codes/Python/Fluorescence data analysis/Data/100_100'
    # path_ext = '/home/asif/Dropbox/Codes/Python/Fluorescence data analysis/Data/100_100/*.tif'
    
    
    # returns mean of each submask (a list), size of each submask (a list),  and the mean of the background (a single number)
    
    # Mod: May 2, 2024: now it retursn the size of the submask along with mean and background mean.
    
    print_log('\n\n==================================================================')
    
  
    
    
    # =============================================================================
    
    
    
    
    # =============================================================================
    # Control parameter
    # =============================================================================
   
                                      
    # th_factor = 0.6                  # Fraction of maximum brightness. Pixels having this brightness fraction is assumed to be part of the object
    
    # N_plot_row = 3                    # Number of rows in the subplot when plotting the detected object at each frame
    blur_order = 5                    # Order of the median blur
    obj_size_th_factor = (0.5)**2     # for composite mask from fragments, size of object to include = max_size * obj_size_th_factor. Note that obj_size is defined in terms of 2D pixel count which is equivalent to area. (0.2 before)
    
    
    cmp = 'viridis'
    # =============================================================================
    
    
    
    # =============================================================================
    # initialize empty lists/arrays
    # =============================================================================
       
    mean_store = []   # list that will contain mean of each detected object (from submask)
    median_store = [] # list that will contain median of each detected object (from submask)
    mode_store = []   # list that will contain mode of each detected object (from submask)
    
    size_store = []   # list to store the size of each detected object (from submask)
    x_store = []      # store x position of each detected object (from submask)
    y_store = []      # store y position of each detected object (from submask)
    
    iB_store = []     # store the coordinates of the bounding box
    
    # note that the x and y poistion is saved from the bounding box coordinate. Hence, there will be an offset from the top left edge of the object. These values
    # should not be taken as the center of each object! They are top left position of the object with some added offset (should be equal to im_box_border variable defined in the
    # bbox1() function)
    # =============================================================================
    
    
           
 
    # Dynamic name
    name = fpath
 
    # Shape of Image
    
    
    # Read image files
    # =============================================================================
    
    
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)       # Assume the image is 16bit
    print_log('img.dtype = %s' % img.dtype)
    if img.dtype == 'uint8':                            # if the image is 8 bit
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)   # then read it differently
        
        
    imb = cv2.medianBlur(img, blur_order)    # blurred image to extract max min values (gets rid of hot-pixel spike)
    bmx = np.max(imb)                        # max of blurred image
    bmn = np.min(imb)                        # min of blurred image
    
    print_log('\n')
    print_log(name, 'Shape :', img.shape)
    
    
    # parameters for object mask finding algorithm function    
    # block_size = 81
    # mean_th_fct = 0.1
    mean_th_fct = th_factor
    
    
    # im_mask = object_mask_find(img,blur_order,th_factor)                     # object mask determined from the current image
    
    im_mask = adaptive_threshold_16bit(img, blur_order, block_size, mean_th_fct)
    
    
    
    im_bck = img*(im_mask < 1)    # mask the background and turn everything else to zero
       
    ind_bck = np.where(im_mask < 1)         # find indices of all zero elements in the masked image. This should be the indices of the background image
    
    b_mean, b_mediam, b_mode = calc_stats(im_bck[ind_bck])
    
    
    print_log('Masking done')
    
    if np.sum(im_mask) == 0:
        print_log('\n================File Error=============\n')
        print_log(fpath)
        print_log('Skipping file and moving to next file \n\n')
        # mean_store, median_store, mode_store, size_store, b_mean, x_store, y_store, iB_store
        return 0, 0, 0, 0, 0, 0, 0, 0
    
    smasks, smasks_f, mask_r = mask_fragment(im_mask,obj_size_th_factor )    # mask fragment and composite reforming
    
    # smasks_f, mask_r = skew_refine(smasks_f)
                 
    # print_log(np.shape(smasks_f[:,:,0]))
    print_log('Submasking done')
    Nsub_masks = len(smasks_f)
    
    N_plot_col = int(np.ceil(np.sqrt(Nsub_masks))) 
    N_plot_row = int(np.ceil(Nsub_masks/N_plot_col))

    
    # Display original image 
    py.figure(36)
    py.clf()
    py.imshow(img, cmap = cmp)
    py.clim(bmn,bmx)
    # py.clim(700,2000)
    py.axis('off')
    py.suptitle(fpath + '\n' + 'Image (B) min, max = ' + str(bmn) + ', ' +  str(bmx))   # print sourcce file and image min-max in the title
    
    # py.figure(37)
    # py.clf()
    # fg, ax = py.subplot(N_plot_row, N_plot_col,m+1)
    # fg, ax = py.subplots(N_plot_row, N_plot_col)
    # Loop over the submaks (each submask represents one detected bead in the frame)
    print_log('Looping over submasks...')
    
    
    
    # loop over each submask, each submask represents one detected object
    for m in range(Nsub_masks):
        
        im_mask = np.array(smasks_f[m].todense())                    # select a submask
        N_pixel = sum(sum(im_mask))              # number of pixels in the mask
        
        im_tmp = img*im_mask                     # mask the bead and turn everything else to zero
        iB = bbox1(im_tmp)                       # calculate bounding box around the detected bead
        
        ind_temp = np.where(im_mask > 0)
        
        im_mean, im_median, im_mode = calc_stats(im_tmp[ind_temp])  # calculate mean, median, and mode of each submask
        
        # mean_store.append(sum(sum(np.asfarray(im_tmp)))/N_pixel )
        mean_store.append(im_mean)      # append list to store submask mean
        median_store.append(im_median)  # append list to store submask median
        mode_store.append(im_mode)      # append list to store submask mode
        size_store.append(N_pixel)      # append list to store submask size
        
        # py.subplot(N_plot_row + 1, 1, 1)
        # py.imshow(img)
        # py.axis('off')
        

        
        # py.close(37)
        

        # plot original image with marked boundary box and label
        # =============================================================================
        # py.figure(36)
        yt = [iB[0], iB[1], iB[1], iB[0], iB[0]]
        xt = [iB[2], iB[2], iB[3], iB[3], iB[2]]
        
        x_store.append(iB[2])    # store the top left x position of the object (with offset)
        y_store.append(iB[0])    # store the top left y position of the object (with offset)
        
        py.plot(xt,yt,'r', linewidth = 0.5)    # plot boxes
        py.text(iB[2]-2, iB[0]-2, str(m), fontsize = 6, color = 'w')  # plot label text
        
        
        # =============================================================================
    
    # a separate loop for plotting submasks. Previously, it was inside the previous loop. It needed to be decaupled so that the
    # bounding box could be drawn in a figure opened in the main wrap without needing a figure number reference. 
    for m in range(Nsub_masks):
        # =============================================================================
        # Displaying the image
        # =============================================================================       
        # N_plot_col = int(np.ceil(Nsub_masks/N_plot_row))
        # print('N_plot_col = %i, N_plot_row = %i, m = %i' %(N_plot_col, N_plot_row, m))
        
        
        im_mask = np.array(smasks_f[m].todense())                # select a submask
        im_tmp = img*im_mask                     # mask the bead and turn everything else to zero
        iB = bbox1(im_tmp)                       # calculate bounding box around the detected bead
        
        iB_store.append(iB)
        # Plot individual submasked region
        # =============================================================================
        py.figure(37)
                           
        # Plot target submasked region
        py.subplot(N_plot_row, N_plot_col,m+1)
        py.imshow(im_tmp, cmap = cmp)
        py.text(iB[2]+4, iB[0]+4, str(m), fontsize = 12, color = 'w')   # show label text
        
        # show/crop to x and y axis limits
        py.ylim([iB[0],iB[1]])  
        py.xlim([iB[2],iB[3]])
        # py.clim(bmn,bmx)
        py.axis('off')   # turn off axes text
        py.suptitle(fpath)
    
    
    # =============================================================================
    # Save figures
    # =============================================================================
    
    pp = 'out_'
    p = path_out + '/' + pp 
    
    py.figure(37)
    py.savefig(p+ 'crop.png', dpi = 300)  
    
    py.figure(36)
    py.savefig(p+ 'box.png', dpi = 300)  
    
    
    # =============================================================================        
            
    
   
    
    print_log('\n\nMean intensity of the image set = %1.2f ' %np.mean(mean_store))
    print_log('Mean intensity of the background = %1.2f \n' %b_mean)
    


    
    
    
    # =============================================================================
    
 
    
 
    
    # # =============================================================================
    # # Plot test image
    # # =============================================================================
    
    # im = cv2.imread(files_list[test_img_ind], cv2.IMREAD_GRAYSCALE)
    # im_mask = object_mask_find(im,blur_order,th_factor)
    
    # smasks, smasks_f, mask_r = mask_fragment(im_mask,obj_size_th_factor )
    
    # # plot raw image
    # # py.set_cmap('viridis')
    # py.figure()
    # py.subplot(2,2,1)
    # py.imshow(im)
    # py.title('Image')
    # py.clim([bck-100,np.max(im*mask_r)])
    # py.axis('off')
    
    # # plot image mask
    # # py.set_cmap('gray')
    # py.subplot(2,2,2)
    # py.imshow(im_mask)
    # py.title('Mask')
    # py.axis('off')
    
    # # plot refined image mask
    # py.subplot(2,2,3)
    # py.imshow(mask_r)
    # py.title('Refined mask')
    # py.axis('off')
    
    # # plot image after masking
    # py.subplot(2,2,4)
    # py.imshow(im*mask_r)
    # py.title('Image after masking')
    # py.clim([bck-100,np.max(im*mask_r)])
    # py.axis('off')
   
    # # =============================================================================

   
    print_log('===================================================================')
    
    return mean_store, median_store, mode_store, size_store, b_mean, x_store, y_store, iB_store


def single_img_analysis(fpath, path_out, block_size, th_factor):
    
    # returns the same thing as analyze_images() function (with some numpy conversion). Mostly, prints results This can perhaps be included in the analyze_images() function itself.
    
    # temp =  np.array(analyze_images(path_full,path_out + '/' + folder[m], block_size, th_factor_set[m]))*scale_factor[m]           # analyze the data set images and multiply with the scale factor
    obj_mean, obj_median, obj_mode, obj_size, b_mean, x_store, y_store, iB_store = analyze_images(fpath, path_out, block_size, th_factor)
    
    # convert list to numpy array
    obj_mean = np.array(obj_mean)
    obj_median = np.array(obj_median)
    obj_mode = np.array(obj_mode)
    obj_size = np.array(obj_size)
    x_store = np.array(x_store)
    y_store = np.array(y_store)
    
    img_mean = np.mean(obj_mean)   # mean intensity of all the detected objects (aka image mean)
    size_mean = np.mean(obj_size)  # mean size of all detected objects
   
    # convert the numbers (list of numbers) to integers 
    obj_mean = obj_mean.astype('int32') 
    obj_median = obj_median.astype('int32') 
    obj_mode = obj_mode.astype('int32') 
    obj_size = obj_size.astype('int32')
    b_mean = b_mean.astype('int32')
    img_mean = img_mean.astype('int32') 
    size_mean = size_mean.astype('int32') 
    x_store = x_store.astype('int32')
    y_store = y_store.astype('int32')
    
    
    
    
    print_log('Finsihed analysis. Back to main \n\n')
   
    
   
    
    # convert to pandas dataframe and plot a table-figure
    # x = np.linspace(0,len(obj_mean)-1, len(obj_mean))
    # x = x.astype('int32')
    
    # df = pd.DataFrame({'index': x, 'Avg. ': obj_mean})
    
    # py.figure()
    # py.axis('off')
    # py.axis('tight')
    # py.table(cellText=df.values,  colLabels = df.columns, loc='center')
    # py.title('Background (Avg.) = ' + str(b_mean))
    
    
 
    # print_log('|  Obj#  |'  '  Avg. |' )
    # print_log('|--------|-------|')
    # # list object and brightness (April 30,2024)
    # for m in range(len(obj_mean)):
    #     print_log('| ' + str(m).rjust(6) + ' |' +  str(obj_mean[m]).rjust(6) + ' |')
    
    
    # list object and brightness in sorted order (April 30,2024)
    # https://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list
    sort_index = np.argsort(obj_mean)
    
    print_log('List'.ljust(38) + ' '.ljust(10) + 'Sorted list'.ljust(38))
    print_log('_______________________________________'.ljust(18) + ' '.ljust(10) + '_______________________________________'.ljust(18))
    print_log('|  Obj#  |'  '  Mean |'  '  Area |'  '    (x,y)   |'  + ' '.ljust(10)  + '|  Obj#  |'  '  Mean |'  '  Area |' '    (x,y)   |')
    print_log('|--------|-------|-------|------------|' + ' '.ljust(10) + '|--------|-------|-------|------------|')
    # list object and brightness (April 30,2024)
    for m in range(len(obj_mean)):
        n = sort_index[m]
        print_log('| ' + str(m).rjust(6) + ' |' +  str(obj_mean[m]).rjust(6) + ' |' +  str(obj_size[m]).rjust(6) + ' |' +  '(' + str(x_store[m]).rjust(4) + ','+  str(y_store[m]).rjust(4) + ')' ' |'  
                  + ' '.ljust(10) +   
                  '| ' + str(n).rjust(6) + ' |' +  str(obj_mean[n]).rjust(6) + ' |' +  str(obj_size[n]).rjust(6) + ' |' +  '(' + str(x_store[n]).rjust(4) + ','+  str(y_store[n]).rjust(4) + ')' ' |'  )
    
    print_log('|--------|-------|-------|------------|' + ' '.ljust(10) + '|--------|-------|-------|------------|')
    
    print_log('| ' + ' Avg. '.rjust(6) + ' |' +  str(img_mean).rjust(6) + ' |' +  str(size_mean).rjust(6) + ' |'   +  ' '.rjust(11) + ' |'   
              + ' '.ljust(10) +    
              '| ' + ' Avg. '.rjust(6) + ' |' +  str(img_mean).rjust(6) + ' |'+  str(size_mean).rjust(6) + ' |'    +  ' '.rjust(11) + ' |'   )
    
    print_log('|--------|-------|-------|------------|' + ' '.ljust(10) + '|--------|-------|-------|------------|')
   
    print_log('| ' + 'Back.'.rjust(6) + ' |' +  str(b_mean).rjust(6) + ' |' + ' |'.rjust(8)  +  ' '.rjust(11) + ' |'   
              + ' '.ljust(10) +    
              '| ' + 'Back.'.rjust(6) + ' |' +  str(b_mean).rjust(6) + ' |' + ' |'.rjust(8) +  ' '.rjust(11) + ' |')
    
    print_log('|--------|-------|-------|------------|' + ' '.ljust(10) + '|--------|-------|-------|------------|')
    
    
    # move the log file to the output folder
    # https://stackoverflow.com/questions/69363867/difference-between-os-replace-and-os-rename
    # Pth('log.txt').rename(path_out +  '/log.txt')
    os.replace('log.txt',path_out +  '/log.txt')
    
    
    # Modifications: July 8,, 2024
    # Expor settings
    pp = 'out_'
    p = path_out + '/' + pp 
    
    
    b_mean_array = np.ones(np.size(obj_mean))*b_mean  # create an array of repeated background mean value (for exporting)
    
    header_str = ['Mean', 'Median', 'Mode', 'Size', 'x', 'y', 'Background', 'Contrast (mean)']  # pandas header/column names
    
    try:
        Mdata_out = np.transpose([obj_mean, obj_median, obj_mode.squeeze(), obj_size, x_store, y_store, b_mean_array, obj_mean - b_mean_array])  # numpy output data matrix
        data_frame_out = pd.DataFrame(data = Mdata_out, columns = header_str)  # convert numpy matrix to pandas dataframe
        # np.savetxt(p + 'data.csv', Mdata_out, delimiter=",")   # save numpy output matrix
        data_frame_out.to_csv(p + 'data.csv', index=True)        # save pandas output data frame
    except:
        print_log('Could not create/save data frame')
    
    # plot histograme
    py.figure()
    try:
        Avg_N_element_per_bin = 9
        py.hist(obj_mean, edgecolor = 'black', linewidth = 1.2, bins = int(len(obj_mean)/Avg_N_element_per_bin))
        py.xlabel('Fluorescence (arb. units)')
        py.ylabel('Count')
        py.savefig(p+ 'hist.png', dpi = 300)   # save image
    except:
        print_log('Error in plotting histogram.\n')
    
    
    
    
    print_log('\nFinished running.\n')
    
    return obj_mean, obj_size, b_mean, x_store, y_store, iB_store
    


 
# py.close('all')

# test = analyze_images('/home/asif/Dropbox/Codes/Python/Fluorescence data analysis/Data/100_300/*.tif')