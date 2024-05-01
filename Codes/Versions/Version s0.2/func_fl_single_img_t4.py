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
    
"""

# Load libraries
import numpy as np
import matplotlib.pyplot as py
import cv2
import pandas as pd 


from PIL import Image as PIL_Image

from skimage import measure
from scipy.stats import kurtosis, skew

import glob

import random



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
 
    
    
    
    sub_masks = np.zeros([np.shape(img)[0],np.shape(img)[1],N_obj])  # array of image arrays. A mask for each detected object.
    obj_size_sum = np.zeros(N_obj)     # initialization. The array will store size of each submask.
    
    for m in range(N_obj):             # loop through each object
        uu = uu*0                      # reset uu array for reuse
        uu[idx[m]] = 1                 # populate uu   
        obj_size_sum[m] = np.sum(uu)   # size of the object. Defined by the number of ones.
        sub_masks[:,:,m] = uu          # store the submask
        
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
    
  
    
    sub_masks_refined = np.zeros([np.shape(img)[0],np.shape(img)[1],N_obj_refined])  # initialize array of image arrays where submasks will be stored
    mask_refined = np.zeros(np.shape(img))   # initialize composite mask from refined sub-masks 
    # py.figure()
    
    counter = 0
    for m in range(N_obj):   # loop over all objects
        
        if obj_size_sum[m] > obj_size_sum_th:   # for objects that meet threshold size
            sub_masks_refined[:,:,counter] = sub_masks[:,:,m]             # store refined sub-mask
            mask_refined = mask_refined + sub_masks_refined[:,:,counter]  # superimpose on the refined mask
             
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
    
    print_log('\n\n =============================================================================')
    
  
    
    
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
       
    mean_store = []
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
        return 0
    
    smasks, smasks_f, mask_r = mask_fragment(im_mask,obj_size_th_factor )    # mask fragment and composite reforming
    
    # smasks_f, mask_r = skew_refine(smasks_f)
                 
    # print_log(np.shape(smasks_f[:,:,0]))
    print_log('Submasking done')
    Nsub_masks = np.shape(smasks_f)[2]
    
    N_plot_col = int(np.ceil(np.sqrt(Nsub_masks))) 
    N_plot_row = int(np.ceil(Nsub_masks/N_plot_col))

    
    py.figure(36)
    py.clf()
    py.imshow(img, cmap = cmp)
    py.clim(bmn,bmx)
    # py.clim(700,2000)
    py.axis('off')
    py.suptitle(fpath + '\n' + 'Image (B) min, max = ' + str(bmn) + ', ' +  str(bmx))   # print sourcce file and image min-max in the title
    
    py.figure(37)
    py.clf()
    # fg, ax = py.subplot(N_plot_row, N_plot_col,m+1)
    # fg, ax = py.subplots(N_plot_row, N_plot_col)
    # Loop over the submaks (each submask represents one detected bead in the frame)
    print_log('Looping over submasks...')
    
    
    
    for m in range(Nsub_masks):
        
        im_mask = smasks_f[:,:,m]                # select a submask
        N_pixel = sum(sum(im_mask))              # number of pixels in the mask
        im_tmp = img*im_mask                     # mask the bead and turn everything else to zero
        iB = bbox1(im_tmp)                       # calculate bounding box around the detected bead
        
        ind_temp = np.where(im_mask > 0)
        
        im_mean, im_mediam, im_mode = calc_stats(im_tmp[ind_temp])
        
        # mean_store.append(sum(sum(np.asfarray(im_tmp)))/N_pixel )
        mean_store.append(im_mean)
        
        
        # py.subplot(N_plot_row + 1, 1, 1)
        # py.imshow(img)
        # py.axis('off')
        
        # =============================================================================
        # Displaying the image
        # =============================================================================       
        # N_plot_col = int(np.ceil(Nsub_masks/N_plot_row))
        # print('N_plot_col = %i, N_plot_row = %i, m = %i' %(N_plot_col, N_plot_row, m))
        
        
        
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
        

        # plot original image with marked boundary box and label
        # =============================================================================
        py.figure(36)
        yt = [iB[0], iB[1], iB[1], iB[0], iB[0]]
        xt = [iB[2], iB[2], iB[3], iB[3], iB[2]]
        
        py.plot(xt,yt,'r', linewidth = 0.5)    # plot boxes
        py.text(iB[2]-2, iB[0]-2, str(m), fontsize = 6, color = 'w')  # plot label text
        
        
        # =============================================================================
    
    
    
    
    # =============================================================================
    # Save figures
    # =============================================================================
    
    pp = 'out_'
    p = path_out + '/' + pp 
    py.savefig(p+ '.jpg', dpi = 300)  
    py.figure(37)
    py.savefig(p+ '.png')  
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

   
    print_log('=============================================================================')
    
    return mean_store, b_mean


def single_img_analysis(fpath, path_out, block_size, th_factor):
    # temp =  np.array(analyze_images(path_full,path_out + '/' + folder[m], block_size, th_factor_set[m]))*scale_factor[m]           # analyze the data set images and multiply with the scale factor
    temp, b_mean = analyze_images(fpath, path_out, block_size, th_factor)
    
    temp = np.array(temp)
    b_mean = b_mean.astype('int32')
    
    print_log('Finsihed analysis. Back to main \n\n.')
    py.figure()
    py.axis('off')
    py.axis('tight')
    # py.xticks(np.linspace(1,len(temp)))
    
    
    # py.bar(np.linspace(0,len(temp)-1, len(temp)), temp)
    
    x = np.linspace(0,len(temp)-1, len(temp))
    x = x.astype('int32')
    temp = temp.astype('int32')
    # print(b_mean)
    # print(len(x))
    # print(len(temp))
    # print(x)
    df = pd.DataFrame({'index': x, 'Avg. ': temp})
    py.table(cellText=df.values,  colLabels = df.columns, loc='center')
    
    py.title('Background (Avg.) = ' + str(b_mean))
    
    
    # list object and brightness (April 30,2024)
    for m in range(len(temp)):
        print_log('Object ' + str(m) + ' = %i' %temp[m])
    
    
    
    # https://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list
    # list object and brightness in sorted order (April 30,2024)
    sort_index = np.argsort(temp)
    
    print_log('\n Sorted list \n')
    for m in sort_index:
        print_log('Object ' + str(m) + ' = %i' %temp[m])
    
    
    return temp, b_mean
    

# py.close('all')

# test = analyze_images('/home/asif/Dropbox/Codes/Python/Fluorescence data analysis/Data/100_300/*.tif')