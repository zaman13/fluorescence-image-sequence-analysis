# Fluorescence Image Analysis

<p float="left">
<a href = "https://github.com/zaman13/fluorescence-image-sequence-analysis/tree/main/Codes"> <img src="https://img.shields.io/badge/Language-Python-blue" alt="alt text"> </a>
<a href = "https://github.com/zaman13/fluorescence-image-sequence-analysis/blob/main/LICENSE"> <img src="https://img.shields.io/badge/License-MIT-green" alt="alt text"></a>
<a href = "[https://github.com/zaman13/Poisson-solver-2D/tree/master/Code](https://github.com/zaman13/fluorescence-image-sequence-analysis/tree/main/Codes)"> <img src="https://img.shields.io/badge/Version-0.8-red" alt="alt text"> </a>
</p>


<img align = "right" src="https://github.com/zaman13/fluorescence-image-sequence-analysis/blob/main/Sample%20output/13-02.1..png" alt="alt text" width="460">


A program for calculating the average brightness of beads/samples from a set of image. It also calculates histogram of the brightness distribution. 





## Features of the avarage brightness analysis code  
- Can handle a single image or a series of tiff images located in a folder
- Can work with 16bit tiff images
- Adaptive thresholding is used for masking
- Custom 16 bit adaptive thresholding algorithm

## Library requirements
- Opencv
- Matplotlib
- Numpy
  

## Sample Output



### Bead detection and fluorescence analysis:

The following figure shows detection of individual beads in a frame. 

<img src="https://github.com/zaman13/fluorescence-image-sequence-analysis/blob/main/Sample%20output/sample_out.png"  width="800">



## Acknowledgement
This work was developed at the Hesselink research lab, Stanford University. The work was partially supported by the National Institute of Health (NIH) Grant R01GM138716 and 5R21HG009758.

