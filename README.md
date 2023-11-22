# Fluorescence Image-Sequence Analysis

<p float="left">
<a href = "https://github.com/zaman13/fluorescence-image-sequence-analysis/tree/main/Codes"> <img src="https://img.shields.io/badge/Language-Python-blue" alt="alt text"> </a>
<a href = "https://github.com/zaman13/fluorescence-image-sequence-analysis/blob/main/LICENSE"> <img src="https://img.shields.io/badge/License-MIT-green" alt="alt text"></a>
<a href = "[https://github.com/zaman13/Poisson-solver-2D/tree/master/Code](https://github.com/zaman13/fluorescence-image-sequence-analysis/tree/main/Codes)"> <img src="https://img.shields.io/badge/Version-0.8-red" alt="alt text"> </a>
</p>


<img align = "right" src="https://github.com/zaman13/fluorescence-image-sequence-analysis/blob/main/Sample%20output/13-02.1..png" alt="alt text" width="460">


Two separate applications are included in this repository. The first one is analysis of photobleaching from a timeseries image set. The second one is calculating the average brightness of beads/samples from a set of image.  

The photobleaching code can process a tiff image stack and detect bright objects of arbitrary size and shape. It can calculate the brightness/intensity of those objects in each frame and produce a intensity vs time plot. The code can be used to analyze photobleaching rate. 



# Features of the photobleach analysis code
- Can handle multiple objects in a frame
- Can handle irregularly shaped objects
- Can dynamically adjust the mask every frame to account for object drift
- Average birghtness of all the detected beads in the frames are calculated

# Features of the avarage brightness analysis code  
- Can handle a series of tiff images located in a folder
- Can work with 16bit tiff images
  

# Sample Output



## Bead detection and fluorescence analysis:

The following figure shows detection of individual beads in a frame. 

<img src="https://github.com/zaman13/fluorescence-image-sequence-analysis/blob/main/Sample%20output/sample_out.png"  width="800">




## Photobleaching:

The following figure shows the average fluorescence intensity vs time plot. A time series image set containing multiple beads is used as the input data. 

  <img src="https://github.com/zaman13/fluorescence-image-sequence-analysis/blob/main/Sample%20output/image_2_c1.png" alt="alt text" width="600">


## Acknowledgement
This work was developed at the Hesselink research lab, Stanford University. The work was partially supported by the National Institute of Health (NIH) Grant R01GM138716 and 5R21HG009758.

