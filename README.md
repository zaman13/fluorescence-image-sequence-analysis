# Fluorescence Image-Sequence Analysis

<p float="left">
<a href = "https://github.com/zaman13/fluorescence-image-sequence-analysis/tree/main/Codes"> <img src="https://img.shields.io/badge/Language-Python-blue" alt="alt text"> </a>
<a href = "https://github.com/zaman13/fluorescence-image-sequence-analysis/blob/main/LICENSE"> <img src="https://img.shields.io/badge/License-MIT-green" alt="alt text"></a>
<a href = "[https://github.com/zaman13/Poisson-solver-2D/tree/master/Code](https://github.com/zaman13/fluorescence-image-sequence-analysis/tree/main/Codes)"> <img src="https://img.shields.io/badge/Version-1.3-red" alt="alt text"> </a>
</p>


<img align = "right" src="https://github.com/zaman13/fluorescence-image-sequence-analysis/blob/main/Sample%20output/Figure_3.png" alt="alt text" width="360">

The code can process a tiff image stack and detect bright objects of arbitrary size and shape. It can calculate the brightness/intensity of those objects in each frame and produce a intensity vs time plot. The code can be used to analyze photobleaching rate. 



# Features
- Can handle multiple objects in a frame
- Can dynamically adjust the mask every frame to account for object drift
- Average intensity of all the detected beads in the frames are calculated
  

# Sample Output
The following figure shows the average fluorescence intensity vs time plot. A time series image set containing multiple beads is used as the input data.

<img align = "left" src="https://github.com/zaman13/fluorescence-image-sequence-analysis/blob/main/Sample%20output/image_2_c1.png" alt="alt text" width="600">
  
