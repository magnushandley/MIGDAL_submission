Submission folder for the part iii project of BGN: 8217X

Link to logbook:
https://universityofcambridgecloud-my.sharepoint.com/:o:/g/personal/mdh52_cam_ac_uk/Evcq1lRTL_9PgqMRrtjTizcBEY_Bm49Estqp2jUq4cyTAw?e=RV2qoU


This folder contains the programs used to produce the results as demonstrated in the report, as well as all simulated events generated as part of the project. 
Most code here is in python - to run, I would recommend the use of python 3.10 or newer.

There are quite a few dependencies.

Those for the ridgefinder are:
-numpy
-matplotlib
-scipy
-scikit-image
-glob
-re
-plotly
-astropy

These should also cover the dependencies for the monte carlo approach.

The file steiner_tree.py requires NetworkX

The structure of this submission folder is:

Inside no folder: 

single_elec_response_10k.npy - this is a numpy array 
containing the simulated single electron response functions.

toytracks.py: the script used for the generation of artificial tracks, which also applies gaussian 
diffusion. Returns a file with 6 columns, the first three being relative coordinates before drift,
 and the last three after. Note, the z columns (3/6) are not giving distances, but rather times 
 in nanoseconds. 

steiner_tree.py runs the NetworkX implementation of Melhorns algorithm to extract the 
Steiner tree for a set of points, implemented as nodes in the algorithm.

Inside artificial_tracks:
This folder contains all the simulated tracks generated. These are often 
organised such that one set of tracks is divided between two folders, labeled inputs 
(containing the initial ionisation/degrad points) and outputs, containing the TIFF images, 
raw ITO output files, and ITO output files after convolution with the electronic response 
functions. Typically we start from the raw ITO output files in reconstruction, as the
electronic response functions are easy to remove.

Inside monte_carlo:
Contains the files for 2D and 3D monte carlo reconstruction - these are broadly the
same files, just for different dimensionality of data.

Inside ridgefinder:
main_ridgefinder_analysis.py: Can perform all types of ridgefinder reconstruction by varying the 
parameters at the top, i.e. basic (break_degen = False), degeneracy breaking (break_degen = True), 
and with any type of deconvolution applied, or not.
y_separations_test: runs the analysis to generate figures 22 and 24
z_resolution_test: runs the analysis to generate figures 21 and 23
rf_cropping_test: tests the average speed to extract the ridgefinder spline from 10 Fe55 events with varying 
degrees of cropping, to test how the time complexity scales with number of pixels (figure 5)
RF_Functions: Contains Tom Neeps implementation of the core functions underpinning the python 
implementation of Steger's ridgefinder
RFsingletiff: Contains two functions which run the ridgefinder analysis on a TIFF image. The function
 used in main_ridgefinder_analysis.py is returnlines.