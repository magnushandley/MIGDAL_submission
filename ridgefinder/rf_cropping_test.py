import numpy as np
import glob
from scipy import interpolate
from scipy.signal import find_peaks
from skimage import io, restoration
from scipy.fft import fft2, ifft2
from RFsingletiff import *
import time
from scipy import stats
import re
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
Crops Fe55 events by varying amounts, to measure the time complexity of the ridgefinder algorithm.
"""

params = {
   'axes.labelsize': 21,
   'font.size': 22,
   'font.family': 'sans-serif',
   'font.serif': 'Arial',
   'legend.fontsize': 18,
   'xtick.labelsize': 18,
   'ytick.labelsize': 18,
   'axes.labelpad': 15,
   
   'figure.figsize': [10,8], # value in inches based on dpi of monitor
   'figure.dpi': 105.5, # My monitor has a dpi of around 105.5 px/inch
   
   'axes.grid': True,
   'grid.linestyle': '-',
   'grid.alpha': 0.25,
   'axes.linewidth': 1,
   'figure.constrained_layout.use': True,
   
   
   # Using Paul Tol's notes:
   'axes.prop_cycle':
      mpl.cycler(color=['#4477aa', # blue
                        '#ee6677', # red/pink
                        '#228833', # green
                        '#aa3377', # purple
                        '#66ccee', # cyan
                        '#ccbb44', # yellow
                        '#bbbbbb', # grey
                        ]),
     
      # Pick either the cycler above, or the cycler below:
       
      # (mpl.cycler(color=['#4477aa', # blue
      #                     '#ee6677', # red/pink
      #                     '#228833', # green
      #                     '#aa3377', # purple
      #                     '#66ccee', # cyan
      #                     '#ccbb44', # yellow
      #                     '#bbbbbb', # grey
      #                     ]) +
      #   mpl.cycler(linestyle=['-', # solid
      #                         '--', # dashed
      #                         ':', # dotted
      #                         '-.', # dash dot
      #                         (0, (3, 1, 1, 1, 1, 1)), # narrow dash dot dot
      #                         (0, (1, 2, 5, 2, 5, 2)), # dash dash dot
      #                         (0, (5, 2.5, 1, 2.5, 1, 2.5)), # dash dot dot
      #                         ])),
     
   'lines.linewidth': 2.5,
   
   'image.cmap': 'jet',
}

mpl.rcParams['text.usetex'] = True
plt.rcParams.update(params)


TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/stripreader/simulatedtracks/photoelectrons_4xi22/*.tiff"))

def deconvolveimg(im, iterations = 10, lr_sigma = 0.042*(1152/8.1), hole_sigma = 120):
    def fft(im):
        return np.fft.fftshift(np.fft.fft2(im))

    def ifft(im):
        return np.abs(np.fft.ifft2(np.fft.ifftshift(im)))

    def filter_holes(im, sigma):
        fft_im = fft(im)
        masked_im = gauss_kernel(len(im), sigma) * fft_im
        return ifft(masked_im)
        
    def gauss_kernel(size, sigma):
        """
        Quantitative approximation of a 2D Gaussian kernel
        """
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        
        return g/g.sum()
        
    filtered = filter_holes(im, hole_sigma)
    
    PSF = gauss_kernel(25,lr_sigma) # 2D quantised PSF

    deconv_im = restoration.richardson_lucy(filtered, PSF, iterations, clip=False)*65535
    
    return(deconv_im)

SIGMA = 2.5 #sigma for derivative determination ~> Related to track width
lthresh = 0.02 #tracks with a response lower than this are rejected (0 accepts all)
uthresh = 0 #tracks with a response higher than this are rejected (0 accepts all)
minlen = 6 #minimum track length accepted
linkthresh = 50 #maximum distance to be linked
logim = False

outputs = np.zeros((3,len(TIFF_files)))
#output format is predicted, reconstructed, error in reconstructed.

img_midpoint = 576

deltax_arr = 50*np.arange(1,10)

num_images = 10
resolutions = np.zeros(len(deltax_arr))
avg_time = np.zeros(len(deltax_arr))
avg_time_sd = np.zeros(len(deltax_arr))


j = 0
for dx in deltax_arr:
    print("j = ",j)
    resolutions[j] = 2*dx
    print("Resolution = ",resolutions[j])
    totaltime = 0
    times = np.zeros(num_images)
    for i in range(num_images):
        print("i = ",i)
        tic2 = time.time()
        rawimage = io.imread(TIFF_files[i])
        image = deconvolveimg(rawimage)
        
        tic = time.time()
        
        y,x = returnlines(image[img_midpoint-dx:img_midpoint+dx,img_midpoint-dx:img_midpoint+dx],
            SIGMA,
            lthresh,
            uthresh,
            minlen,
            linkthresh,
            logim,
            fromfile = False)
            
        toc = time.time()
        toc2 = time.time()
        times[i] = toc - tic
    avg_time[j] = np.mean(times)
    print("avg_time = ",avg_time[j])
    avg_time_sd[j] = np.std(times)
    print("avg_time_sd = ",avg_time_sd[j])
    j += 1

to_file = np.zeros((3,len(resolutions)))
to_file[0][:] = resolutions
to_file[1][:] = avg_time
to_file[2][:] = avg_time_sd

np.savetxt("cropped_test.csv",to_file)

reg = stats.linregress(resolutions**2,avg_time)
gradient = reg.slope
intercept = reg.intercept
line = (resolutions**2)*gradient + intercept
print(gradient,intercept)
plt.scatter(resolutions**2, avg_time, marker="+")
labelstr = "Linear fit, y = "+str("%.3f" % gradient)+"x + "+str("%.3f" % intercept)
plt.plot(resolutions**2,line,color="Black",label = labelstr)
plt.xlabel("Total number of pixels")
plt.ylabel("Time taken by ridgefinder [s]")
plt.title("Time complexity of ridgefinder with resolution")
plt.savefig("/Users/magnus/Documents/MIGDAL/Part III/cropping_test.png", dpi=300)
plt.show()
