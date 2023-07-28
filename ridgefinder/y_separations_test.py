import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import interpolate
from scipy.signal import find_peaks
from skimage import io, restoration
from scipy.fft import fft2, ifft2
from RFsingletiff import *
import matplotlib as mpl
import re

im_size = 10
im_res = 1152

# TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/u_shaped_y_separations/*.tiff"))
TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/u_shaped_high_diff_outputs/*.tiff"))

params = {
   'axes.labelsize': 21,
   'font.size': 16,
   'font.family': 'sans-serif', 
   'font.serif': 'Arial', 
   'legend.fontsize': 18,
   'xtick.labelsize': 18,
   'ytick.labelsize': 18, 
   'axes.labelpad': 15,
   
   'figure.figsize': [8,6], # value in inches based on dpi of monitor
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

# markers = ["x","+","*","o"]
# mpl.rcParams['text.usetex'] = True
# plt.rcParams.update(params)

def findturningpoint(x,y):
    """
    Takes the output spline from the ridgefinder and returns the first point where
    """
    diff = x[1:]-x[:-1]
    product = np.multiply(diff[:-1],diff[1:])
    xi = np.where(product < 0)
    for i in range(len(xi[0])):
        if ((abs(x[xi[0][i]] - x[0]) > 10) and (abs(x[xi[0][i]] - x[-1]) > 10)):
            xturn = x[xi[0][i]]
            yturn = y[xi[0][i]]
            return (xturn, yturn,xi[0][i])

    return("None","None","None")


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
    

offset = 0
driftvel = 0.013

outputs = np.zeros((3,len(TIFF_files)))
#output format is predicted, reconstructed, error in reconstructed.

for i in range(len(TIFF_files)):
    rawimage = io.imread(TIFF_files[i])
    image = deconvolveimg(rawimage)
    pred_sep = float(re.search('sep_(.+?)_gem_out', TIFF_files[i]).group(1).replace('_', '.'))
    outputs[0][i] = pred_sep

    y,x = returnlines(image,
        SIGMA,
        lthresh,
        uthresh,
        minlen,
        linkthresh,
        logim,
        fromfile = False)

    x = x[::5]
    y = y[::5]

    xturn, yturn, indexturn = findturningpoint(x,y)
    print

    y_dist = y*(im_size/im_res)

    if (xturn == "None"):
        print(i,"No turn")
        outputs[1][i] = 0
        outputs[2][i] = 0

    else:
        print(i,"Turn")
        print(len(y))
        mean1 = np.mean(y_dist[:30])
        sd1 = np.std(y_dist[:30])
        mean2 = np.mean(y_dist[-30:])
        sd2 = np.std(y_dist[-30:])
        difference = abs(mean1-mean2)
        error = np.sqrt(sd1**2 + sd2**2)
        outputs[1][i] = difference
        outputs[2][i] = error

deviations = []
for i in range(len(outputs[0])):
    if (outputs[1][i] != 0):
        deviations.append(outputs[1][i] - outputs[0][i])

print("Average deviation: ",np.mean(deviations))
print("Standard deviation of offset: ",np.std(deviations))

plt.scatter(outputs[0],outputs[1],label="Reconstructed separation")
plt.errorbar(outputs[0], outputs[1], yerr=outputs[2], fmt="o")
plt.plot(outputs[0],outputs[0],color="black",label="Perfect reconstructed separation, y=x")
plt.xlabel("True separation in y [cm]")
plt.ylabel("Reconstructed separation in y [cm]")
plt.legend()
plt.title("U-shaped reconstructions, separation in the y-axis")


#plt.savefig("/Users/magnus/Documents/MIGDAL/Part III/u_shaped_y_separations_highdep.png", dpi=300)
plt.show()
