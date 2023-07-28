import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from scipy import interpolate
from skimage import io
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import time


TIFF_files = sorted(glob.glob("/Users/magnus/Downloads/TIFF_250_keV_NR_10_keV_ER 2/*_gem_out.tiff"))


def deconvolveimg(im, iterations = 10, lr_sigma = 0.042*(1152/8.1), hole_sigma = 120):
    """
    Removal of gaussian hole pattern
    Parameters
    --------
    im: Numpy array containing the image
    iterations: Number of Lucy-Richardson iterations to perform
    lr_sigma: sd for gaussian kernal used in lr_deconvolution
    hole_sigma: sd in fourier space for the gaussian windowing function used to remove the hole pattern.
    Returns
    --------
    deconv_im: The filtered and deconvolved image
    """
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

    #deconv_im = restoration.richardson_lucy(filtered, PSF, iterations, clip=False)*65535
    
    return(filtered)

def generate_points(outer,inner, N):
    """
    Generates approximately N points randomly scattered within a radius 'radius', centered on 0,0. We
    want this to be circularly symmetric to avoid selction biases towards the corners.
    Try to be as efficient as possible - means avoiding trig fns where possible.
    Parameters
    -----------
    radius: float, radius (in pixels) within which the points should be generated
    N: int, number of sample points
    Returns:
    ------------
    An Nx2 numpy array of points randomly sampled in the x, y plane
    """
    num_points = int((N*4*outer**2)/(np.pi*outer**2 + np.pi*inner**2))
    raw_output = 2*(np.random.random((num_points,2))-0.5)*outer
    sq_magnitudes = np.multiply(raw_output[:,0],raw_output[:,0]) + np.multiply(raw_output[:,1],raw_output[:,1])

    output = []

    for i in range(num_points):
        if ((sq_magnitudes[i] < outer*outer) and (sq_magnitudes[i] > inner*inner)):
            output.append(raw_output[i])

    return(output)

class track:
    def __init__(self, image):
        """
        Initialise an instance of the track class with the image
        """
        self.img = image

    def find_start(self, threshold):
        """
        Identifies the start of the track - in this case the leftmost point above threshold*max_brightness
        """
        threshold = threshold*np.max(self.img)
        thresh_img = self.img
        thresh_img[thresh_img < threshold] = 0
        positions = np.nonzero(thresh_img)
        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        print("left:",left)
        right = positions[1].max()
        x = left
        y = positions[0][np.where(positions[1] == left)[0][0]]
        print(y," ",x)
        self.start = [[x,y]]
        # print(self.start)
        self.current_point = self.start
        self.points = self.current_point
        return(x,y)

    def start_track(self,outer,inner,N):
        """
        First scatter search - no weighting needed for this, just looking for the best
        point of the N investigated between radius 'inner' and 'outer'
        Parameters
        --------
        outer: Outer radius of search, in cm
        inner: Inner radius of search, in cm
        N: Number of points between these radii investigated
        """
        raw_points = generate_points(outer,inner,N)
        coords = np.add(raw_points,self.start)
        # ax[1].scatter(coords[:,0],coords[:,1],marker="+")
        magnitudes = np.zeros(len(coords))

        for i in range(len(coords)):
            magnitudes[i] = self.img[int(coords[i][1]),int(coords[i][0])]

        next_best = coords[np.where(magnitudes == np.max(magnitudes))[0][0]]
        self.current_point = [[next_best[0],next_best[1]]]
        self.points = np.append(self.points,self.current_point,axis=0)

    def propagate_track(self,outer,inner,N,a):
        """
        Main iterative step in the reconstruction - scatters N points between radius `inner' and `outer', in pixels, then determines the best next point using the weighting function, which includes the weighting parameter a.
        """        offset = np.subtract(self.points[-1],self.points[-2])
        raw_points = generate_points(outer,inner,N)

        for i in range(len(raw_points)):
            if (np.dot(raw_points[i],offset) < 0):
                raw_points[i] = -raw_points[i]

        coords = np.add(raw_points,self.current_point)
        magnitudes = np.zeros(len(coords))

        # print(offset)
        # print(self.points)
        for i in range(len(coords)):
            #potentially slow, come back to:

            weighting = 0
            for j in range(len(self.points)):
                rvec = self.points[j]-coords[i]
                rsquare = rvec.dot(rvec)
                weighting += 1/rsquare

            print(weighting)
            magnitudes[i] = self.img[int(coords[i][1]),int(coords[i][0])]/(a+weighting)

        next_best = coords[np.where(magnitudes == np.max(magnitudes))[0][0]]
        self.current_point = [[next_best[0],next_best[1]]]
        self.points = np.append(self.points,self.current_point,axis=0)
event = 0


raw_image = io.imread(TIFF_files[event])
print(TIFF_files[event])
# log_image = np.log(raw_image+(1e-10))
filtered_img = deconvolveimg(raw_image)


# H_elems = hessian_matrix(filtered_img, sigma=2.5, order='rc')
# e_vals = hessian_matrix_eigvals(H_elems)[1]
fig, ax = plt.subplots()

# ax[0].imshow(raw_image)
im = ax.imshow(filtered_img)

track1 = track(filtered_img)
# track1.start = [[529,557]]

tic = time.time()
track1.find_start(1/10)
track1.points = track1.start
track1.current_point = track1.start

track1.start_track(7,4,30)
for i in range(20):
    track1.propagate_track(7,4,30,1)

toc = time.time()

print("Time:",toc-tic)


fig.colorbar(im, ax=ax,label="Image intensity, arbitrary units")
ax.plot(track1.start[0][0],track1.start[0][1],marker="+",markersize=10,color="red")
ax.set_ylabel("Camera x-coordinate (pixels)")
ax.set_xlabel("Camera y-coordinate (pixels)")
ax.plot(track1.points[:,0],track1.points[:,1],color="orange")


plt.show()
