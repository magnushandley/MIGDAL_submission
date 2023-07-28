import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from scipy import interpolate
from skimage import io, restoration
from skimage import io
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import time
import re
import math

case = 'migdal'
event = 9
interpolated = False
lrdeconvolvedito = True
projection = "3d"

global driftvel,pixelwidth,centerx
driftvel = 0.013
pixelwidth=10/1152
centerx=576

prfpath = "/Users/magnus/Documents/MIGDAL/stripreader/stripdeconvolution/single_elec_response_10k.npy"


if case == 'migdal':
    TIFF_files = sorted(glob.glob("/Users/magnus/Downloads/TIFF_250_keV_NR_10_keV_ER 2/*_gem_out.tiff"))
    raw_ITO_files = sorted(glob.glob("/Users/magnus/Downloads/250_keV_NR_10_keV_ER_ITO/raw/*"))
    convl_ITO_files = sorted(glob.glob("/Users/magnus/Downloads/250_keV_NR_10_keV_ER_ITO/conv/*"))
    degrad_files = sorted(glob.glob("/Users/magnus/Downloads/250_keV_NR_10_keV_ER(1)/*"))
    iterations = 12
    binning = 10

elif case == 'fe55':
    TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/stripreader/simulatedtracks/photoelectrons_4xi22/*.tiff"))
    raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/stripreader/simulatedtracks/photoelectrons_4xi22/*ITO_raw.txt"))
    convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/stripreader/simulatedtracks/photoelectrons_4xi22/*ITO_conv_ml.txt"))
    degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/stripreader/simulatedtracks/photoelectron_5.25keV-inputs/*"))
    iterations = 5
    binning = 1

elif case == 'artificial1':
    convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6kev_uniform_ITO_TIFF/*ITO_conv_ml.txt"))
    raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6kev_uniform_ITO_TIFF/*ITO_raw.txt"))
    TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6kev_uniform_ITO_TIFF/*.tiff"))
    degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6keV_uniform_inputs/*"))
    iterations = 5
    binning = 1

elif case == 'artificial1_new':
    convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6keV_practicalrange_line_outputs/*ITO_conv_ml.txt"))
    raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6keV_practicalrange_line_outputs/*ITO_raw.txt"))
    TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6keV_practicalrange_line_outputs/*.tiff"))
    degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/6keV_practicialrange_line_inputs/*"))
    binning = 1
    iterations = 10

elif case == 'artificial2':
    convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/250kev_uniform_ITO_tiff/*ITO_conv_ml.txt"))
    raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/250kev_uniform_ITO_tiff/*ITO_raw.txt"))
    TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/250kev_uniform_ITO_tiff/*.tiff"))
    degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/250keV_uniform_inputs/*"))
    iterations = 10
    binning = 20

elif case == 'artificial3':
    convl_ITO_files = ["/Users/magnus/Documents/MIGDAL/artificial_tracks/offset_v/outputs/offset_v/v_shaped_inclined_third_kink_gem_out_ITO_conv_ml.txt"]
    raw_ITO_files = ["/Users/magnus/Documents/MIGDAL/artificial_tracks/offset_v/outputs/offset_v/v_shaped_inclined_third_kink_gem_out_ITO_raw.txt"]
    TIFF_files = ["/Users/magnus/Documents/MIGDAL/artificial_tracks/offset_v/outputs/offset_v/v_shaped_inclined_third_kink_gem_out.tiff"]
    degrad_files = ["/Users/magnus/Documents/MIGDAL/artificial_tracks/offset_v/inputs/v_shaped_inclined_third_kink.txt"]
    binning = 1
    iterations = 10

elif case == 'artificial_inclined':
    convl_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/inclinations_v_shaped/outputs/*ITO_conv_ml.txt"))
    raw_ITO_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/inclinations_v_shaped/outputs/*ITO_raw.txt"))
    TIFF_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/inclinations_v_shaped/outputs/*.tiff"))
    degrad_files = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/inclinations_v_shaped/inputs/*"))
    binning = 1
    iterations = 10

def extract_theta_phi(filename):
    """
    Takes the filenames of our artificial files to extract the polar coordinates from the filenames
    """
    theta = re.search('tht(.*)phi', filename)
    theta = theta.group(1)
    theta = float(theta.replace("_","."))
    phi = re.search('phi(.*).csv', filename)
    phi = phi.group(1)
    phi = float(phi.replace("_","."))
    return(theta,phi)

def lrdeconvolve(ITO,prfpath,iterations=10,ratio=1/1000):
    """
    Lucy-Richardson deconvolution of the ITO responses. iterations is set to 8 after the optimisation performed in section 3 of the report.
    parameters
    -------
    ITO: Numpy array of the ITO waveforms
    prfpath: path to the numpy file contating the simulated single electron point response functions.
    Iterations: The number of Lucy-Richardson iterations to perform.
    ratio: The ratio of the maximum peak height used to set the clipping threshold. Insensitive to this parameter, all that matters is that this value is << 1 but non-zero.
    Returns
    ----------
    out: Numpy array with deconvolved waveforms
    """
    epsilon = np.max(ITO)*ratio
    prf = np.load(prfpath)
    norm = 1/np.sum(prf)
    prf = prf*norm

    #reindex and reduce size
    psf = np.zeros((5,40))
    print(np.shape(prf))
    psf[2:,20:] = prf[:3,5:25]
    psf[:2,20:] = prf[28:,5:25]

    return(restoration.richardson_lucy(ITO, psf, num_iter=iterations, clip=False, filter_epsilon=epsilon))

def rmsoffset(raw, reconstructed):
    """
    Parameters
    ----------
    raw: 3d numpy array of x,y,z, containing the degrad primary ionisation
    reconstructed: 3d numpy array of x,y,z containing the reconstructed points
    Returns
    ----------
    rms: root mean square deviation of a degrad point from a point in the reconstructed line.
    """
    rawpoints = len(raw[0])
    ms = np.zeros(rawpoints)

    raw = np.transpose(raw)
    reconstructed = np.transpose(reconstructed)
    #transposition is neccesary for the vectorisation to work well

    for i in range(rawpoints):
        diff = reconstructed - raw[i]
        sdiff = (diff*diff).sum(axis=1)
        ms[i] = np.min(sdiff)
        
    rms = np.sqrt(np.mean(ms))
    return(rms)

def scanoffsets(x, y, deltaz, track, min, max, N = 100):
    """
    Scans through all possible z offsets, looking for the optimum solution
    Parameters
    ----------
    x, y, deltaz: arrays of the coordinates of the reconstructed data
    track: 6xN array of the degrad simulated points,
    min: minimum offset in z to scan
    max: maximum offset in z to scan
    N: number of points within the interval to investigate
    """
    rmsvec = np.zeros(N)
    offsetvec = np.zeros(N)
    
    for n in range(N):
        offset = min + (n*(max-min)/N)
        z = deltaz + offset
        offsetvec[n] = offset
        rmsvec[n] = rmsoffset(np.array((track[3],track[4],track[5]*driftvel)),np.array((x,y,z)))

    index = np.where(rmsvec == np.min(rmsvec))[0][0]

    return(offsetvec[index])

def deconvolveimg(im, iterations = 10, hole_sigma = 120):
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
    num_points = int((N*8*outer**3)/((4/3)*np.pi*(outer**3 - inner**3)))
    raw_output = 2*(np.random.random((num_points,3))-0.5)*outer
    sq_magnitudes = np.multiply(raw_output[:,0],raw_output[:,0]) + np.multiply(raw_output[:,1],raw_output[:,1]) + np.multiply(raw_output[:,2],raw_output[:,2])

    output = []

    for i in range(num_points):
        if ((sq_magnitudes[i] < outer*outer) and (sq_magnitudes[i] > inner*inner)):
            output.append(raw_output[i])

    return(output)

def pixeltoxy(pixelx,pixely, pixelwidth=10/1152,centerx=576, centery=576):
    """
    Takes pixel indicies and returns x and y coordinates
    """
    return((pixelx-centerx)*pixelwidth,(pixely-centery)*pixelwidth)

def coordtopixeltime(cx,cy,cz,pixelwidth=10/1152,driftvel=0.013,centerx=576, centery=576):
    """
    Takes coordinates and returns the x,y pixels and time index
    """
    px,py = (cx/pixelwidth)+centerx, (cy/pixelwidth)+centery
    tz = cz/driftvel
    return(px,py,tz)

def timetodepth(time, driftvel=0.013):
    """
    Converts time to relative depth
    """
    return(time*driftvel)

def pixeltostrip(pixel, offset,centerx=576,middle=15):
    """
    Takes the x coordinates of all points along the ridge and returns the coordinates in strips
    """
    pixel = pixel-centerx
    strip = np.add(((10/1152)*(120/10)*pixel), (middle + offset))
    return(strip)



class track:
    def __init__(self, image, ITOdata):
        """
        Initialise an instance of the track class with the image and ITO waveforms
        """
        self.img = image
        self.ito = ITOdata

    def normalise_coords(self):
        """
        Normalise both coordinate systems such that they sum to 1
        """
        self.img = self.img/np.sum(self.img)
        self.ito = self.ito/np.sum(self.ito)

    def find_start(self,threshold):
        """
        Identifies the start of the track - as is this will return just right of center, on strip 15 of the simulated events. Uncommenting the large commented out block will change this to the rightmost point of the 2D image brighter than threshold*(maximum image brightness)
        """
#
        startpix_x = 577
        x, y = 0.01,0
#        threshold = threshold*np.max(self.img)
#        max = np.max(self.img)
#
#        thresh_img = self.img
#        thresh_img[thresh_img < threshold] = 0
#
#        positions = np.nonzero(thresh_img)
#        top = positions[0].min()
#        bottom = positions[0].max()
#        left = positions[1].min()
#        # print("left:",left)
#        right = positions[1].max()
#        x = right
#        y = positions[0][np.where(positions[1] == x)[0][0]]
#        maxstart = np.where(self.img == max)
#        print("maxstart:",maxstart)
#
#        x = maxstart[0][0]
#        y = maxstart[1][0]
#        x, y = pixeltoxy(x,y)
  

        

        startstrip = int(pixeltostrip(startpix_x,0))

        print(startstrip)
        starttime = np.where(self.ito[startstrip] == np.max(self.ito[startstrip]))[0][0]
        z = timetodepth(starttime)

        self.start = [[x,y,z]]

        self.current_point = self.start
        self.points = self.current_point
        return(x,y,z)

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
        x_ar = np.arange(30)
        y_ar = np.arange(350)
        phi = -data

        #Provision below for artificial resolution enhancement through linear interpolation, 
        #leave interpolated = False for normal reconstruction.
        if interpolated == True:
            f = interpolate.interp2d(x_ar, y_ar, phi)

        raw_points = generate_points(outer,inner,N)
        coords = np.add(raw_points,self.start)
        magnitudes = np.zeros(len(coords))
        bragg_sum = 0

        for i in range(len(coords)):
            px,py,tz = coordtopixeltime(coords[i][0],coords[i][1],coords[i][2])
            if interpolated == False:
                mag = self.img[int(px),int(py)]*self.ito[int(pixeltostrip(px,0))][int(tz)]
            elif interpolated == True:
                mag = self.img[int(px),int(py)]*f(pixeltostrip(px,-0.5),tz)
            magnitudes[i] = mag
            bragg_sum += mag

        next_best = coords[np.where(magnitudes == np.max(magnitudes))[0][0]]
        self.current_point = [[next_best[0],next_best[1],next_best[2]]]
        self.points = np.append(self.points,self.current_point,axis=0)
        return(np.sqrt(bragg_sum/len(coords)))

    def propagate_track(self,outer,inner,N,a):
        """
        Main iterative step in the reconstruction - scatters N points between radius `inner' and `outer', then determines the best next point using the weighting function, which includes the weighting parameter a.
        """

        x_ar = np.arange(30)
        y_ar = np.arange(350)
        phi = -data
        
        #below is a legacy of a test of smoothing the ITO data - not included in the report but had some interesting implications. I chose not to pursue this as it was potentially misleading.
        #could move out of this function, which reduces the number of calls to generate the interpolated spline
        if interpolated == True:
            f = interpolate.interp2d(x_ar, y_ar, phi)


        offset = np.subtract(self.points[-1],self.points[-2])
        raw_points = generate_points(outer,inner,N)

        coords = np.add(raw_points,self.current_point)
        magnitudes = np.zeros(len(coords))
        bragg_sum = 0

        for i in range(len(coords)):
            px,py,tz = coordtopixeltime(coords[i][0],coords[i][1],coords[i][2])
            
            #calculate sum(1/r^2)
            weighting = 0
            for j in range(len(self.points)):
                rvec = self.points[j]-coords[i]
                rsquare = rvec.dot(rvec)
                weighting += 1/rsquare

            #The coordinates beign interpolated or not changes the coordinates of a strip by 1/2 a strip
            if interpolated == False:
                mag = self.img[int(px),int(py)]*self.ito[int(pixeltostrip(px,0))][int(tz)]
            elif interpolated == True:
                mag = self.img[int(px),int(py)]*f(pixeltostrip(px,-0.5),tz)

            bragg_sum += mag

            magnitudes[i] = mag/(a+weighting)

        next_best = coords[np.where(magnitudes == np.max(magnitudes))[0][0]]
        self.current_point = [[next_best[0],next_best[1],next_best[2]]]
        self.points = np.append(self.points,self.current_point,axis=0)
        return(np.sqrt(bragg_sum/len(coords)))



def reconstruct(event,TIFF_files=TIFF_files,raw_ITO_files=raw_ITO_files,degrad_files=degrad_files,iterations=iterations,binning=binning):
    """
    Main 3D reconstruction, also including provision for bragg curve determination, and projection in various axes
    Parameters
    ----------
    TIFF_files: sorted list of TIFF filenames
    raw_ITO_files: sorted list of ITO waveform files
    degrad_files: sorted list of files contataining simulated ionisation electrons
    interations: number of steps the monte-carlo reconstruction should make
    binning: Every `binning'th simulated point is displayed in 3D - this does not affect the reconstruction, is just a visualisation parameter
    """
    tic = time.time()
    
    #read ITO
    with open(raw_ITO_files[event]) as f:
        lines = f.readlines()
        
    global data
    data = np.zeros((350,30))

    i = 0

    for line in lines:
        split = line.split()
        data[i] = split[1:]
        i += 1
            

    datat = -np.transpose(data)
    if lrdeconvolvedito == True:
        datat = lrdeconvolve(datat,prfpath)
        data = -np.transpose(datat)

    degrad = np.transpose(np.loadtxt(degrad_files[event]))


    raw_image = io.imread(TIFF_files[event])
    filtered_img = deconvolveimg(raw_image)

    track1 = track(filtered_img, datat)
    track1.normalise_coords()
    
    #Start the track
    track1.find_start(1/20)
    track1.points = track1.start

    #Setup for bragg curve determination
    bragg_curve = []
    bragg_curve.append(track1.start_track(7*(10/1152),3*(10/1152),200))
    
    #Propagate the reconstruction forward
    for i in range(iterations):
        bragg_curve.append(track1.propagate_track(7*(10/1152),3*(10/1152),200,100))

    toc = time.time()

    # print("Time:",toc-tic)

    pointst = np.transpose(track1.points)

    fig = plt.figure()

    if projection == "3d":
        ax2 = plt.axes(projection='3d')
        scatter_plot = ax2.scatter3D(degrad[3][::binning],degrad[4][::binning],degrad[5][::binning]*0.013)
        scatter_plot = ax2.scatter3D(pointst[0],pointst[1],pointst[2]-0.3)
        ax2.plot3D(pointst[0],pointst[1],pointst[2]-0.3, color="black")
        ax2.set_xlabel("x [cm]")
        ax2.set_ylabel("y [cm]")
        ax2.set_zlabel("z [cm]")

    if projection == "xz":
        ax2 = plt.axes()
        scatter_plot = ax2.scatter(pointst[0],pointst[2]-0.47) 
        scatter_plot = ax2.scatter(degrad[3],degrad[5]*0.013)
        ax2.plot(pointst[0],pointst[2]-0.47, color="black")
        ax2.set_xlabel("x [cm]")
        ax2.set_ylabel("z [cm]")

    # offset = scanoffsets(pointst[0],pointst[1],pointst[2],degrad,-0.7,-0.2)
    rms = rmsoffset(np.array((degrad[3],degrad[4],degrad[5]*driftvel)),np.array((pointst[0],pointst[1],pointst[2]-0.47)))
    sd = np.sqrt((np.std(degrad[3])**2)+(np.std(degrad[4])**2)+(np.std(degrad[5]*0.013)**2))
    # print("rms:",rms,"sd:",sd,"ratio:",rms/sd)

    
    

    

    # lim = 0.6
    # ax2.set_xlim(-lim,lim)
    # ax2.set_ylim(-lim,lim)
    # ax2.set_zlim(-0.1,lim)

    fig, ax = plt.subplots()

    tpxs = (track1.points[:,0]/pixelwidth)+centerx
    tstrs = pixeltostrip(tpxs,0)
    ttime = (track1.points[:,2])/driftvel

    ax.pcolormesh(-data)
    ax.plot(tstrs,ttime)

    ax.set_xlabel("strip")
    ax.set_ylabel("time [ns]")

    fig, ax3 = plt.subplots()
    ax3.plot(bragg_curve)
    print("Bragg sum:",np.sum(bragg_curve))

    plt.show()

    return(rms,sd)

reconstruct(event)
