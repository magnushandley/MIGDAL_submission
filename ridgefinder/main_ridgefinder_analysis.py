import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import interpolate
from scipy.signal import find_peaks
from skimage import io, restoration
from scipy.fft import fft2, ifft2
from RFsingletiff import *
from file_selection import *
import time
import re


case = 'fe55'

TIFF_files,raw_ITO_files,convl_ITO_files,degrad_files,binning = file_select(case)

event = 13

scan = False
lucyrichardsonito = True
lindeconvito = False
projection = "3d"
plot = True
break_degen = True

#timeresolution is the detector time resolution, in nanoseconds. Output of the simulations is samples at 1ns.
timeresolution = 1
lr_prominence_thresh = 0.575
resolution = 1152
res_midpoint = 576
ITO_size = 10
total_strips = 120

#path to single electron response functions
prfpath = "/Users/magnus/Documents/MIGDAL/stripreader/stripdeconvolution/single_elec_response_10k.npy"


def lrdeconvolve(ITO,prfpath,iterations=8,ratio=1/1000):
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
    # print(np.shape(prf))
    psf[2:,20:] = prf[:3,5:25]
    psf[:2,20:] = prf[28:,5:25]

    return(restoration.richardson_lucy(ITO, psf, num_iter=iterations, clip=False, filter_epsilon=epsilon))

def deconv2d(signals, prfpath):
    """
    Linear deconvolution of the ITO signals - this can be more accurate than Lucy-Richardson but is also prone to large instabilities so should be used with great caution.
    Parameters:
    ----------
    signals: Numpy array of ITO waveforms.
    prfpath: path to the numpy file contating the simulated single electron point response functions.
    Returns
    ----------
    out: Numpy array with deconvolved waveforms
    """
    prf = np.load(prfpath)
    norm = 1/np.sum(prf)
    prf = prf*norm

    #reindex and reduce size
    psf = np.zeros((5,40))
    # print(np.shape(prf))
    psf[2:,20:] = prf[:3,5:25]
    psf[:2,20:] = prf[28:,5:25]

    paddedpsf = np.zeros((30,350))
    paddedpsf[13:18,0:20]= psf[:,:20]
    paddedpsf[13:18,330:350]= psf[:,20:]
    out = np.zeros((30,350))

    intermediate = np.abs(ifft2(fft2(signals) / fft2(paddedpsf)))
    out[15:]=intermediate[:15]
    out[:15]=intermediate[15:]
    return(out)

def extract_theta_phi(filename):
    """
    Takes the filenames of our artificial lines files to extract the polar coordinates from the filenames.
    Parameters
    ---------
    filename: string of filename
    Returns:
    ---------
    theta, phi: Polar angles of straight lines
    """
    theta = re.search('tht(.*)phi', filename)
    theta = theta.group(1)
    theta = float(theta.replace("_","."))
    phi = re.search('phi(.*).csv', filename)
    phi = phi.group(1)
    phi = float(phi.replace("_","."))
    return(theta,phi)

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
    
def findturningpoint(x,y):
    """
    Takes the output spline from the ridgefinder and returns the first point where the diction of the spline reverses in the x-axis. Looks for a change in sign of the differences between x-coordinates.
    Parameters
    ----------
    x, y: numpy arrays containing the x and y coordiante arrays of the 2D image spline
    Returns:
    ----------
    
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
    """
    Removal of gaussian hole pattern then Lucy richardson deconvolution of the camera image.
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

    deconv_im = restoration.richardson_lucy(filtered, PSF, iterations, clip=False)*65535
    
    return(deconv_im)

def cdf(data):
    """
    Plotting of cumulative charge distributions.
    """
    N = len(data)
    
    # sort the data in ascending order
    x = np.sort(data)
    
    # get the cdf values of y
    y = np.arange(N) / float(N)
    
    # plotting
    plt.xlabel('Time offset in z [ns]')
    plt.ylabel('Normalised cumulative charge')
    
    plt.title('Charge CDFs in the z-direction \n raw-perfect deconvolution')
    
    plt.plot(x, y,label="Degrad charge")

def pixeltostrip(pixel, offset):
    """
    Takes the x coordinates of all points along the ridge and returns the coordinates in strips
    """
    pixel = pixel-res_midpoint
    strip = np.add(((ITO_size/resolution)*(total_strips/ITO_size)*pixel), (15 + offset))
    return(strip)

def peaktime(eventdata,timeresolution):
        """
        Returns the sample number at which each ITO strip is at a maximum
        """
        maximums = np.zeros(30)
        maxtimes = np.zeros(30)
        
        #the [::2] is to take every second sample, to mimic the real data
        for i in range(30):
            max = np.amax(eventdata[i][::timeresolution])
            maximums[i] = max
            index = int(np.where(eventdata[i][::timeresolution] == max)[0][0])
            maxtimes[i] = index*1
            
        return(maxtimes)

def scanoffsets(x, y, deltaz, track, min, max, N = 200):
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
        rmsvec[n] = rmsoffset(np.array((track[3],track[4],track[5]*0.013)),np.array((x,y,z)))

    index = np.where(rmsvec == np.min(rmsvec))[0][0]
    return(offsetvec[index])

def stripintegrals(eventdata):
    """
    Inputs
    --------
    eventdata: 3d numpy array, indexing - 1st specifies the strip and 2nd the sample number

    Outputs:
    ---------
    rowintegrals: returns the energy deposition on each strip, as a numpy array
    """
    rowintegrals = np.zeros(30)
    for i in range(30):
        rowdata = eventdata[i]
        max = np.amax(rowdata)
        maxindex = int(np.where(rowdata == max)[0][0])
        
        #finds the points at which the signal drops below zero
        startfound = False
        j = maxindex
        
        while ((startfound == False) and (j >= 0)):
            if rowdata[j] <= 0:
                startpoint = j
                startfound = True
            else:
                j -= 1
          
        endfound = False
        k = maxindex
        while ((endfound == False) and (k<1000)):
            if rowdata[k] <= 0:
                endpoint = k
                endfound = True
            else:
                k += 1
          
        if (max != 0):
            integral = np.sum(rowdata[startpoint:endpoint])
            rowintegrals[i] = integral
        else:
            rowintegrals[i] = 0
        
    return(rowintegrals)

def reconstruct(imgpath, ITOpath, trackpath):
    """
    Main reconstuction
    Parameters:
    ----------
    imgpath: File path to the TIFF file being analysed
    ITO: Filepath to the non-deconvolved ITO response
    trackpath: Filepath to the degrad simulated events
    """

    SIGMA = 2.5 #sigma for derivative determination ~> Related to track width
    lthresh = 0.02 #tracks with a response lower than this are rejected (0 accepts all)
    uthresh = 0 #tracks with a response higher than this are rejected (0 accepts all)
    minlen = 6 #minimum track length accepted
    linkthresh = 50 #maximum distance to be linked
    logim = False
    
    
    offset = 0
    driftvel = 0.013
    
    #read and deconvolve the camera image
    rawimage = io.imread(imgpath)
    image = deconvolveimg(rawimage)
    
    #load the ITO file
    with open(ITOpath) as f:
        lines = f.readlines()
    
    data = np.zeros((350,30))
    
    i = 0
    for line in lines:
        split = line.split()
        data[i] = split[1:]
        i += 1
    
    datat = -np.transpose(data)
    
    #load the degrad track
    track = np.transpose(np.loadtxt(trackpath))


        
    
    #apply relevant deconvolution to the ITO
    if lucyrichardsonito == True:
        datat = lrdeconvolve(datat, prfpath)
        data = -np.transpose(datat)

    elif lindeconvito == True:
        datat = deconv2d(datat, prfpath)
        data = -np.transpose(datat)
        
    #The original ridgefinder implementation was not aware that python arrays were indexed y then x, hence reverse this when taking the output from returnlines.
    y,x = returnlines(image,
        SIGMA,
        lthresh,
        uthresh,
        minlen,
        linkthresh,
        logim,
        fromfile = False)
    bragg_curve = []
        
    #The ridgefinder returns a very finely quantised spline, do not need to reconstruct all these points so reduce the density below
    x = x[::5]
    y = y[::5]
    """
    REDONE RECONSTUCTION START
    """
    brightness = np.zeros(len(x))
    for i in range(len(x)):
        brightness[i] = image[int(x[i])][int(y[i])]
    
    #Provision to only reconstruct a segment of the track, useful for a type A degeneracy.
    def segmentz(x,datat,start,end,peak):
        for i in range(start,end):
            strip = int(pixeltostrip(x[i],0))
            max = np.max(datat[strip][::timeresolution])

            strip_peaks, properties = find_peaks(datat[strip][::timeresolution],prominence=lr_prominence_thresh)

            xrec.append(x[i])
            yrec.append(y[i])
            deltaz.append(strip_peaks[peak]*driftvel*timeresolution)

    global xrec, yrec,deltaz
    deltaz = np.zeros(len(x))

    xturn, yturn, indexturn = findturningpoint(x,y)
    
    #not worth trying to break the degeneracy if the turning point is within a strip width of the end of the track
    if ((xturn != "None") and ((abs(xturn - x[0]) < 10) or (abs(xturn - x[-1]) < 10))):
        xturn = "None"
    
    #degeneracy breaking from here, implementation in flowchart
    if (break_degen == True):
        if (xturn == "None"):
            degenerate = False
            for i in range(len(x)):
                strip = int(pixeltostrip(x[i],0))
                max = np.max(datat[strip][::timeresolution])

                strip_peaks, properties = find_peaks(datat[strip][::timeresolution],prominence=lr_prominence_thresh)
                if (len(strip_peaks) > 1):
                    degenerate = True

            if (degenerate == False):
                # No turn, deg = False"
                peaks = peaktime(datat,timeresolution)
                
                trackstrip = pixeltostrip(x,offset)
                interpolatedpeaks = interpolate.interp1d(np.arange(30), peaks)

                # deltaz = np.zeros(len(x))
                
                for i in range(len(trackstrip)):
                    deltaz[i] = interpolatedpeaks(trackstrip[i])*driftvel*timeresolution
                    bragg_curve.append(np.sqrt(np.abs(brightness[i]*datat[int(pixeltostrip(x[i],0))][int(deltaz[i]/driftvel)])))

                fig, ax = plt.subplots()
                ax.plot(bragg_curve)

            else:

                # No turn, deg = true
                xrec = []
                yrec = []
                deltaz = []

                for i in range(len(x)):
                    strip = int(pixeltostrip(x[i],0))
                    max = np.max(datat[strip][::timeresolution])

                    strip_peaks, properties = find_peaks(datat[strip][::timeresolution],prominence= lr_prominence_thresh)

                    for j in range(len(strip_peaks)):
                        xrec.append(x[i])
                        yrec.append(y[i])
                        deltaz.append(strip_peaks[j]*driftvel*timeresolution)

                x = np.array(xrec)
                y = np.array(yrec)
                deltaz = np.array(deltaz)

        else:
            # Type A degeneracy breaking
            xrec = []
            yrec = []
            deltaz = []

            #CASE 1, top then bottom:
            segmentz(x,datat,0,indexturn,0)
            segmentz(x,datat,indexturn,len(x),-1)

            deltaz1 = deltaz

            xrec = []
            yrec = []
            deltaz = []

            sum1 = 0
            for i in range(len(x)):
                sum1 += (brightness[i]*datat[int(pixeltostrip(x[i],0))][int(deltaz1[i]/driftvel)])


            #CASE 2, top then bottom:
            segmentz(x,datat,0,indexturn,-1)
            segmentz(x,datat,indexturn,len(x),0)

            deltaz2 = deltaz
            

            sum2 = 0
            for i in range(len(x)):
                sum2 += (brightness[i]*datat[int(pixeltostrip(x[i],0))][int(deltaz2[i]/driftvel)])

            x = np.array(xrec)
            y = np.array(yrec)
            if (sum1>sum2):
                deltaz = np.array(deltaz1)
                # print("1 greater!!")
            else:
                deltaz = np.array(deltaz2)
                # print("2 greater!!")

    elif (break_degen == False):
        peaks = peaktime(datat,timeresolution)
                
        trackstrip = pixeltostrip(x,offset)
        interpolatedpeaks = interpolate.interp1d(np.arange(30), peaks)

        # deltaz = np.zeros(len(x))
                
        for i in range(len(trackstrip)):
            deltaz[i] = interpolatedpeaks(trackstrip[i])*driftvel*timeresolution
            bragg_curve.append(np.sqrt(np.abs(brightness[i]*datat[int(pixeltostrip(x[i],0))][int(deltaz[i]/driftvel)])))

        fig, ax = plt.subplots()
        ax.plot(bragg_curve)
    """
    STOP
    """
    
    x_pix = x
    y_pix = y

    x = (x-res_midpoint)*(ITO_size/resolution)
    y = (y-res_midpoint)*(ITO_size/resolution)

    zoffset = scanoffsets(x, y, deltaz, track, -1.0,0)
    z = deltaz + zoffset

    output = np.array([x,y,z])
    np.savetxt("nx_test_3.txt",output)

    rms = rmsoffset(np.array((track[3],track[4],track[5]*0.013)),np.array((x,y,z)))
    rms2 = rmsoffset(np.array((x,y,z)),np.array((track[3],track[4],track[5]*driftvel)))
    sd = np.sqrt((np.std(track[3])**2)+(np.std(track[4])**2)+(np.std(track[5]*driftvel)**2))

    if plot == True:
        fig = plt.figure()
        if projection == "3d":
            ax2 = plt.axes(projection='3d')
            scatter_plot = ax2.scatter3D(x,y,z,color="red")
            print(len(x))
            scatter_plot = ax2.scatter3D(track[3][::binning],track[4][::binning],track[5][::binning]*0.013,color="blue")
            # ax2.plot3D(x,y,z, color="black")
            ax2.set_xlabel("x [cm]")
            ax2.set_ylabel("y [cm]")
            ax2.set_zlabel("z [cm]")

        if projection == "xz":
            ax2 = plt.axes()
            scatter_plot = ax2.scatter(track[3],track[5]*0.013,label="diffused electrons")
            scatter_plot = ax2.scatter(x,z,label="reconstructed points") 
            ax2.legend()
            # ax2.plot(x,z, color="black")
            ax2.set_xlabel("x [cm]")
            ax2.set_ylabel("z [cm]")

        if projection == "ITO":
            try:
                theta, phi = extract_theta_phi(imgpath)
                print(theta, phi)
            except:
                theta, phi = 0,0
            ax2 = plt.axes()
            im = ax2.pcolormesh(-data,vmin=0)
            fig.colorbar(im, ax=ax2,label="ITO waveform intensity")
            ax2.set_xlabel("Strip number")
            ax2.set_ylabel("Time [ns]")
            ax2.set_title("LR deconvolved ITO response \n for simulated migdal event 0")

        if projection == "lr_image":
            ax2 = plt.axes()
            im = ax2.pcolormesh(np.transpose(np.log(image+1)),vmin=0)
            fig.colorbar(im, ax=ax2,label="Log(image intensity)")
            plt.plot(x_pix,y_pix,color="white")
            print(x_pix,y_pix)
            ax2.set_xlabel("x-coordinate, pixels")
            ax2.set_ylabel("y-coordinate, pixels")
            ax2.set_title("LR deconvolved image and ridgefinder spline fit, \n for simulated migdal event 0")

        toc = time.time()

        plt.show()
    
    return(rms, sd, zoffset, rms/sd,rms2)

global counter

eventstoinvestigate = np.transpose(np.loadtxt("/Users/magnus/Documents/MIGDAL/stripreader/degeneracies.txt"))


if (scan == True):
    for event in range(len(TIFF_files)):
    
        # theta, phi = extract_theta_phi(TIFF_files[event])
        theta,phi = 0,0
        rms, sd, zoffset, ratio, rms2 = reconstruct(TIFF_files[int(event)],raw_ITO_files[int(event)],degrad_files[int(event)])
        print(event," ",rms," ",sd," ",ratio," ",zoffset," ",theta," ",phi," ",rms2)

elif (scan == False):
     rms, sd, zoffset, ratio,rms2 = reconstruct(TIFF_files[event],raw_ITO_files[event],degrad_files[event])
     print(rms," ",sd," ",ratio," ",zoffset," ",rms2)


