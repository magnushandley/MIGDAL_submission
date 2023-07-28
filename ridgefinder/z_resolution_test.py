import numpy as np
import matplotlib.pyplot as plt
from skimage import io, restoration
import re
import glob
import matplotlib as mpl
from scipy.signal import find_peaks
from scipy.fft import fft2, ifft2

lucyrichardsonito = True
lindeconv = False
timeresolution = 2
driftvel = 0.013
lr_prominence_thresh = 0.575

# raw_ITO_fp = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/z_separation_test/*ITO_raw.txt"))
raw_ITO_fp = sorted(glob.glob("/Users/magnus/Documents/MIGDAL/artificial_tracks/z_separation_high_dep/*ITO_raw.txt"))
prfpath = "/Users/magnus/Documents/MIGDAL/stripreader/stripdeconvolution/single_elec_response_10k.npy"

strip = 14

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

markers = ["x","+","*","o"]
mpl.rcParams['text.usetex'] = True
plt.rcParams.update(params)

def lrdeconvolve(ITO,prfpath,iterations=7,ratio=1/1000):
    """
    Performs Lucy-Richardson deconvolution of the ITO
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
    Linear deconvolution of the ITO signals, the point spread functions must be padded for th
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


for j in range(9):
    zerosep = []
    timeresolution = j+1
    
    predicted = []
    reconstructed = []
    for event in range(len(raw_ITO_fp)):
        with open(raw_ITO_fp[event]) as f:
            lines = f.readlines()
            
        data = np.zeros((350,30))

        i = 0
        for line in lines:
            split = line.split()
            data[i] = split[1:]
            i += 1
                
        datat = -np.transpose(data)

        if lucyrichardsonito == True:
            datat = lrdeconvolve(datat, prfpath)
            data = -np.transpose(datat)

        if lindeconv == True:
            datat = deconv2d(datat, prfpath)
            data = -np.transpose(datat)

        pred_sep = float(re.search('dep_sep_(.+?)_gem_out', raw_ITO_fp[event]).group(1).replace('_', '.'))
        predicted.append(pred_sep)

        strip_peaks, properties = find_peaks(datat[strip][::timeresolution],prominence=lr_prominence_thresh)
        # if (pred_sep > 0.25):
        #     plt.plot(datat[strip][::timeresolution])
        #     plt.show()
        if (len(strip_peaks) > 1):
            exp_sep = (strip_peaks[-1]*driftvel*timeresolution) - (strip_peaks[0]*driftvel*timeresolution)
            reconstructed.append(exp_sep)
        else:
            exp_sep = 0
            zerosep.append(pred_sep)
            reconstructed.append(exp_sep)
        
    def moving_average(a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    rolling_pred = moving_average(predicted,7)
    rolling_exp = moving_average(reconstructed,7)

    end_found = False
    for i in range(len(rolling_pred)):
        if (end_found == False):
            if (rolling_exp[i] > rolling_pred[i]/2) and (rolling_pred[i] > 0):
                end_found = True
                print(j+1," ns threshold:",rolling_pred[i])

    deviations = []
    for i in range(len(predicted)):
        if reconstructed[i] != 0:
            deviations.append(reconstructed[i] - predicted[i])

    print("Average deviation: ",np.mean(deviations))
    print("Standard deviation of offset: ",np.std(deviations))

    plt.scatter(predicted,reconstructed,label="Reconstructed separation, "+str(j+1)+"ns sampling")
    plt.errorbar(predicted, reconstructed, yerr=timeresolution*0.013, fmt="o")
    plt.plot(rolling_pred,rolling_exp,label="Rolling average over 7 reconstructed points")


    plt.plot(predicted,predicted,color="black",label="Perfect reconstructed separation, y=x")
    plt.xlabel("True separation [cm]")
    plt.ylabel("Reconstructed separation [cm]")
    plt.legend()
    plt.title("Reconstruction of low energy lines, separated in z")
    # plt.savefig("/Users/magnus/Documents/MIGDAL/Part III/z_sep_high_dep_lin_deconv"+str(j+1)+"ns.png", dpi=300)
    plt.show()



