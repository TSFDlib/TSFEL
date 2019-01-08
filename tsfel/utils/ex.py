import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

###########################################a


def n_squared(x, No):
    """The model function"""
    return No*(x)**2

def n_nlog(x, No):
    """The model function"""
    return No*x*np.log(x)

def n_linear(x, No):
    """The model function"""
    return No*(x)

def n_log(x, No):
    """The model function"""
    return No*np.log(x)

def n_const(x, No):
    """The model function"""
    return np.zeros(len(x)) + No

variance = [1.4,0.4,0.8,4.3,3.5,8.7]
centroid = [0.7,1.1,9,77,142,269]
sdev = [0.2,0.3,0.7,1.9,3.2,8.9]
rms = [0.6,0.5,0.8,1.1,1.9,6.2]
itqr = [1.7,1,21,6.1,14,34]
madev = [0.3,1,10,112,208,385]
zerocr = [0.2,0.3,0.7,1.7,2.2,3.7]
autoc = [0.2,0.3,3.6,0.8,0.4,0.7]
maxf = [1.4,1,3.4,25,68,145]
medf = [1.4,1,3.3,25,67,144]
funf = [0.6,0.8,10,341]
maxps = [1.6,1,2.6,15,41,82]
toten = [0.3,0.6,6.9,74,140,270]
sp_centroid = [2.2,0.8,6.3,23,64,135]
sp_spread = [1.6,1.1,4.8,47,126,273]
sp_skew = [1.9,1.5,10,120,286,612]
sp_slope = [1.5,1.1,4.1,25,55,126]
sp_decrease = [0.3,1,13,138,265,543]
sp_rollon = [0.3,0.7,5.3,49,102,212]
sp_rolloff = [1.5,1.1,5.7,50,100,206]
sp_var = [0.4,0.6,2.7,24,54,128]
lin_reg = [0.8,0.6,2.8,25,54,134]
###########################################a

def find_best_curve(signal):

    signal = signal
    all_chisq = []
    list_curves = [n_squared, n_nlog, n_linear, n_log, n_const]
    all_curves = []
    # Model parameters
    stdev = 2
    t = np.array([200,2000,20000,200000])
    sig = np.zeros(len(signal)) + stdev

    # Fit the curve
    for curve in list_curves:

        start = 1
        popt, pcov = curve_fit(curve,t,signal,sigma = sig,p0 = start,absolute_sigma=True)

        # Compute chi square
        Nexp = curve(t, *popt)
        r = signal - Nexp
        chisq = np.sum((r/stdev)**2)
        df = len(signal) - 2
        print("chisq =",chisq,"df =",curve)

        all_chisq.append(chisq)
        all_curves.append(Nexp)


    idx_best = np.argmin(all_chisq)

    plt.errorbar(t, signal, yerr=sig, fmt='o', label='"data"')
    plt.plot(t, all_curves[idx_best], label='fit')
    # plt.plot(t, all_curves[3], label='fit')
    # plt.plot(t, all_curves[1], label='fit')
    plt.legend()
    plt.show()
    curve_name = str(list_curves[idx_best])
    idx1 = curve_name.find("n_")
    idx2 = curve_name.find("at")
    curve_name = curve_name[idx1+2:idx2-1]
    print("This function is "+ curve_name)

    return np.min(all_chisq), curve_name

    # Plot the data with error bars along with the fit result

###########################################a



find_best_curve(funf)

