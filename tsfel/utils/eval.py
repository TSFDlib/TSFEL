import numpy as np
from scipy.optimize import curve_fit
###########################################
#curves
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
    
    curve_name = str(list_curves[idx_best])
    idx1 = curve_name.find("n_")
    idx2 = curve_name.find("at")
    curve_name = curve_name[idx1+2:idx2-1]

    return np.min(all_chisq), curve_name

    # Plot the data with error bars along with the fit result

def compute_complexity(feat_dict):
    
    #chisq, curve_name = find_best_curve(signal)
    return 'not'

#find_best_curve(signal)

