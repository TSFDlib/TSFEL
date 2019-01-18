from numpy.testing import assert_array_equal, run_module_suite
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.features import *
import matplotlib.pylab as plt
import novainstrumentation as ni
import time

const0 = np.zeros(20)
const1 = np.ones(20)
constNeg = np.ones(20)*(-1)
constF = np.ones(20) * 2.5
lin = np.arange(20)
lin0 = np.linspace(-10,10,20)
f = 5
sample = 1000
x = np.arange(0, sample, 1)
Fs = 1000
wave = np.sin(2 * np.pi * f * x / Fs)
np.random.seed(seed=10)
noiseWave = wave + np.random.normal(0,0.1,1000)
offsetWave = wave + 2
def test_mean():
    np.testing.assert_equal(np.mean(const0), 0.0)
    np.testing.assert_equal(np.mean(const1), 1.0)
    np.testing.assert_equal(np.mean(constNeg), -1.0)
    np.testing.assert_equal(np.mean(constF), 2.5)
    np.testing.assert_equal(np.mean(lin), 9.5)
    np.testing.assert_almost_equal(np.mean(lin0), -3.552713678800501e-16, decimal=5)
    np.testing.assert_almost_equal(np.mean(wave), 7.105427357601002e-18, decimal=5)
    np.testing.assert_almost_equal(np.mean(offsetWave), 2.0, decimal=5)
    np.testing.assert_almost_equal(np.mean(noiseWave), -0.0014556635615470554, decimal=5)

def test_max():
    np.testing.assert_equal(np.max(const0), 0.0)
    np.testing.assert_equal(np.max(const1), 1.0)
    np.testing.assert_equal(np.max(constNeg), -1.0)
    np.testing.assert_equal(np.max(constF), 2.5)
    np.testing.assert_equal(np.max(lin), 19)
    np.testing.assert_almost_equal(np.max(lin0), 10.0, decimal=5)
    np.testing.assert_almost_equal(np.max(wave), 1.0, decimal=5)
    np.testing.assert_almost_equal(np.max(noiseWave), 1.221757617217142, decimal=5)
    np.testing.assert_almost_equal(np.max(offsetWave), 3.0, decimal=5)


def test_min():
    np.testing.assert_equal(np.min(const0), 0.0)
    np.testing.assert_equal(np.min(const1), 1.0)
    np.testing.assert_equal(np.min(constNeg), -1.0)
    np.testing.assert_equal(np.min(constF), 2.5)
    np.testing.assert_equal(np.min(lin), 0)
    np.testing.assert_almost_equal(np.min(lin0), -10.0, decimal=5)
    np.testing.assert_almost_equal(np.min(wave), -1.0, decimal=5)
    np.testing.assert_almost_equal(np.min(noiseWave), -1.2582533627830566, decimal=5)
    np.testing.assert_almost_equal(np.min(offsetWave), 1.0, decimal=5)
#
def test_calc_meanad():
    # output = np.sum(np.abs(b-np.median(b)))/len(b)
    np.testing.assert_equal(calc_meanad(const0), 0.0)
    np.testing.assert_equal(calc_meanad(const1), 0.0)
    np.testing.assert_equal(calc_meanad(constNeg), 0.0)
    np.testing.assert_equal(calc_meanad(constF), 0.0)
    np.testing.assert_equal(calc_meanad(lin), 5.0)
    np.testing.assert_almost_equal(calc_meanad(lin0), 5.263157894736842, decimal=5)
    np.testing.assert_almost_equal(calc_meanad(wave), 0.6365674116287159, decimal=5)
    np.testing.assert_almost_equal(calc_meanad(noiseWave), 0.6392749078483896, decimal=5)
    np.testing.assert_almost_equal(calc_meanad(offsetWave), 0.6365674116287157, decimal=5)

def test_calc_medad():
    # output = np.sum(np.abs(b-np.median(b)))/len(b)
    np.testing.assert_equal(calc_medad(const0), 0.0)
    np.testing.assert_equal(calc_medad(const1), 0.0)
    np.testing.assert_equal(calc_medad(constNeg), 0.0)
    np.testing.assert_equal(calc_medad(constF), 0.0)
    np.testing.assert_equal(calc_medad(lin), 5.0)
    np.testing.assert_almost_equal(calc_medad(lin0), 5.2631578947368425, decimal=5)
    np.testing.assert_almost_equal(calc_medad(wave), 0.7071067811865475, decimal=5)
    np.testing.assert_almost_equal(calc_medad(offsetWave), 0.7071067811865475, decimal=5)
    np.testing.assert_almost_equal(calc_medad(noiseWave), 0.7068117164205888, decimal=5)

def test_calc_meandiff():
    # output = np.sum(np.abs(b-np.median(b)))/len(b)
    np.testing.assert_equal(calc_meandiff(const0), 0.0)
    np.testing.assert_equal(calc_meandiff(const1), 0.0)
    np.testing.assert_equal(calc_meandiff(constNeg), 0.0)
    np.testing.assert_equal(calc_meandiff(constF), 0.0)
    np.testing.assert_equal(calc_meandiff(lin), 1.0)
    np.testing.assert_almost_equal(calc_meandiff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(calc_meandiff(wave), -3.1442201279407477e-05, decimal=5)
    np.testing.assert_almost_equal(calc_meandiff(offsetWave), -3.1442201279407036e-05, decimal=5)
    np.testing.assert_almost_equal(calc_meandiff(noiseWave), -0.00010042477181949707, decimal=5)

def test_calc_meddiff():
    np.testing.assert_equal(calc_meddiff(const0), 0.0)
    np.testing.assert_equal(calc_meddiff(const1), 0.0)
    np.testing.assert_equal(calc_meddiff(constNeg), 0.0)
    np.testing.assert_equal(calc_meddiff(constF), 0.0)
    np.testing.assert_equal(calc_meddiff(lin), 1.0)
    np.testing.assert_almost_equal(calc_meddiff(lin0), 1.0526315789473681, decimal=5)
    np.testing.assert_almost_equal(calc_meddiff(wave), -0.0004934396342684, decimal=5)
    np.testing.assert_almost_equal(calc_meddiff(offsetWave), -0.0004934396342681779, decimal=5)
    np.testing.assert_almost_equal(calc_meddiff(noiseWave), -0.004174819648320949, decimal=5)

def test_calc_meanadiff():
    np.testing.assert_equal(calc_meanadiff(const0), 0.0)
    np.testing.assert_equal(calc_meanadiff(const1), 0.0)
    np.testing.assert_equal(calc_meanadiff(constNeg), 0.0)
    np.testing.assert_equal(calc_meanadiff(constF), 0.0)
    np.testing.assert_equal(calc_meanadiff(lin), 1.0)
    np.testing.assert_almost_equal(calc_meanadiff(lin0), 1.0526315789473684, decimal=5)
    np.testing.assert_almost_equal(calc_meanadiff(wave), 0.019988577818740614, decimal=5)
    np.testing.assert_almost_equal(calc_meanadiff(offsetWave), 0.019988577818740614, decimal=5)
    np.testing.assert_almost_equal(calc_meanadiff(noiseWave), 0.10700252903161508, decimal=5)

def test_calc_medadiff():
    np.testing.assert_equal(calc_medadiff(const0), 0.0)
    np.testing.assert_equal(calc_medadiff(const1), 0.0)
    np.testing.assert_equal(calc_medadiff(constNeg), 0.0)
    np.testing.assert_equal(calc_medadiff(constF), 0.0)
    np.testing.assert_equal(calc_medadiff(lin), 1.0)
    np.testing.assert_almost_equal(calc_medadiff(lin0), 1.0526315789473681, decimal=5)
    np.testing.assert_almost_equal(calc_medadiff(wave), 0.0218618462348652, decimal=5)
    np.testing.assert_almost_equal(calc_medadiff(offsetWave), 0.021861846234865645, decimal=5)
    np.testing.assert_almost_equal(calc_medadiff(noiseWave), 0.08958750592592835, decimal=5)

def test_calc_sadiff():
    np.testing.assert_equal(calc_sadiff(const0), 0.0)
    np.testing.assert_equal(calc_sadiff(const1), 0.0)
    np.testing.assert_equal(calc_sadiff(constNeg), 0.0)
    np.testing.assert_equal(calc_sadiff(constF), 0.0)
    np.testing.assert_equal(calc_sadiff(lin), 19)
    np.testing.assert_almost_equal(calc_sadiff(lin0), 20.0, decimal=5)
    np.testing.assert_almost_equal(calc_sadiff(wave), 19.968589240921872, decimal=5)
    np.testing.assert_almost_equal(calc_sadiff(offsetWave), 19.968589240921872, decimal=5)
    np.testing.assert_almost_equal(calc_sadiff(noiseWave), 106.89552650258346, decimal=5)

def test_variance():
    # np.sqrt(np.mean(abs(b - np.mean(b))**2))**2
    # pvariance
    np.testing.assert_equal(np.var(const0), 0.0)
    np.testing.assert_equal(np.var(const1), 0.0)
    np.testing.assert_equal(np.var(constNeg), 0.0)
    np.testing.assert_equal(np.var(constF), 0.0)
    np.testing.assert_equal(np.var(lin), 33.25)
    np.testing.assert_almost_equal(np.var(lin0), 36.84210526315789, decimal=5)
    np.testing.assert_almost_equal(np.var(wave), 0.5, decimal=5)
    np.testing.assert_almost_equal(np.var(offsetWave), 0.5, decimal=5)
    np.testing.assert_almost_equal(np.var(noiseWave), 0.5081167177369529, decimal=5)

def test_std(): #population standard deviation
    # np.sqrt(np.mean(abs(b - np.mean(b))**2)
    np.testing.assert_equal(np.std(const0), 0.0)
    np.testing.assert_equal(np.std(const1), 0.0)
    np.testing.assert_equal(np.std(constNeg), 0.0)
    np.testing.assert_equal(np.std(constF), 0.0)
    np.testing.assert_equal(np.std(lin), 5.766281297335398)
    np.testing.assert_almost_equal(np.std(lin0), 6.069769786668839, decimal=5)
    np.testing.assert_almost_equal(np.std(wave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(np.std(offsetWave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(np.std(noiseWave), 0.7128230620125536, decimal=5)

def test_rms():
    # np.sqrt(np.mean(b**2))
    np.testing.assert_equal(rms(const0), 0.0)
    np.testing.assert_equal(rms(const1), 1.0)
    np.testing.assert_equal(rms(constNeg), 1.0)
    np.testing.assert_equal(rms(constF), 2.5)
    np.testing.assert_equal(rms(lin), 11.090536506409418)
    np.testing.assert_almost_equal(rms(lin0), 6.06976978666884, decimal=5)
    np.testing.assert_almost_equal(rms(wave), 0.7071067811865476, decimal=5)
    np.testing.assert_almost_equal(rms(offsetWave), 2.1213203435596424, decimal=5)
    np.testing.assert_almost_equal(rms(noiseWave), 0.7128245483240299, decimal=5)

def test_int_range():
    # np.percentile(b, 75) - np.percentile(b, 25)
    np.testing.assert_equal(interq_range(const0), 0.0)
    np.testing.assert_equal(interq_range(const1), 0.0)
    np.testing.assert_equal(interq_range(constNeg), 0.0)
    np.testing.assert_equal(interq_range(constF), 0.0)
    np.testing.assert_equal(interq_range(lin), 9.5)
    np.testing.assert_almost_equal(interq_range(lin0), 10.0, decimal=5)
    np.testing.assert_almost_equal(interq_range(wave), 1.414213562373095, decimal=5)
    np.testing.assert_almost_equal(interq_range(offsetWave), 1.414213562373095, decimal=5)
    np.testing.assert_almost_equal(interq_range(noiseWave), 1.4277110228590328, decimal=5)

def test_zeroCross():
    np.testing.assert_equal(zero_cross(const0), 0)
    np.testing.assert_equal(zero_cross(const1), 0)
    np.testing.assert_equal(zero_cross(constNeg), 0)
    np.testing.assert_equal(zero_cross(constF), 0)
    np.testing.assert_equal(zero_cross(lin), 1)
    np.testing.assert_almost_equal(zero_cross(lin0), 1, decimal=5)
    np.testing.assert_almost_equal(zero_cross(wave), 10, decimal=5)
    np.testing.assert_almost_equal(zero_cross(offsetWave), 0, decimal=5)
    np.testing.assert_almost_equal(zero_cross(noiseWave), 38, decimal=5)

def test_autocorr():
    np.testing.assert_equal(autocorr(const0), 0.0)
    np.testing.assert_equal(autocorr(const1), 20.0)
    np.testing.assert_equal(autocorr(constNeg), 20.0)
    np.testing.assert_equal(autocorr(constF), 125.0)
    np.testing.assert_equal(autocorr(lin), 2470.0)
    np.testing.assert_almost_equal(autocorr(lin0), 736.8421052631579, decimal=0)
    np.testing.assert_almost_equal(autocorr(wave), 500.5, decimal=0)
    np.testing.assert_almost_equal(autocorr(offsetWave), 4500.0, decimal=0)
    np.testing.assert_almost_equal(autocorr(noiseWave), 508.6149018530489, decimal=0)

def test_skew():
    # corrcoef(a, b)[0, 1]
    np.testing.assert_equal(skew(const0), 0.0)
    np.testing.assert_equal(skew(const1), 0.0)
    np.testing.assert_equal(skew(constNeg), 0.0)
    np.testing.assert_equal(skew(constF), 0.0)
    np.testing.assert_equal(skew(lin), 0)
    np.testing.assert_almost_equal(skew(lin0), -1.0167718723297815e-16, decimal=5)
    np.testing.assert_almost_equal(skew(wave), -2.009718347115232e-17, decimal=5)
    np.testing.assert_almost_equal(skew(offsetWave), 9.043732562018544e-16, decimal=5)
    np.testing.assert_almost_equal(skew(noiseWave), -0.0004854111290521465, decimal=5)

def test_kurtosis():
    np.testing.assert_equal(kurtosis(const0), -3)
    np.testing.assert_equal(kurtosis(const1), -3)
    np.testing.assert_equal(kurtosis(constNeg), -3)
    np.testing.assert_equal(kurtosis(constF), -3.0)
    np.testing.assert_almost_equal(kurtosis(lin), -1.206015037593985, decimal=2)
    np.testing.assert_almost_equal(kurtosis(lin0), -1.2060150375939847, decimal=2)
    np.testing.assert_almost_equal(kurtosis(wave), -1.501494077162359, decimal=2)
    np.testing.assert_almost_equal(kurtosis(offsetWave), -1.5014940771623597, decimal=2)
    np.testing.assert_almost_equal(kurtosis(noiseWave), -1.4606204906023366, decimal=2)
#
# def test_max_fre():
#     np.testing.assert_equal(max_frequency(const0, Fs), 0)
#     np.testing.assert_equal(max_frequency(const1, Fs), 3)
#     np.testing.assert_equal(max_frequency(constNeg, Fs), 3)
#     np.testing.assert_equal(max_frequency(constF, Fs), 0)
#     np.testing.assert_equal(max_frequency(lin, Fs), 3)
#     np.testing.assert_almost_equal(max_frequency(lin0, Fs), 3, decimal=5)
#     np.testing.assert_almost_equal(max_frequency(wave, Fs), 0, decimal=5)
#     np.testing.assert_almost_equal(max_frequency(offsetWave, Fs), 3, decimal=5)
#     np.testing.assert_almost_equal(max_frequency(noiseWave, Fs), 3, decimal=5)
#     np.testing.assert_almost_equal(max_frequency(x, Fs),  0.7301126158288906, decimal=1)

# def test_corr():
#     # corrcoef(a, b)[0, 1]
#     np.testing.assert_equal(correlation(const0), 0)
#     np.testing.assert_equal(correlation(const1), 3)
#     np.testing.assert_equal(correlation(constNeg), 3)
#     np.testing.assert_equal(correlation(constF), 0)
#     np.testing.assert_equal(correlation(lin), 3)
#     np.testing.assert_almost_equal(correlation(lin0), 3, decimal=5)
#     np.testing.assert_almost_equal(correlation(wave), 0, decimal=5)
#     np.testing.assert_almost_equal(correlation(offsetWave), 3, decimal=5)
#     np.testing.assert_almost_equal(correlation(noiseWave), 3, decimal=5)
#

# def test_max_fre():
#     f = 0.2
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f * x / Fs)
#     np.testing.assert_almost_equal(max_frequency(y, Fs), 0.2, decimal=0)
#     f2 = 0.5
#     y = np.cos(2 * np.pi * f * x / Fs) + np.cos(2 * np.pi * f2 * x / Fs)
#     np.testing.assert_almost_equal(max_frequency(y, Fs), 0.5, decimal=0)
#     a = np.ones(1000)
#     np.testing.assert_almost_equal(max_frequency(a, 1), 0, decimal=0)

def test_max_fre():
    np.testing.assert_equal(max_frequency(const0, Fs), 0)
    np.testing.assert_equal(max_frequency(const1, Fs), 3)
    np.testing.assert_equal(max_frequency(constNeg, Fs), 3)
    np.testing.assert_equal(max_frequency(constF, Fs), 0)
    np.testing.assert_equal(max_frequency(lin, Fs), 3)
    np.testing.assert_almost_equal(max_frequency(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(max_frequency(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(max_frequency(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(max_frequency(noiseWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(max_frequency(x, Fs),  0.7301126158288906, decimal=1)

def test_med_fre():
    np.testing.assert_equal(median_frequency(const0, Fs), 0.0)
    np.testing.assert_equal(median_frequency(const1, Fs), 0.0)
    np.testing.assert_equal(median_frequency(constNeg, Fs), 0.0)
    np.testing.assert_equal(median_frequency(constF, Fs), 0)
    np.testing.assert_equal(median_frequency(lin, Fs), 3)
    np.testing.assert_almost_equal(median_frequency(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(median_frequency(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(median_frequency(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(median_frequency(noiseWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(median_frequency(x, Fs),  0.7301126158288906, decimal=1)


#
# def fundamentals(frames0, samplerate):
#     mid = 16
#     sample = mid*2+1
#     res = []
#     for first in xrange(sample):
#         last = first-sample
#         frames = frames0[first:last]
#         res.append(_fundamentals(frames, samplerate))
#     res = sorted(res)
#     return res[mid] # We use the medium value


# def _fundamentals(frames, samplerate):
#     frames2=frames*hamming(len(frames));
#     frameSize=len(frames);
#     ceps=np.fft.ifft(np.log(np.abs(fft(frames2))))
#     nceps=ceps.shape[-1]*2/3
#     peaks = []
#     k=3
#     while(k  y1 and y2 >= y3): peaks.append([float(samplerate)/(k+2),abs(y2), k, nceps])
#         k=k+1
#     maxi=max(peaks, key=lambda x: x[1])
#     return maxi[0]
#
#
def test_fund_fre():
    np.testing.assert_equal(fundamental_frequency(const0, 1), 0)
    np.testing.assert_equal(fundamental_frequency(const1, 1), 3)
    np.testing.assert_equal(fundamental_frequency(constNeg, Fs), 3)
    np.testing.assert_equal(fundamental_frequency(constF, Fs), 0)
    np.testing.assert_equal(fundamental_frequency(lin, Fs), 3)
    np.testing.assert_almost_equal(fundamental_frequency(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(fundamental_frequency(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(fundamental_frequency(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(fundamental_frequency(noiseWave, Fs), 3, decimal=5)


def test_hist():
    np.testing.assert_equal(hist_json(const0, 10, 5), 0)
    np.testing.assert_equal(hist_json(const1, 10, 5), 3)
    np.testing.assert_equal(hist_json(constNeg, 10, 5), 3)
    np.testing.assert_equal(hist_json(constF, 10, 5), 0)
    np.testing.assert_equal(hist_json(lin, 10, 5), 3)
    np.testing.assert_almost_equal(hist_json(lin0, 10, 5), 3, decimal=5)
    np.testing.assert_almost_equal(hist_json(wave, 10, 5), 0, decimal=5)
    np.testing.assert_almost_equal(hist_json(offsetWave, 10, 5), 3, decimal=5)
    np.testing.assert_almost_equal(hist_json(noiseWave, 10, 5), 3, decimal=5)


def test_power_spec():
    np.testing.assert_equal(power_spectrum(const0, Fs), 0)
    np.testing.assert_equal(power_spectrum(const1, Fs), 3)
    np.testing.assert_equal(power_spectrum(constNeg, Fs), 3)
    np.testing.assert_equal(power_spectrum(constF, Fs), 0)
    np.testing.assert_equal(power_spectrum(lin, Fs), 3)
    np.testing.assert_almost_equal(power_spectrum(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(power_spectrum(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(power_spectrum(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(power_spectrum(noiseWave, Fs), 3, decimal=5)


def test_total_energy():
    np.testing.assert_equal(total_energy_(const0, Fs), 0)
    np.testing.assert_equal(total_energy_(const1, Fs), 3)
    np.testing.assert_equal(total_energy_(constNeg, Fs), 3)
    np.testing.assert_equal(total_energy_(constF, Fs), 0)
    np.testing.assert_equal(total_energy_(lin, Fs), 3)
    np.testing.assert_almost_equal(total_energy_(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(total_energy_(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(total_energy_(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(total_energy_(noiseWave, Fs), 3, decimal=5)

#
# def centroid(spectrum, f):
#     s = spectrum / np.sum(spectrum)
#     return np.dot(s, f)
#
#
def test_spectral_centroid():
    np.testing.assert_equal(spectral_centroid(const0, Fs), 0)
    np.testing.assert_equal(spectral_centroid(const1, Fs), 3)
    np.testing.assert_equal(spectral_centroid(constNeg, Fs), 3)
    np.testing.assert_equal(spectral_centroid(constF, Fs), 0)
    np.testing.assert_equal(spectral_centroid(lin, Fs), 3)
    np.testing.assert_almost_equal(spectral_centroid(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_centroid(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(spectral_centroid(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_centroid(noiseWave, Fs), 3, decimal=5)


def test_spectral_spread():
    np.testing.assert_equal(spectral_spread(const0, Fs), 0)
    np.testing.assert_equal(spectral_spread(const1, Fs), 3)
    np.testing.assert_equal(spectral_spread(constNeg, Fs), 3)
    np.testing.assert_equal(spectral_spread(constF, Fs), 0)
    np.testing.assert_equal(spectral_spread(lin, Fs), 3)
    np.testing.assert_almost_equal(spectral_spread(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_spread(noiseWave, Fs), 3, decimal=5)


def test_spectral_skewness():
    np.testing.assert_equal(spectral_skewness(const0, Fs), 0)
    np.testing.assert_equal(spectral_skewness(const1, Fs), 3)
    np.testing.assert_equal(spectral_skewness(constNeg, Fs), 3)
    np.testing.assert_equal(spectral_skewness(constF, Fs), 0)
    np.testing.assert_equal(spectral_skewness(lin, Fs), 3)
    np.testing.assert_almost_equal(spectral_skewness(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_skewness(noiseWave, Fs), 3, decimal=5)

#
# def _kurtosis(spectrum, f, spr, c):
#     s = spectrum / np.sum(spectrum)
#     return np.dot(s, (f - c)**4) / spr**2
#
#
def test_spectral_kurtosis():
    np.testing.assert_equal(spectral_kurtosis(const0, Fs), 0)
    np.testing.assert_equal(spectral_kurtosis(const1, Fs), 3)
    np.testing.assert_equal(spectral_kurtosis(constNeg, Fs), 3)
    np.testing.assert_equal(spectral_kurtosis(constF, Fs), 0)
    np.testing.assert_equal(spectral_kurtosis(lin, Fs), 3)
    np.testing.assert_almost_equal(spectral_kurtosis(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_kurtosis(noiseWave, Fs), 3, decimal=5)

#
# def slope(ff, f):
#     c = np.vstack([f, np.ones(len(f))]).T
#     return np.linalg.lstsq(c, ff)[0][0]
#
#
def test_spectral_slope():
    np.testing.assert_equal(spectral_slope(const0, Fs), 0)
    np.testing.assert_equal(spectral_slope(const1, Fs), 3)
    np.testing.assert_equal(spectral_slope(constNeg, Fs), 3)
    np.testing.assert_equal(spectral_slope(constF, Fs), 0)
    np.testing.assert_equal(spectral_slope(lin, Fs), 3)
    np.testing.assert_almost_equal(spectral_slope(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_slope(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(spectral_slope(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_slope(noiseWave, Fs), 3, decimal=5)



def test_spectral_decrease():
    np.testing.assert_equal(spectral_decrease(const0, Fs), 0)
    np.testing.assert_equal(spectral_decrease(const1, Fs), 3)
    np.testing.assert_equal(spectral_decrease(constNeg, Fs), 3)
    np.testing.assert_equal(spectral_decrease(constF, Fs), 0)
    np.testing.assert_equal(spectral_decrease(lin, Fs), 3)
    np.testing.assert_almost_equal(spectral_decrease(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_decrease(noiseWave, Fs), 3, decimal=5)


# def roll_on(spectrum, f):
#     sqr = np.square(spectrum)
#     total = np.sum(sqr)
#     s = 0
#     output = 0
#     for i in range(0, len(f)):
#         s += sqr[i]
#         if s >= 0.05 * total:
#             output = f[i]
#             break
#     return output
#
#
def test_spectral_roll_on():
    np.testing.assert_equal(spectral_roll_on(const0, Fs), 0)
    np.testing.assert_equal(spectral_roll_on(const1, Fs), 3)
    np.testing.assert_equal(spectral_roll_on(constNeg, Fs), 3)
    np.testing.assert_equal(spectral_roll_on(constF, Fs), 0)
    np.testing.assert_equal(spectral_roll_on(lin, Fs), 3)
    np.testing.assert_almost_equal(spectral_roll_on(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spectral_roll_on(noiseWave, Fs), 3, decimal=5)


# def roll_off(spectrum, f):
#     sqr = np.square(spectrum)
#     total = np.sum(sqr)
#     s = 0
#     output = 0
#     for i in range(0, len(f)):
#         s += sqr[i]
#         if s >= 0.95 * total:
#             output = f[i]
#             break
#     return output
#
#
def test_spectral_roll_off():
    np.testing.assert_equal(test_spectral_roll_off(const0, Fs), 0)
    np.testing.assert_equal(test_spectral_roll_off(const1, Fs), 3)
    np.testing.assert_equal(test_spectral_roll_off(constNeg, Fs), 3)
    np.testing.assert_equal(test_spectral_roll_off(constF, Fs), 0)
    np.testing.assert_equal(test_spectral_roll_off(lin, Fs), 3)
    np.testing.assert_almost_equal(test_spectral_roll_off(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(test_spectral_roll_off(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(test_spectral_roll_off(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(test_spectral_roll_off(noiseWave, Fs), 3, decimal=5)


def test_spectral_curve_distance():
    np.testing.assert_equal(curve_distance(const0, Fs), 0)
    np.testing.assert_equal(curve_distance(const1, Fs), 3)
    np.testing.assert_equal(curve_distance(constNeg, Fs), 3)
    np.testing.assert_equal(curve_distance(constF, Fs), 0)
    np.testing.assert_equal(curve_distance(lin, Fs), 3)
    np.testing.assert_almost_equal(curve_distance(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(curve_distance(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(curve_distance(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(curve_distance(noiseWave, Fs), 3, decimal=5)


def test_spect_variation():
    np.testing.assert_equal(spect_variation(const0, Fs), 0)
    np.testing.assert_equal(spect_variation(const1, Fs), 3)
    np.testing.assert_equal(spect_variation(constNeg, Fs), 3)
    np.testing.assert_equal(spect_variation(constF, Fs), 0)
    np.testing.assert_equal(spect_variation(lin, Fs), 3)
    np.testing.assert_almost_equal(spect_variation(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spect_variation(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(spect_variation(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(spect_variation(noiseWave, Fs), 3, decimal=5)

def test_distance():
    np.testing.assert_equal(distance(const0, Fs), 0)
    np.testing.assert_equal(distance(const1, Fs), 3)
    np.testing.assert_equal(distance(constNeg, Fs), 3)
    np.testing.assert_equal(distance(constF, Fs), 0)
    np.testing.assert_equal(distance(lin, Fs), 3)
    np.testing.assert_almost_equal(distance(lin0, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(distance(wave, Fs), 0, decimal=5)
    np.testing.assert_almost_equal(distance(offsetWave, Fs), 3, decimal=5)
    np.testing.assert_almost_equal(distance(noiseWave, Fs), 3, decimal=5)

# # def test_centroid():
# #     a = np.arange(0,20)
# #     fs = len(a) / a[-1]
# #     t = np.arange(len(a))
# #     output = np.sum(rms(a)*t)/np.sum(rms(a))
# #     np.testing.assert_equal(centroid(a,fs), output)
#
#
#
# def test_corr():
#     # corrcoef(a, b)[0, 1]
#     a = [1, 2, 3, 4, 5]
#     b = [1, 2, 3, 4, 5]
#     c = [-5, 0, -2, 6, -3]
#     d = [-1, -2, -3, -4, -5]
#
#     np.testing.assert_equal(correlation(a, b), 1)
#     np.testing.assert_almost_equal(correlation(b, c), 0.37582, decimal=5)
#     np.testing.assert_equal(correlation(a, d), -1)
#
#
# def test_autocorr():
#     a = np.array([0, 0, 0, 0, 0])
#     b = np.array([1, 2, 3, 4, 5])
#
#     np.testing.assert_equal(autocorr(a), 0)
#     np.testing.assert_equal(autocorr(b), 55)
#
# def test_kurtosis():
#     np.random.seed(seed=23)
#     x = np.random.normal(0, 2, 1000000)
#     np.testing.assert_almost_equal(kurtosis(x), 0, decimal=1)
#     mu, sigma = 0, 0.1
#     x = mu + sigma * np.random.randn(100)
#     np.testing.assert_almost_equal(kurtosis(x),  0.7301126158288906, decimal=1)
#
#
# def test_max_fre():
#     f = 0.2
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f * x / Fs)
#     np.testing.assert_almost_equal(max_frequency(y, Fs), 0.2, decimal=0)
#     f2 = 0.5
#     y = np.cos(2 * np.pi * f * x / Fs) + np.cos(2 * np.pi * f2 * x / Fs)
#     np.testing.assert_almost_equal(max_frequency(y, Fs), 0.5, decimal=0)
#     a = np.ones(1000)
#     np.testing.assert_almost_equal(max_frequency(a, 1), 0, decimal=0)
#
#
# def test_med_fre():
#     f = 0.2
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f * x / Fs)
#     np.testing.assert_almost_equal(median_frequency(y, Fs), 0.1, decimal=0)
#     f2 = 1
#     y = np.cos(2 * np.pi * f * x / Fs) + np.cos(2 * np.pi * f2 * x / Fs) # ???
#     np.testing.assert_almost_equal(median_frequency(y, Fs), 0.010000300007000151, decimal=0)
#     a = np.ones(1000)
#     np.testing.assert_almost_equal(median_frequency(a, 1), 0, decimal=0)
#     # f, fs = ni.plotfft(y, 100, doplot=False)
#     # ny.percentile(fs, 50)
#     # fs_sorted = np.sort(fs)
#     # f_med = f[fs.tolist().index(fs_sorted[len(fs) // 2])]
#
#
# def fondamentals(frames0, samplerate):
#     mid = 16
#     sample = mid*2+1
#     res = []
#     for first in xrange(sample):
#         last = first-sample
#         frames = frames0[first:last]
#         res.append(_fondamentals(frames, samplerate))
#     res = sorted(res)
#     return res[mid] # We use the medium value
#
#
# def _fondamentals(frames, samplerate):
#     frames2=frames*hamming(len(frames));
#     frameSize=len(frames);
#     ceps=np.fft.ifft(np.log(np.abs(fft(frames2))))
#     nceps=ceps.shape[-1]*2/3
#     peaks = []
#     k=3
#     while(k < nceps - 1):
#         y1 = (ceps[k - 1])
#         y2 = (ceps[k])
#         y3 = (ceps[k + 1])
#         if (y2 > y1 and y2 >= y3): peaks.append([float(samplerate)/(k+2),abs(y2), k, nceps])
#         k=k+1
#     maxi=max(peaks, key=lambda x: x[1])
#     return maxi[0]
#
#
# def test_fund_fre():
#     f = 20
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x)/x[-1]
#     y = np.sin(2 * np.pi * f * x / Fs)
#     # call fondamentals(y, Fs)
#     np.testing.assert_almost_equal(fundamental_frequency(y, Fs), 0, decimal=0)
#
# def test_hist():
#     x = np.ones(10)
#     np.testing.assert_almost_equal(hist_json(x, 10, 5), (0, 0, 0, 0, 0, 0, 10, 0, 0, 0), decimal=0)
#
#
# def test_power_spec():
#     f = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f * x / Fs)
#     # max(plt.psd(y/np.std(y), int(Fs))[0])
#     np.testing.assert_almost_equal(power_spectrum(y, Fs), 32.99582867944011, decimal=5)
#
#
# def test_total_energy():
#     f = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f * x / Fs)
#     time = compute_time(y, Fs)
#     # sum(abs(y)**2.0)/(time[-1]-time[0])
#     np.testing.assert_almost_equal(total_energy_(y, Fs), 50.001500029605715, decimal=5)
#     y = [0,0,0,0,0,0,0]
#     np.testing.assert_equal(total_energy_(y, Fs), 0)
#
#
# def centroid(spectrum, f):
#     s = spectrum / np.sum(spectrum)
#     return np.dot(s, f)
#
#
# def test_spectral_centroid():
#     f1 = 0.1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f1 * x / Fs)
#     f, ff = ni.plotfft(y, Fs)
#     # centroid(ff, f)
#     np.testing.assert_almost_equal(spectral_centroid(y, Fs), 0.0021658004162115937, decimal=2)
#
#
# def test_spectral_spread():
#     f1 = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f1 * x / Fs)
#     f, ff = ni.plotfft(y, Fs)
#     p = ff / np.sum(ff)
#     # np.dot(((f-np.mean(centroid(ff, f)))**2),p)
#     np.testing.assert_almost_equal(spectral_spread(y, Fs), 0.31264360946306424, decimal=5)
#
#
# def test_spectral_skewness():
#     f1 = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f1 * x / Fs)
#     np.testing.assert_almost_equal(spectral_skewness(y, Fs), 34.789505786824407, decimal=5)
#     np.random.seed(seed=23)
#     x = np.random.normal(0, 2, 1000000)
#     np.testing.assert_almost_equal(spectral_skewness(x, Fs), 0, decimal=2)
#
#
# def _kurtosis(spectrum, f, spr, c):
#     s = spectrum / np.sum(spectrum)
#     return np.dot(s, (f - c)**4) / spr**2
#
#
# def test_spectral_kurtosis():
#     # np.random.seed(seed=1)
#     # x = np.random.normal(0, 2, 10000)
#     # np.testing.assert_almost_equal(spectral_kurtosis(x, 100), 0, decimal=3)
#     # mu, sigma = 20, 0.1
#     # x = mu + sigma * np.random.randn(100)
#     #_kurtosis(ff,f,sp,c)
#     f1 = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f1 * x / Fs)
#     f, ff = ni.plotfft(y, Fs)
#     sp = spectral_spread(y, Fs)
#     c = spectral_centroid(y, Fs)
#     np.testing.assert_almost_equal(spectral_kurtosis(y, Fs), 4293.9932884381524, decimal=0)
#
#
# def slope(ff, f):
#     c = np.vstack([f, np.ones(len(f))]).T
#     return np.linalg.lstsq(c, ff)[0][0]
#
#
# def test_spectral_slope():
#     f1 = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f1 * x / Fs)
#     f, ff = ni.plotfft(y, Fs)
#     # slope(ff, f)
#     np.testing.assert_almost_equal(spectral_slope(y, Fs), -0.1201628830466239, decimal=5)
#
#
# def test_spectral_decrease():
#     f1 = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f1 * x / Fs)
#     np.testing.assert_almost_equal(spectral_decrease(y, Fs), 0.12476839555000206, decimal=5)
#
#
# def roll_on(spectrum, f):
#     sqr = np.square(spectrum)
#     total = np.sum(sqr)
#     s = 0
#     output = 0
#     for i in range(0, len(f)):
#         s += sqr[i]
#         if s >= 0.05 * total:
#             output = f[i]
#             break
#     return output
#
#
# def test_spectral_roll_on():
#     f1 = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f1 * x / Fs)
#     f, ff = ni.plotfft(y, Fs)
#     #roll_on(ff, f)
#     np.testing.assert_almost_equal(spectral_roll_on(y, Fs), 0.010000300007000151, decimal=5)
#
#
# def roll_off(spectrum, f):
#     sqr = np.square(spectrum)
#     total = np.sum(sqr)
#     s = 0
#     output = 0
#     for i in range(0, len(f)):
#         s += sqr[i]
#         if s >= 0.95 * total:
#             output = f[i]
#             break
#     return output
#
#
# def test_spectral_roll_off():
#     f1 = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f1 * x / Fs)
#     f, ff = ni.plotfft(y, Fs)
#     #roll_off(ff, f)
#     np.testing.assert_almost_equal(spectral_roll_off(y, Fs), 0.010000300007000151, decimal=5)
#
#
# def test_spectral_curve_distance():
#     f1 = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f1 * x / Fs)
#     f, ff = ni.plotfft(y, Fs)
#     np.testing.assert_almost_equal(curve_distance(y, Fs), -1251684201.9742942, decimal=5)
#
#
# def test_spect_variation():
#     f1 = 1
#     sample = 1000
#     x = np.arange(0, sample, 0.01)
#     Fs = len(x) / x[-1]
#     y = np.sin(2 * np.pi * f1 * x / Fs)
#     f, ff = ni.plotfft(y, Fs)
#     np.testing.assert_almost_equal(spect_variation(y, Fs), 0.9999999999957343, decimal=5)


if __name__ == "__main__":

    run_module_suite()

