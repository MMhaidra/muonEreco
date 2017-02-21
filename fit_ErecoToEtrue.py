from scipy.optimize import curve_fit
import numpy as np 
import math

# For calibrating raw Truncated Energy output to True Muon Energy (at detector center).

def func(x, a, b): # takes Truncated Energy as 'x'
    return np.log10(1./b*np.subtract(x,a))

def fit(Xarr,Yarr):  # takes Truncated Energy as 'x' and True Muon Energy as 'y'
        #y_f =  func(x, 3, 1)
        popt, pcov = curve_fit(func, Xarr, Yarr)
        print(popt)
        return popt
