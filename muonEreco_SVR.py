import sys 
import time

import tables
from tables import *
#import numpy
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import AdaBoostRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation

from scipy.optimize import curve_fit

# Same as BDT Ereco but swapping in support vector regressors instead.  

# Functional form for the fit of Truncated Energy to True Energy
def funcTE(x, a1, b1, c1, e1, a2, b2, e2, a3, b3, c3): # takes Truncated Energy as 'x'
    return np.piecewise(x, [x<e1, x>=e2], [lambda x:a1 + b1*x + c1*np.power(x,2), lambda x:a3 + b3*x + c3*np.power(x,2), lambda x:np.log10(1./b2*np.subtract(np.power(10,x),a2))])

# Fitting wrapper function
def fit(Xarr,Yarr,pguess):  # takes Truncated Energy array as 'Xarr' and True Muon Energy array as 'Yarr'
        popt, pcov = curve_fit(funcTE, Xarr, Yarr,p0=pguess)
        print(popt)
        return popt

#lets open files in a a loop from commandline
import sys

if __name__ == '__main__':
    
    #adding m
    print "Number of simulation files: ", len(sys.argv)
    if len(sys.argv) == 0:
        print "Error! No files entered"
        sys.exit(0)

    start_time = time.time()
        
    # Set the random state
    rng = np.random.RandomState(1)
    
    #open files -- Edited
    simFiles = sys.argv[1:]
    #print simFiles

    # Create empty placeholder arrays
    energies = []
    muonEnergies = []#7-12 muon energies added
    nchans = []
    cogzs = []
    weights = []
    zens = []
    TEs = []
    zTravels = []
    nDirs = []
    qTots = []
    qDirs = []
    qEarlys = []
    qLates = []
    logls = []
    nDofs = []
    statuses = []

    # Loop over sim files and extract useful data
    for filename in simFiles:
        file  = open_file(filename, mode = "r")
        weights = np.append(weights, file.root.I3MCWeightDict.cols.OneWeight[:])
        energies = np.append(energies, file.root.I3MCPrimary.cols.energy[:])
        muonEnergies = np.append(muonEnergies, file.root.MCMuonEnergy.cols.Entry[:])#7-12 muon energies added
        nchans = np.append(nchans, file.root.OnlineL2_HitMultiplicityValues.cols.n_hit_doms[:])
        cogzs = np.append(cogzs, file.root.OnlineL2_HitStatisticsValues.cols.cog_z[:])
        zens = np.append(zens, file.root.OnlineL2_SplineMPE.cols.zenith[:])
        TEs =  np.append(TEs, file.root.OnlineL2_BestFit_TruncatedEnergy_AllDOMS_Muon.cols.energy[:])

        zTravels = np.append(zTravels, file.root.OnlineL2_HitStatisticsValues.cols.z_travel[:])
        nDirs = np.append(nDirs, file.root.OnlineL2_MPEFitDirectHitsC.cols.n_dir_doms[:])
        qTots = np.append(qTots, file.root.OnlineL2_HitStatisticsValues.cols.q_tot_pulses[:])
        qDirs = np.append(qDirs, file.root.OnlineL2_MPEFitDirectHitsC.cols.q_dir_pulses[:])
        qEarlys = np.append(qEarlys, file.root.OnlineL2_MPEFitDirectHitsC.cols.q_early_pulses[:])
        qLates = np.append(qLates, file.root.OnlineL2_MPEFitDirectHitsC.cols.q_late_pulses[:])
        logls = np.append(logls, file.root.OnlineL2_MPEFitFitParams.cols.logl[:])
        nDofs = np.append(nDofs, file.root.OnlineL2_MPEFitFitParams.cols.ndof[:])
        statuses = np.append(statuses, file.root.OnlineL2_MPEFit.cols.fit_status[:])

        file.close()

    # Trim all arrays for any cuts and make logs
    cuts = (muonEnergies > 0) &  (TEs > 0) & (qTots>0) & (statuses==0)
    weights = weights[cuts]
    energies = energies[cuts]
    muonEnergies = muonEnergies[cuts] #7-12 muon energies added

    nchans = nchans[cuts]
    cogzs = cogzs[cuts]
    zens = zens[cuts]
    TEs = TEs[cuts]

    zTravels = zTravels[cuts]
    nDirs = nDirs[cuts]
    qDirs = qDirs[cuts]
    qTots = qTots[cuts]
    qEarlys = qEarlys[cuts]
    qLates = qLates[cuts]
    logls = logls[cuts]
    nDofs = nDofs[cuts]
    statuses = statuses[cuts]

    print("Number of events before cuts: %d"%len(cuts))
    print("Number of events after  cuts: %d"%len(energies))

    # Other post-processing requirements to transform raw variables from files into something more useful
    lTEs = np.log10(TEs)
    lenergies = np.log10(energies)
    lmuonEnergies = np.log10(muonEnergies)
    zTravels = np.absolute(zTravels)
    qEarlysFrac = qEarlys/qTots
    qLatesFrac = qLates/qTots
    pLogls = logls/(nDofs+2.5)

    # Make events E^-1 weighted
    eweight = np.power(energies,-1)
    weights = weights*eweight

    #EXPERIMENTING -- DIFF CONFIGS FOR X - 7/8/16
    #X = nchans[:, np.newaxis]
    #X = lTEs[:, np.newaxis]

    #X = zip(nchans,cogzs,zens,lTEs,zTravels,nDirs,qDirs,qTots,qEarlysFrac,qLatesFrac,pLogls)
    #X = zip(nchans,cogzs,lTEs,zTravels)
    X = zip(lTEs)
    #y = lenergies
    y = lmuonEnergies #7-12 muon energies added
    #regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10,splitter='best',max_features=1,\
    #                                               min_samples_split=100),n_estimators=100, random_state=rng)
    #regr = SVR(C=20.0, epsilon=2.5)
    regr = SVR(C=15.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
               kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
                                    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, \
                                                                         train_size=0.6, random_state=1)
    weights_train, weights_test, y_train, y_test = cross_validation.train_test_split(weights, y, \
                                                                         train_size=0.6, random_state=1)
    '''
    # Useful if you want to test whole data set with no train/test separation
    X_train = X
    X_test = X
    y_train = y
    y_test = y
    weights_train = weights
    weights_test = weights
    '''
    
    #regr.fit(X[0:sliceIdx], y, sample_weight=weights)#regr.fit(X, y, sample_weight=weights)
    #sklearn.cross_validation.train_test_split
    regr.fit(X_train, y_train, sample_weight=weights_train)
    #regr.fit(X_train, y_train)

    # Predict
    y_predict_train = regr.predict(X_train)
    y_predict_test = regr.predict(X_test)

    print "The script took", time.time() - start_time, "seconds to run"

    # Plot the results wrt one of the input variables
    inputVar = 0  # E.g. 3 = lTE.  These are in the order above in the definition of X
    plt.figure(1)
    plt.hist2d(np.array(X_test)[:,inputVar], y_test, bins=40, weights=weights_test, norm=LogNorm())
    plt.colorbar()
    plt.scatter(np.array(X_test)[:,inputVar], y_predict_test, c="k", label="prediction", s=1.0)
    # Fit the T.E. vs True Energy to establish baseline Energy resolution comparison
    opts = fit(np.array(X_test)[:,inputVar],y_test,[1,1,1,3,1,1,6,1,1,1])
    Xplotting = np.arange(2,8,0.1)
    plt.plot(np.array(Xplotting), funcTE(Xplotting, *opts), 'r-',linewidth=3)
    plt.xlabel("log [TE/(dE/dx)]")
    plt.ylabel("log [Energy / GeV]")
    plt.title("Test Sample")
    #plt.legend()
    #plt.show()

    plt.figure(2)
    plt.hist2d(np.array(X_train)[:,inputVar], y_train, bins=40, weights=weights_train, norm=LogNorm())
    plt.colorbar()
    plt.scatter(np.array(X_train)[:,inputVar], y_predict_train, c="k", label="prediction", s=1.0)
    plt.xlabel("log [TE/(dE/dx)]")
    plt.ylabel("log [Energy / GeV]")
    plt.title("Training Sample")
    #plt.title("Boosted Decision Tree Regression")
    #plt.legend()
    #plt.show()
    
    # T.E. Energy Resolution Calculation:
    EdiffTE= y_test - funcTE(np.array(X_test)[:,inputVar], *opts)
    print("The standard deviation of the Truncated Energy resolution is: %f"%np.std(EdiffTE))

    # Relative Energy Resolution Calculation:
    Ediff_test =  y_test - y_predict_test
    Ediff_train =  y_train - y_predict_train
    #relEres = Ediff/y
    print("The standard deviation of the Test Energy resolution is          : %f"%np.std(Ediff_test))
    print("The standard deviation of the Train Energy resolution is          : %f"%np.std(Ediff_train))

    plt.figure(3)
    plt.yscale('log')
    plt.hist(EdiffTE,bins=50,log=True,color='r',histtype='step',weights=weights_test,label='T.E.')
    plt.hist(Ediff_test,bins=50,log=True,color='b',histtype='step',weights=weights_test,label='BDT_test')
    plt.hist(Ediff_train,bins=50,log=True,color='k',histtype='step',weights=weights_train,label='BDT_train')
    plt.xlabel("log10(Predicted Energy / True Energy)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.yscale('log')
    
    plt.show()
        


