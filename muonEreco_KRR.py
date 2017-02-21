#File Edit Options Buffers Tools Python Help                                     
import sys 
import time

import tables
from tables import *
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation

from scipy.optimize import curve_fit

from sklearn.kernel_ridge import KernelRidge

# Same as muonEreco_BDT.py but for a Kernel Ridge estimator.

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
    runIDs = []
    evIDs = []
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
    #bdtScoreUp = []  #new added cut

    # Loop over sim files and extract useful data
    for filename in simFiles:
        file  = open_file(filename, mode = "r")
        weights = np.append(weights, file.root.I3MCWeightDict.cols.OneWeight[:])
        energies = np.append(energies, file.root.I3MCPrimary.cols.energy[:])
        muonEnergies = np.append(muonEnergies, file.root.MCMuonEnergy.cols.Entry[:])#7-12 muon energies added
        runIDs = np.append(runIDs, file.root.I3EventHeader.cols.Run[:])
        evIDs = np.append(evIDs, file.root.I3EventHeader.cols.Event[:])
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

        #bdtScoreUp = np.append(bdtScoreUp, file.root.GFU_BDT_Score_Up.cols.value[:]) #new added cut

        file.close()

    # Start by 

    # Trim all arrays for any cuts and make logs
    cuts = (muonEnergies > 0) &  (TEs > 0) & (qTots>0) & (statuses==0) #&(zens>math.radians(90)) & (bdtScoreUp > 0.1)
    weights = weights[cuts]
    energies = energies[cuts]
    muonEnergies = muonEnergies[cuts] #7-12 muon energies added
    runIDs = runIDs[cuts]
    evIDs = evIDs[cuts]

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

    inputVar = 3  # For plotting the 1-d projection at the end, must match an index of X usually for TE
    #X = zip(lTEs)
    #X = zip(nchans,cogzs,zens,lTEs,zTravels,nDirs,qDirs,qTots,qEarlysFrac,qLatesFrac,pLogls)#all
    #X = zip(cogzs,zens,lTEs,zTravels,nDirs,qDirs,qTots,qEarlysFrac,qLatesFrac,pLogls)#nchans rm
    #X = zip(nchans,zens,lTEs,zTravels,nDirs,qDirs,qTots,qEarlysFrac,qLatesFrac,pLogls)#cozs rm
    #X = zip(nchans,cogzs,lTEs,zTravels,nDirs,qDirs,qTots,qEarlysFrac,qLatesFrac,pLogls)#zens rm
    #X = zip(nchans,cogzs,zens,zTravels,nDirs,qDirs,qTots,qEarlysFrac,qLatesFrac,pLogls)#lTEs rm
    #X = zip(nchans,cogzs,zens,lTEs,nDirs,qDirs,qTots,qEarlysFrac,qLatesFrac,pLogls)#zTravels rm
    #X = zip(nchans,cogzs,zens,lTEs,zTravels,qDirs,qTots,qEarlysFrac,qLatesFrac,pLogls)#nDirs rm
    #X = zip(nchans,cogzs,zens,lTEs,zTravels,nDirs,qTots,qEarlysFrac,qLatesFrac,pLogls)#qDirs rm
    #X = zip(nchans,cogzs,zens,lTEs,zTravels,nDirs,qDirs,qEarlysFrac,qLatesFrac,pLogls)#qTots rm
    #X = zip(nchans,cogzs,zens,lTEs,zTravels,nDirs,qDirs,qTots,qLatesFrac,pLogls)#qEarlysFrac rm
    #X = zip(nchans,cogzs,zens,lTEs,zTravels,nDirs,qDirs,qTots,qEarlysFrac,pLogls)#qLatesFrac rm
    #X = zip(nchans,cogzs,zens,lTEs,zTravels,nDirs,qDirs,qTots,qEarlysFrac)#rm pLogls
    X = zip(nchans,cogzs,zens,lTEs,zTravels,qEarlysFrac,qTots)#all added qTots
    #X = zip(cogzs,zens,lTEs,zTravels,qEarlysFrac)#rm nchans
    #X = zip(nchans,zens,lTEs,zTravels,qEarlysFrac)#rm cogzs
    #X = zip(nchans,cogzs,lTEs,zTravels,qEarlysFrac)#rm zens
    #X = zip(nchans,cogzs,zens,zTravels,qEarlysFrac)#rm lTES
    #X = zip(nchans,cogzs,zens,lTEs,qEarlysFrac)#rm zTravels
    #X = zip(nchans,cogzs,zens,lTEs,zTravels)#rm qEarlysFrac
    #y = lenergies
    y = lmuonEnergies #7-12 muon energies added
    #regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),n_estimators=100, random_state=rng)
    regr = KernelRidge(alpha=1.0,kernel='polynomial',degree=3)

    #regr = DecisionTreeRegressor(max_depth=5)
                                    
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
    
    # Fit the T.E. vs True Energy to establish baseline Energy resolution cut
    opts_train = fit(np.array(X_train)[:,inputVar],y_train,[1,1,1,3,1,1,6,1,1,1])
    # T.E. Energy Resolution Calculation for training sample:
    EdiffTE_train= y_train - funcTE(np.array(X_train)[:,inputVar], *opts_train)

    EresCutValue = 1.0  # Difference in logE between true and predicted to cut on

    cuts2 = np.absolute(EdiffTE_train) < EresCutValue
    print("The standard deviation of the Truncated Energy training sample resolution before cleaning: %f"%np.std(EdiffTE_train))
    print("The standard deviation of the Truncated Energy training sample resolution after  cleaning: %f"%np.std(EdiffTE_train[cuts2]))
    regr.fit(np.array(X_train)[cuts2], y_train[cuts2], sample_weight=weights_train[cuts2])

    # Predict
    y_predict_train = regr.predict(X_train)
    y_predict_test = regr.predict(X_test)

    print "multFilesReg  took", time.time() - start_time, "seconds to run"
    
    # Plot the results wrt one of the input variables
    
    plt.figure(1)
    plt.hist2d(np.array(X_test)[:,inputVar], y_test, bins=40, weights=weights_test, norm=LogNorm())
    plt.colorbar()
    plt.scatter(np.array(X_test)[:,inputVar], y_predict_test, c="k", label="prediction", s=1.0)
    
    # Fit the T.E. vs True Energy to establish baseline Energy resolution comparison
    opts = fit(np.array(X_test)[:,inputVar],y_test,[1,1,1,3,1,1,6,1,1,1])
    Xplotting = np.arange(2,8,0.1)
    plt.plot(np.array(Xplotting), funcTE(Xplotting, *opts), 'r-',linewidth=3)
    plt.xlabel("log10 [TE/(dE/dx)]")
    plt.ylabel("log10 [Energy / GeV]")
    plt.title("Test Sample")
    #plt.legend()
    #plt.show()

    plt.figure(2)
    plt.hist2d(np.array(X_train)[:,inputVar], y_train, bins=40, weights=weights_train, norm=LogNorm())
    plt.colorbar()
    plt.scatter(np.array(X_train)[:,inputVar], y_predict_train, c="k", label="prediction", s=1.0)
    plt.xlabel("log10 [TE/(dE/dx)]")
    plt.ylabel("log10 [Energy / GeV]")
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
    print("The standard deviation of the BDT Test Energy resolution is          : %f"%np.std(Ediff_test))
    print("The standard deviation of the BDT Train Energy resolution is          : %f"%np.std(Ediff_train))
    
    
    plt.figure(3)
    plt.yscale('log')
    plt.hist(EdiffTE,bins=50,log=True,color='r',histtype='step',weights=weights_test,label='T.E.')
    plt.hist(Ediff_test,bins=50,log=True,color='b',histtype='step',weights=weights_test,label='BDT_test')
    plt.hist(Ediff_train,bins=50,log=True,color='k',histtype='step',weights=weights_train,label='BDT_train')
    plt.xlabel("log10[True Energy / Predicted Energy]")
    plt.ylabel("nEvents")
    plt.legend()
    plt.yscale('log')
    
    plt.figure(4)
    plt.hist2d(y_predict_test, y_test, bins=40, weights=weights_test, norm=LogNorm())
    plt.xlabel("log10[BDT Predicted Energy]")
    plt.ylabel("log10[True Energy]")
    plt.colorbar()

    plt.figure(5)
    plt.hist2d(funcTE(np.array(X_test)[:,inputVar], *opts), y_test, bins=40, weights=weights_test, norm=LogNorm())
    plt.title("Training Sample")
    plt.xlabel("log10[TE Predicted Energy]")
    plt.ylabel("log10[True Energy]")
    plt.colorbar()

    showPlots = True
    if showPlots: 
        plt.show()



"""
from sklearn.kernel_ridge import KernelRidge
import numpy as np
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = KernelRidge(alpha=1.0)
clf.fit(X, y) 
KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='linear',
            kernel_params=None)
"""
