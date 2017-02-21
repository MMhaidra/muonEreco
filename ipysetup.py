import sys
import glob

# Just a little helper script for running things interactively.

sys.argv = glob.glob("sims/11069/Level2_nugen_numu_IC86.2012.011069.0040??_gfu.h5")
execfile("muonEreco_BDT.py")
#execfile("muonEreco_KRR.py")

