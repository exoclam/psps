##################################################
####### For plotting, etc ########################
##################################################

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
matplotlib.rcParams.update({'errorbar.capsize': 1})
pylab_params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(pylab_params)

import numpy as np
import pandas as pd

path = '/Users/chrislam/Desktop/psps/' 

def plot_properties(teffs, ages):
    """
    Make 2-subplot figure showing distributions of Teff and age. Tentatively Figs 1 & 2, in Paper III

    Input: 
    - 

    """

    ### VISUALIZE TRILEGAL SAMPLE PROPERTIES, FOR PAPER FIGURE
    teff_hist, teff_bin_edges = np.histogram(teffs, bins=50)
    print("Teff peak: ", teff_bin_edges[np.argmax(teff_hist)])
    age_hist, age_bin_edges = np.histogram(ages, bins=50)
    print("age peak: ", age_bin_edges[np.argmax(age_hist)])

    #fig, axes = plt.subplots(figsize=(7,5))
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7, 5))

    #ax1 = plt.subplot2grid((2,1), (0,0))
    ax1.hist(teffs, bins=50, alpha=0.7)
    ax1.set_ylabel("count")
    ax1.set_xlabel(r"$T_{eff}$ [K]")
    # plot vertical red line through median Teff
    ax1.plot([np.median(teffs), np.median(teffs)], 
            [0,4000], color='r', alpha=0.3, linestyle='--', label=r'median $T_{eff}$')
    #ax1.set_xlim([4800, 7550])
    ax1.legend()

    #ax2 = plt.subplot2grid((2,1), (1,0))
    ax2.hist(ages, bins=50, alpha=0.7)
    # plot vertical red line through median age 
    ax2.plot([np.median(ages), np.median(ages)], 
            [0,3600], color='r', alpha=0.3, linestyle='--', label='median age')
    ax2.set_ylabel("count")
    ax2.set_xlabel("age [Gyr]")
    #ax2.set_xlim([0, 18])
    ax2.legend()
    fig.tight_layout()

    print("median Teff: ", np.median(teffs))
    print("median age: ", np.median(ages))

    plt.savefig(path+'plots/sample_properties_trilegal_heights_only.pdf', format='pdf')
    plt.show()

    return