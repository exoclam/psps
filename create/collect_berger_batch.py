##########################################
# MAKE STARS AND PLANETS FROM BERGER+ 2020
###########################################

import os
import os.path
import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform, gamma
from math import lgamma
import jax
import jax.numpy as jnp
from tqdm import tqdm
from ast import literal_eval
import seaborn as sns
from glob import glob
from tqdm import tqdm
import numpyro
from numpyro import distributions as dist, infer
import numpyro_ext
import arviz as az

from itertools import zip_longest
import numpy.ma as ma # for masked arrays

from astropy.table import Table, join

from psps.transit_class import Population, Star
import psps.simulate_helpers as simulate_helpers
import psps.simulate_transit as simulate_transit
import psps.utils as utils

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

import warnings
warnings.filterwarnings("ignore")

path = '/Users/chrislam/Desktop/psps/' 

# we're gonna need this for reading in the initial Berger+ 2020 data
def literal_eval_w_exceptions(x):
    try:
        return literal_eval(str(x))   
    except Exception as e:
        pass

# planet formation history model parameters
"""
threshold = 11.5 # cosmic age in Gyr; 13.7 minus stellar age, then round; f = 0.31
frac1 = 0.2 # frac1 must be < frac2 if comparing cosmic ages
frac2 = 0.85

threshold = 9.5 # f = 0.45; not using
frac1 = 0.25
frac2 = 0.75

threshold = 9.5 # f = 0.32
frac1 = 0.1
frac2 = 0.65

threshold = 7.5 # f = 0.35, 0.36
frac1 = 0.01
frac2 = 0.6

threshold = 7.5 # f = 0.30; 0.36; 0.34 (final: 0.32); 0.31 (final: 0.29); 0.27
frac1 = 0.01; 0.05; 0.05; 0.05; 0.05
frac2 = 0.5; 0.55; 0.5; 0.45; 0.4

threshold = 5.5 # f = 0.30; 0.28; 0.33 (final: 0.31); 0.30 (final: 0.29)
frac1 = 0.01; 0.05; 0.05; 0.05
frac2 = 0.4; 0.35; 0.4; 0.37

threshold = 5 # f = 0.29
frac1 = 0.05
frac2 = 0.35

threshold = 12. # f = 0.28
frac1 = 0.2
frac2 = 0.9

threshold = 12. # f = 0.29
frac1 = 0.25
frac2 = 0.9

threshold = 11. # f = 0.30
frac1 = 0.15
frac2 = 0.8

threshold = 10. # f = 0.27
frac1 = 0.05
frac2 = 0.7

step (3.5, 1, 35): f=0.30 (low Z too low)

piecewise1: f=0.30
piecewise2: f=0.30
piecewise3: f=0.31
piecewise4: f=0.33 (dropped f2 a smidge) --> f=0.29
piecewise (3.5, 1, 60): f=too small; (3.5, 1, 60): f=0.35; (3.5, 1, 70): f=0.33 (tau=-0.15 +/- 0.11; high Z too high)
"""

# operative parameters
threshold = 11
frac1 = 0.15
frac2 = 0.80

name_thresh = 11
name_f1 = 15
name_f2 = 80
name = 'step_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2)
#name = 'monotonic_'+str(name_f1)+'_'+str(name_f2) # f=
#name = 'piecewise_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2) # f=0.31, f=0.30, f=

# new, one-time completeness map
#completeness_map = pd.read_csv(path+'data/completeness_map.csv') 
#completeness_map = pd.read_csv(path+'data/completeness_map_empirical'+name+'.csv') 
#completeness_map_np = completeness_map.to_numpy()
#completeness_map_np = np.flip(completeness_map_np, axis=0)

#"""
### diagnostic plotting age vs height
sim = sorted(glob(path+'data/berger_gala/' + name + '/' + name + '*'))#[:5]

heights = []
ages = []
fs = []
physical_planet_occurrences = []
physical_planet_occurrences_precut = []
detected_planet_occurrences_all = []
adjusted_planet_occurrences_all = []
transit_multiplicities_all = []
geom_transit_multiplicities_all = []
completeness_all = []

period_grid = np.logspace(np.log10(2), np.log10(300), 10)
radius_grid = np.linspace(1, 4, 10)
#height_bins = np.array([0., 150, 250, 400, 650, 3000]) 
height_bins = np.array([0., 120, 200, 300, 500, 800, 1500]) # the actual Zink Fig 12 height bins
height_bins = np.logspace(2, 3, 6) # ah, so the above are the midpoints of the actual bins they used, I guess
height_bin_midpoints = 0.5 * (np.logspace(2,3,6)[1:] + np.logspace(2,3,6)[:-1])
for i in tqdm(range(len(sim))):
    print(sim[i])
    try:
        berger_kepler_all = pd.read_csv(sim[i], sep=',') #, on_bad_lines='skip'
    except:
        continue
    #berger_kepler_all = pd.read_csv(path+'data/berger_gala/'+name+'.csv')

    num_hosts = berger_kepler_all.loc[berger_kepler_all['num_planets']>0]
    #print("f: ", len(num_hosts)/len(berger_kepler_all))
    f = len(num_hosts)/len(berger_kepler_all)
    fs.append(f)

    berger_kepler_all = berger_kepler_all.dropna(subset=['height'])
    berger_kepler_all['height'] = berger_kepler_all['height'] * 1000 # to pc
    berger_kepler_all['periods'] = berger_kepler_all['periods'].apply(literal_eval_w_exceptions)
    berger_kepler_all['planet_radii'] = berger_kepler_all['planet_radii'].apply(literal_eval_w_exceptions)
    berger_kepler_all['incls'] = berger_kepler_all['incls'].apply(literal_eval_w_exceptions)
    berger_kepler_all['mutual_incls'] = berger_kepler_all['mutual_incls'].apply(literal_eval_w_exceptions)
    berger_kepler_all['eccs'] = berger_kepler_all['eccs'].apply(literal_eval_w_exceptions)
    berger_kepler_all['omegas'] = berger_kepler_all['omegas'].apply(literal_eval_w_exceptions)

    berger_kepler_all = berger_kepler_all.loc[(berger_kepler_all['height'] <= 1000) & (berger_kepler_all['age'] <= 13.7)] 
    print("FINAL SAMPLE COUNT: ", len(berger_kepler_all))

    #utils.plot_properties(berger_kepler_all['iso_teff'], berger_kepler_all['iso_age'])
    heights.append(np.array(berger_kepler_all['height']))
    ages.append(np.array(berger_kepler_all['age']))

    """
    print(berger_kepler_all['height'])
    print(height_bins)
    berger_kepler_all['height_bins'] = pd.cut(berger_kepler_all['height'], bins=height_bins, include_lowest=True)
    print(berger_kepler_all['height_bins'])
    berger_kepler_counts = np.array(berger_kepler_all.groupby(['height_bins']).count().reset_index()['kepid'])
    print(berger_kepler_counts)
    berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all['num_planets'] > 0]
    print(berger_kepler_planets)
    berger_kepler_planets = berger_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas']).reset_index(drop=True)
    print(berger_kepler_planets)
    print(np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid']))
    quit()
    """

    # RESULT PLOT STUFF
    berger_kepler_all['height_bins'] = pd.cut(berger_kepler_all['height'], bins=height_bins, include_lowest=True)
    berger_kepler_counts = np.array(berger_kepler_all.groupby(['height_bins']).count().reset_index()['kepid'])

    # isolate planet hosts and bin them by galactic height
    berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all['num_planets'] > 0]
    berger_kepler_planets = berger_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas']).reset_index(drop=True)
    #print("data frame: ", berger_kepler_planets)
    #print(np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid']))
    #berger_kepler_planets_counts_precut = np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid'])

    berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
    berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 4.] # limit radii to fairly compare with SEs in Zink+ 2023 (2)...or how about include SNs too (4)?
    berger_kepler_planets_counts = np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid'])

    physical_planet_occurrence = 100 * berger_kepler_planets_counts/berger_kepler_counts 
    physical_planet_occurrences.append(physical_planet_occurrence)

    detected_planet_occurrences = []
    adjusted_planet_occurrences = []
    transit_multiplicities = []
    geom_transit_multiplicities = []
    piv_physicals = []
    piv_detecteds = []

    #completeness_all = []
    for k in range(30): 

        #berger_kepler_planets_temp = berger_kepler_planets

        ### Simulate detections from these synthetic systems
        prob_detections, transit_statuses, sn, geom_transit_statuses = simulate_transit.calculate_transit_vectorized(berger_kepler_planets.periods, 
                                        berger_kepler_planets.stellar_radius, berger_kepler_planets.planet_radii,
                                        berger_kepler_planets.eccs, 
                                        berger_kepler_planets.incls, 
                                        berger_kepler_planets.omegas, berger_kepler_planets.stellar_mass,
                                        berger_kepler_planets.rrmscdpp06p0, angle_flag=True) 

        berger_kepler_planets['transit_status'] = transit_statuses[0]
        berger_kepler_planets['prob_detections'] = prob_detections[0]
        berger_kepler_planets['sn'] = sn
        berger_kepler_planets['geom_transit_status'] = geom_transit_statuses

        # need kepid to be str or tuple, else unhashable type when groupby.count()
        berger_kepler_planets['kepid'] = berger_kepler_planets['kepid'].apply(str) 

        # plot color-coded visualization
        #utils.plot_host_vs_height(berger_kepler_all, berger_kepler_planets)
        utils.plot_age_vs_height(berger_kepler_all)
        quit()

        # isolate detected transiting planets
        berger_kepler_transiters = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]
        geom = berger_kepler_planets.loc[berger_kepler_planets['geom_transit_status']==1]
        non_geom = berger_kepler_planets.loc[berger_kepler_planets['geom_transit_status']!=1]
        #print(berger_kepler_planets.loc[((berger_kepler_planets['planet_radii'] > 3.667) & (berger_kepler_planets['planet_radii'] <= 4) & (berger_kepler_planets['periods'] > 3.49) & (berger_kepler_planets['periods'] <= 6.09))])
        #print(list(geom.columns))
        #print(geom.loc[((geom['planet_radii'] > 3.667) & (geom['planet_radii'] <= 4) & (geom['periods'] > 3.49) & (geom['periods'] <= 6.09))])
        #print(berger_kepler_transiters.loc[((berger_kepler_transiters['planet_radii'] > 3.667) & (berger_kepler_transiters['planet_radii'] <= 4) & (berger_kepler_transiters['periods'] > 3.49) & (berger_kepler_transiters['periods'] <= 6.09))])
        #
        #plt.hist(geom['stellar_radius'], alpha=0.3, label='geom')
        #plt.hist(non_geom['stellar_radius'], alpha=0.3, label='non geom')
        #plt.legend()
        #plt.show()
        #print("physical planets: ", len((berger_kepler_planets['kepid'])))
        #print("detected planets: ", len((berger_kepler_transiters['kepid'])))
        
        # read out detected yields
        #berger_kepler_transiters.to_csv(path+'galactic-occurrence/systems/berger_kepler_planets_detected_'+str(i)+'.csv', index=False)

        # Read in pre-generated population
        #transiters_berger_kepler = pd.read_csv(path+'galactic-occurrence/systems/berger_kepler_planets_detected_'+str(i)+'.csv')

        ### Completeness
        # Calculate completeness map
        #if (k==0) or (k==1) or (k==2) or (k==3) or (k==4):
        completeness_map, piv_physical, piv_detected = simulate_helpers.completeness(berger_kepler_planets, berger_kepler_transiters)
        completeness_threshold = 0.0025 # completeness threshold under which period/radius cell is not counted; 0.5% results in full recovery, but let's round up to 1%
        completeness_map = completeness_map.mask(completeness_map < completeness_threshold) # assert that completeness fractions lower than 1% are statistically insignificant
        completeness_all.append(completeness_map)
        #print(piv_physical)
        #print(piv_detected)
        #print(completeness_map)
        # why do some cells always have the same fraction? 

        #print("CHECK")
        #print(completeness_map_np)
        #print(completeness_map)

        ### this is to find the threshold beyond which I can fully recover the physical yield using the detected yield and completeness map
        #print("physical: ", simulate_helpers.adjust_for_completeness(berger_kepler_planets, completeness_map, radius_grid, period_grid, flag='physical'))
        #print("detected, adjusted: ", simulate_helpers.adjust_for_completeness(berger_kepler_transiters, completeness_map, radius_grid, period_grid, flag='detected'))

        ### Calculate transit multiplicity and other Population-wide demographics
        #simulate_helpers.collect_galactic(berger_kepler_planets)

        # compute transit multiplicity
        transit_multiplicity = berger_kepler_transiters.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
        transit_multiplicity = transit_multiplicity.to_list()
        transit_multiplicity += [0.] * (6 - len(transit_multiplicity)) # pad with zeros to match length of k
        transit_multiplicities_all.append(transit_multiplicity)

        # also calculate the geometric transit multiplicity
        geom_transiters_berger_kepler = berger_kepler_planets.loc[berger_kepler_planets['geom_transit_status']==1]
        geom_transit_multiplicity = geom_transiters_berger_kepler.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid
        geom_transit_multiplicity = geom_transit_multiplicity.to_list()
        geom_transit_multiplicity += [0.] * (6 - len(geom_transit_multiplicity)) # pad with zeros to match length of k
        geom_transit_multiplicities_all.append(geom_transit_multiplicity)

        # calculate detected occurrence rate 
        berger_kepler_transiters_counts = np.array(berger_kepler_transiters.groupby(['height_bins']).count().reset_index()['kepid'])
        detected_planet_occurrence = berger_kepler_transiters_counts/berger_kepler_counts
        detected_planet_occurrences_all.append(detected_planet_occurrence)

        # same, but adjust for period- and radius-dependent completeness 
        #berger_kepler_transiters1 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 0) & (berger_kepler_transiters['height'] <= 200)]
        #berger_kepler_transiters2 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 200) & (berger_kepler_transiters['height'] <= 400)]
        #berger_kepler_transiters3 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 400) & (berger_kepler_transiters['height'] <= 600)]
        #berger_kepler_transiters4 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 600) & (berger_kepler_transiters['height'] <= 800)]
        #berger_kepler_transiters5 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 800) & (berger_kepler_transiters['height'] <= 1000)]
        #print(len(berger_kepler_planets))
        #print(len(berger_kepler_transiters))
        #print(simulate_helpers.adjust_for_completeness(berger_kepler_transiters, completeness_map, radius_grid, period_grid))

        #berger_kepler_transiters1 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 0) & (berger_kepler_transiters['height'] <= 120)]
        #berger_kepler_transiters2 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 120) & (berger_kepler_transiters['height'] <= 200)]
        #berger_kepler_transiters3 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 200) & (berger_kepler_transiters['height'] <= 300)]
        #berger_kepler_transiters4 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 300) & (berger_kepler_transiters['height'] <= 500)]
        #berger_kepler_transiters5 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 500) & (berger_kepler_transiters['height'] <= 800)]
        #berger_kepler_transiters6 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 800) & (berger_kepler_transiters['height'] <= 1500)]

        #berger_kepler_transiters1 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 0) & (berger_kepler_transiters['height'] <= 150)]
        #berger_kepler_transiters2 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 150) & (berger_kepler_transiters['height'] <= 300)]
        #berger_kepler_transiters3 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 450) & (berger_kepler_transiters['height'] <= 600)]
        #berger_kepler_transiters4 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 600) & (berger_kepler_transiters['height'] <= 750)]
        #berger_kepler_transiters5 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 750) & (berger_kepler_transiters['height'] <= 900)]
        #berger_kepler_transiters6 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 900) & (berger_kepler_transiters['height'] <= 1050)]
        #berger_kepler_transiters7 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 1050) & (berger_kepler_transiters['height'] <= 1200)]
        #berger_kepler_transiters8 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 1200) & (berger_kepler_transiters['height'] <= 1350)]
        #berger_kepler_transiters9 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 1350) & (berger_kepler_transiters['height'] <= 1500)]

        berger_kepler_transiters1 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 100) & (berger_kepler_transiters['height'] <= np.logspace(2,3,6)[1])]
        berger_kepler_transiters2 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > np.logspace(2,3,6)[1]) & (berger_kepler_transiters['height'] <= np.logspace(2,3,6)[2])]
        berger_kepler_transiters3 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > np.logspace(2,3,6)[2]) & (berger_kepler_transiters['height'] <= np.logspace(2,3,6)[3])]
        berger_kepler_transiters4 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > np.logspace(2,3,6)[3]) & (berger_kepler_transiters['height'] <= np.logspace(2,3,6)[4])]
        berger_kepler_transiters5 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > np.logspace(2,3,6)[4]) & (berger_kepler_transiters['height'] <= 1000)]
        
        #len_berger_kepler_transiters, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters, completeness_map_np, radius_grid, period_grid) #completeness_map_np vs completeness_map
        #print(np.sum(np.mean(physical_planet_occurrences, axis=0)))
        #print(100 * np.sum(len_berger_kepler_transiters/berger_kepler_counts))
        
        #plt.errorbar(np.array(height_bins[:-1]), np.mean(physical_planet_occurrences, axis=0), np.std(physical_planet_occurrences, axis=0), fmt='o', alpha=0.5, capsize=3, elinewidth=1, markeredgewidth=1, label='true')
        #plt.scatter(np.array(height_bins[:-1]), len_berger_kepler_transiters/berger_kepler_counts, color='purple', label='recovered')
        #plt.xlabel(r"$Z_{max}$ [pc]")
        #plt.ylabel("planets per 100 stars")
        #plt.legend()
        #plt.show()
        #quit()

        #"""
        len_berger_kepler_transiters1, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters1, completeness_map, radius_grid, period_grid) #completeness_map_np vs completeness_map
        len_berger_kepler_transiters2, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters2, completeness_map, radius_grid, period_grid)
        len_berger_kepler_transiters3, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters3, completeness_map, radius_grid, period_grid)
        len_berger_kepler_transiters4, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters4, completeness_map, radius_grid, period_grid)
        len_berger_kepler_transiters5, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters5, completeness_map, radius_grid, period_grid)
        len_berger_kepler_transiters = np.array([len_berger_kepler_transiters1, len_berger_kepler_transiters2, len_berger_kepler_transiters3, len_berger_kepler_transiters4, len_berger_kepler_transiters5])

        len_berger_kepler_recovered, recovered_piv = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters, completeness_map, radius_grid, period_grid)
        """
        print("physical piv: ", piv_physical)
        print("detected piv: ", piv_detected)
        print("completeness map: ", completeness_map)
        print("recovered piv: ", recovered_piv)
        print("stars: ", berger_kepler_counts)
        print("number of physical planets; number of stars: ", len(berger_kepler_planets), np.sum(berger_kepler_counts))
        print("physical planet occurrence: ", 100 * len(berger_kepler_planets)/np.sum(berger_kepler_counts))
        print("recovered total vs physical total: ", 100 * len_berger_kepler_recovered/np.sum(berger_kepler_counts), np.sum(physical_planet_occurrence))
        print(100 * len_berger_kepler_transiters1/berger_kepler_counts[0], physical_planet_occurrence[0])
        print(100 * len_berger_kepler_transiters1/berger_kepler_counts[1], physical_planet_occurrence[1])
        print(100 * len_berger_kepler_transiters1/berger_kepler_counts[2], physical_planet_occurrence[2])
        print(100 * len_berger_kepler_transiters1/berger_kepler_counts[3], physical_planet_occurrence[3])
        print(100 * len_berger_kepler_transiters5/berger_kepler_counts[4], physical_planet_occurrence[4])
        print(100 * len_berger_kepler_transiters1/berger_kepler_counts[5], physical_planet_occurrence[5])
        """

        #len_berger_kepler_transiters1, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters1, completeness_map_np, radius_grid, period_grid) #completeness_map_np vs completeness_map
        #len_berger_kepler_transiters2, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters2, completeness_map_np, radius_grid, period_grid)
        #len_berger_kepler_transiters3, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters3, completeness_map_np, radius_grid, period_grid)
        #len_berger_kepler_transiters4, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters4, completeness_map_np, radius_grid, period_grid)
        #len_berger_kepler_transiters5, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters5, completeness_map_np, radius_grid, period_grid)
        #len_berger_kepler_transiters6, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters6, completeness_map_np, radius_grid, period_grid)
        #len_berger_kepler_transiters7, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters7, completeness_map_np, radius_grid, period_grid)
        #len_berger_kepler_transiters8, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters8, completeness_map_np, radius_grid, period_grid)
        #len_berger_kepler_transiters9, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters9, completeness_map_np, radius_grid, period_grid)
        #len_berger_kepler_transiters = np.array([len_berger_kepler_transiters1, len_berger_kepler_transiters2, len_berger_kepler_transiters3, len_berger_kepler_transiters4, len_berger_kepler_transiters5, len_berger_kepler_transiters6, len_berger_kepler_transiters7, len_berger_kepler_transiters8, len_berger_kepler_transiters9])

        adjusted_planet_occurrence = len_berger_kepler_transiters/berger_kepler_counts
        adjusted_planet_occurrences_all.append(adjusted_planet_occurrence)
        #"""

# one-time creation of empirical completeness map, using the first detection of each of the 30 Populations and averaging them
print("COMPLETENESS MAPS")
print(completeness_all)

mean_completeness = np.nanmean(completeness_all, axis=0)
std_completeness = np.nanstd(completeness_all, axis=0)
#print(mean_completeness)
#print(std_completeness)
#utils.plot_completeness(mean_completeness, std_completeness, radius_grid, period_grid)
#pd.DataFrame(np.nanmean(completeness_all, axis=0)).to_csv(path+'data/completeness_map_empirical'+name+'.csv', index=False)

print("")
print("f: ", np.mean(fs), np.std(fs))
print("")

mean_physical_planet_occurrences = np.nanmean(physical_planet_occurrences, axis=0)
yerr = np.std(physical_planet_occurrences, axis=0)
print("mean physical planet occurrences, and yerr: ", mean_physical_planet_occurrences, yerr, np.sum(mean_physical_planet_occurrences))

mean_recovered_planet_occurrences = 100 * np.nanmean(adjusted_planet_occurrences_all, axis=0)
yerr_recovered = 100 * np.std(adjusted_planet_occurrences_all, axis=0)
print("recovered planet occurrences, and yerr: ", mean_recovered_planet_occurrences, yerr_recovered, np.sum(mean_recovered_planet_occurrences))

mean_detected_planet_occurrences = 100 * np.nanmean(detected_planet_occurrences_all, axis=0)
yerr_detected = 100 * np.std(detected_planet_occurrences_all, axis=0)
print("detected occurrence: ", mean_detected_planet_occurrences)

#plt.errorbar(np.array(height_bins[1:]), mean_detected_planet_occurrences, yerr_detected, fmt='o', label='detected')
#height_bins_shifted = height_bins[1:] + np.array([15, 20, 25, 30, 35, 40])
#plt.errorbar(height_bins_shifted, mean_physical_planet_occurrences, yerr, fmt='o', color='purple', alpha=0.5, capsize=3, elinewidth=1, markeredgewidth=1, label='true')
#plt.errorbar(np.array(height_bins[1:]), mean_recovered_planet_occurrences, yerr_recovered, fmt='o', color='#03acb1', alpha=0.5, capsize=3, elinewidth=1, markeredgewidth=1, label='recovered')
#plt.xlabel(r"$Z_{max}$ [pc]")
#plt.ylabel("planets per 100 stars")
#plt.legend()
#plt.savefig(path+'plots/true_vs_recovered_'+name+'.png', format='png', bbox_inches='tight')
#plt.show()

heights = np.concatenate(heights)
ages = np.concatenate(ages)

"""
#fplt.(heights, ages, bins=40)
norm = 10
hist, xedges, yedges = np.histogram2d(ages, heights, bins=20)
hist = hist.T
#with np.errstate(divide='ignore', invalid='ignore'):  # suppress division by zero warnings
    #hist *= norm / hist.sum(axis=0, keepdims=True)
    #hist *= norm / hist.sum(axis=1, keepdims=True)
plt.pcolormesh(xedges, yedges, hist, cmap='Blues')

plt.ylabel('r"$Z_{max}$ [pc]"')
plt.xlabel('age [Gyr]')
#plt.xscale('log')
#plt.xlim([0, 1500])
#plt.ylim([0, 14])
plt.savefig(path+'plots/berger_height_age_'+name+'.png', format='png', bbox_inches='tight')
#plt.show()
"""

### MAKE RESULT PLOT
zink_sn_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([38, 29, 23, 24, 17]), 'occurrence_err1': np.array([5, 3, 2, 2, 4]), 'occurrence_err2': np.array([6, 3, 2, 4, 4])})
zink_se_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([28, 29, 25, 27, 18]), 'occurrence_err1': np.array([5, 3, 3, 4, 4]), 'occurrence_err2': np.array([5, 3, 3, 3, 4])})
zink_kepler_occurrence = np.array([38, 29, 23, 24, 17])+np.array([28, 29, 25, 27, 18])
zink_kepler_occurrence_err1 = np.round(np.sqrt((zink_sn_kepler['occurrence_err1'])**2 + (zink_se_kepler['occurrence_err1']**2)), 2)
zink_kepler_occurrence_err2 = np.round(np.sqrt((zink_sn_kepler['occurrence_err2'])**2 + (zink_se_kepler['occurrence_err2']**2)), 2)
zink_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': zink_kepler_occurrence, 'occurrence_err1': zink_kepler_occurrence_err1, 'occurrence_err2': zink_kepler_occurrence_err2})

z_max = np.logspace(2, 3.02, 100)
def model(x, tau, occurrence):

    dln = 0.0011
    scaleMax= 1000
    scaleMin = 100
    const = (scaleMax)**(tau+1)/(tau+1) - ((scaleMin)**(tau+1)/(tau+1))
    planet_yield = occurrence * x**(tau)/const/dln * 100
    
    return planet_yield

### but first, fit a power law 
def power_model(x, yerr, y=None):

    tau = numpyro.sample("tau", dist.Uniform(-1., 0.))
    occurrence = numpyro.sample("occurrence", dist.Uniform(0.01, 1.))

    dln = 0.0011
    scaleMax= 1000
    scaleMin = 100
    const = (scaleMax)**(tau+1)/(tau+1) - ((scaleMin)**(tau+1)/(tau+1))
    planet_yield = occurrence * x**(tau)/const/dln * 100
    #print("planet yield: ", planet_yield)
    #print("yerr: ", yerr)
    #print("y: ", y)
    #print("tau: ", tau)
    #print("occurrence: ", occurrence)
    #print("sample model: ", model(z_max, tau, occurrence))
    
    with numpyro.plate("data", len(x)):
        numpyro.sample("planet_yield", dist.Normal(planet_yield, yerr), obs=y)

# find MAP solution
init_params = {
    "tau": -0.35,
    "occurrence": 0.3,
}

run_optim = numpyro_ext.optim.optimize(
        power_model, init_strategy=numpyro.infer.init_to_median()
    )
print(height_bins[:-1], yerr_recovered, mean_recovered_planet_occurrences)
opt_params = run_optim(jax.random.PRNGKey(5), height_bin_midpoints, yerr_recovered, y=mean_recovered_planet_occurrences)
#print("opt params: ", opt_params)

# sample posteriors for best-fit model to simulated data
sampler = infer.MCMC(
    infer.NUTS(power_model, dense_mass=True,
        regularize_mass_matrix=False,
        init_strategy=numpyro.infer.init_to_value(values=opt_params)), 
    num_warmup=2000,
    num_samples=3000,
    num_chains=4,
    progress_bar=True,
)

sampler.run(jax.random.PRNGKey(0), height_bin_midpoints, yerr_recovered, y=mean_recovered_planet_occurrences)
inf_data = az.from_numpyro(sampler)
print(az.summary(inf_data))

tau_ours = inf_data.posterior.data_vars['tau'].mean().values
print("tau: ", tau_ours)
tau_std = inf_data.posterior.data_vars['tau'].std().values
print("tau std: ", tau_std)

occurrence_ours = inf_data.posterior.data_vars['occurrence'].mean().values
print("occurrence: ", occurrence_ours)
occurrence_std = inf_data.posterior.data_vars['occurrence'].std().values
print("occurrence std: ", occurrence_std)

### set up plotting
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
left, bottom, width, height = [0.16, 0.3, 0.15, 0.15]
ax2 = fig.add_axes([left, bottom, width, height])

# zink model
# calculate all models so that we can take one-sigma envelope
yield_max = []
yield_min = []
models_se = []
models_sn = []
zink_csv = pd.read_csv(path+'data/SupEarths_combine_GaxScale_teff_fresh.csv')
zink_csv_sn = pd.read_csv(path+'data/SubNeptunes_combine_GaxScale_teff_fresh.csv')

for i in range(len(zink_csv)):
    row = zink_csv.iloc[i]
    models_se.append(model(z_max, row['Tau'], row['Occurrence']))
zink_csv['model'] = models_se

for j in range(len(zink_csv_sn)):
    row = zink_csv_sn.iloc[i]
    models_sn.append(model(z_max, row['Tau'], row['Occurrence']))
zink_csv_sn['model'] = models_sn
sum_model = zink_csv['model'] + zink_csv_sn['model']
for temp_list in zip_longest(*sum_model):
    yield_max.append(np.percentile(temp_list, 84)) # plus one sigma
    yield_min.append(np.percentile(temp_list, 16)) # minus one sigma
ax1.fill_between(z_max, yield_max, yield_min, color='red', alpha=0.3, label='Zink+ 2023 posteriors') #03acb1

# zink+ 2023 data
ax1.errorbar(x=height_bin_midpoints, y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', color='red', alpha=0.5, capsize=3, elinewidth=1, markeredgewidth=1, label='Zink+ 2023 Kepler data')

# our simulated data
#height_bins_shifted1 = height_bins[1:] + np.array([7, 15, 18, 32, 48])
#height_bins_shifted1 = np.logspace(2.01, 3.01, 6)[1:]
height_bin_midpoints1 = 0.5 * (np.logspace(2.02,3.02,6)[1:] + np.logspace(2.02,3.02,6)[:-1])
ax1.errorbar(x=height_bin_midpoints1, y=mean_physical_planet_occurrences, yerr=yerr, fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, color='purple', alpha=0.5, label='model physical yield')

# our recovered data
#height_bins_shifted2 = height_bins[1:] + np.array([15, 30, 40, 65, 100])
#height_bins_shifted2 = np.logspace(2.02, 3.02, 6)[1:]
height_bin_midpoints2 = 0.5 * (np.logspace(2.04,3.04,6)[1:] + np.logspace(2.04,3.04,6)[:-1])
ax1.errorbar(x=height_bin_midpoints2, y=mean_recovered_planet_occurrences, yerr=yerr_recovered, fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, color='#03acb1', alpha=0.5, label='model recovered yield')

# plot our best fit posteriors
our_yield_max = []
our_yield_min = []
our_models = []
for j in range(len(inf_data.posterior.data_vars['occurrence'])):

    tau = 0.5 * (inf_data.posterior.data_vars['tau'].values[0][j] + inf_data.posterior.data_vars['tau'].values[1][j])
    occurrence = 0.5 * (inf_data.posterior.data_vars['occurrence'].values[0][j] + inf_data.posterior.data_vars['occurrence'].values[1][j])
    #tau = inf_data.posterior.data_vars['tau'].values[0][j]
    #occurrence = inf_data.posterior.data_vars['occurrence'].values[0][j] 
    #print(z_max, tau, occurrence)
    #quit()
    our_models.append(model(z_max, tau, occurrence))
for temp_list2 in zip_longest(*our_models):
    our_yield_max.append(np.percentile(temp_list2, 84)) # plus one sigma
    our_yield_min.append(np.percentile(temp_list2, 16)) # minus one sigma
#print("OUR YIELD: ", our_models)
#print(len(our_models))
ax1.fill_between(z_max, our_yield_max, our_yield_min, color='#03acb1', alpha=0.3, label='model best-fit posteriors') 

ax1.set_xlim([100, 1100]) # add buffer room 
ax1.set_ylim([6, 100])
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.set_xticks(ticks=[100, 300, 1000])
ax1.set_yticks(ticks=[10, 30, 100])
ax1.set_xlabel(r"$Z_{max}$ [pc]")
ax1.set_ylabel("planets per 100 stars")
#plt.title('m12z-like SFH')
#ax1.set_title('f=%1.2f' % frac1 + ' if <=%i ' % threshold + 'Gyr; f=%1.2f' % frac2 + ' if >%i ' % threshold + 'Gyr') 
ax1.legend(loc='upper left', bbox_to_anchor=[1.0, 1.05])

# step model
x = np.linspace(0, 14, 1000)
y = np.where(x <= threshold, frac1, frac2)

# monotonic model
#b = frac1
#m = (frac2 - frac1)/(x[-1] - x[0])
#y = b + m * x

# piecewise model
#m = (frac2 - frac1)/(x[-1] - threshold)
#y = np.where(x < threshold, frac1, frac1 + m * (x-threshold))

ax2.plot(x, y, color='powderblue')
ax2.set_xlabel('cosmic age [Gyr]')
ax2.set_ylabel('planet host fraction')
ax2.set_ylim([0,1])

fig.tight_layout()
plt.savefig(path+'plots/model_vs_zink_'+name+'_empirical_completeness.png', format='png', bbox_inches='tight')

#plt.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='Zink+ 2023 Kepler data')
#plt.scatter(x=zink_kepler['scale_height'], y=physical_planet_occurrence, c='red', label='model')
#plt.xlabel(r'$Z_{max}$ [pc]')
#plt.ylabel('planets per 100 stars')
#plt.legend()
#plt.tight_layout()
#plt.savefig(path+'plots/'+name)
plt.show()
