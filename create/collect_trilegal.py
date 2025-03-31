###############################################################
# CALCULATE PHYSICAL AND DETECTED PLANETS FROM TRILEGAL DATASET
###############################################################

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
"""

# operative parameters
threshold = 11
frac1 = 0.20
frac2 = 0.90

name_thresh = 11
name_f1 = 20
name_f2 = 90
name = 'step_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2)
#name = 'monotonic_'+str(name_f1)+'_'+str(name_f2) 
name = 'piecewise_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2) 

sim = sorted(glob(path+'data/trilegal2/' + name + '/' + name + '*'))#[11:]
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

height_adjusted_planet_occurrences_all = []

cdpp1_mean = []
cdpp1_std = []
cdpp3_mean = []
cdpp3_std = []
cdpp5_mean = []
cdpp5_std = []
stellar_radius1_mean = []
stellar_radius1_std = []
stellar_radius3_mean = []
stellar_radius3_std = []
stellar_radius5_mean = []
stellar_radius5_std = []
sn1_mean = []
sn1_std = []
sn3_mean = []
sn3_std = []
sn5_mean = []
sn5_std = []
age1_mean = []
age1_std = []
age3_mean = []
age3_std = []
age5_mean = []
age5_std = []
len1 = []
len3 = []
len5 = []

period_grid = np.logspace(np.log10(2), np.log10(40), 10) # formerly up to 300 days
radius_grid = np.linspace(1, 4, 10)

height_bins = np.logspace(2, 3, 6) # ah, so the above are the midpoints of the actual bins they used, I guess
height_bin_midpoints = 0.5 * (np.logspace(2,3,6)[1:] + np.logspace(2,3,6)[:-1])

for i in tqdm(range(len(sim))):
    trilegal_kepler_all = pd.read_csv(sim[i], sep=',') #, on_bad_lines='skip'
    trilegal_kepler_all = trilegal_kepler_all.reset_index() # convert index to regular column so I can explode on it later

    num_hosts = trilegal_kepler_all.loc[trilegal_kepler_all['num_planets']>0]
    f = len(num_hosts)/len(trilegal_kepler_all)
    print("f: ", len(num_hosts)/len(trilegal_kepler_all))
    fs.append(f)

    trilegal_kepler_all = trilegal_kepler_all.dropna(subset=['height'])
    trilegal_kepler_all['height'] = trilegal_kepler_all['height'] 
    trilegal_kepler_all['periods'] = trilegal_kepler_all['periods'].apply(literal_eval_w_exceptions)
    trilegal_kepler_all['planet_radii'] = trilegal_kepler_all['planet_radii'].apply(literal_eval_w_exceptions)
    trilegal_kepler_all['incls'] = trilegal_kepler_all['incls'].apply(literal_eval_w_exceptions)
    trilegal_kepler_all['mutual_incls'] = trilegal_kepler_all['mutual_incls'].apply(literal_eval_w_exceptions)
    trilegal_kepler_all['eccs'] = trilegal_kepler_all['eccs'].apply(literal_eval_w_exceptions)
    trilegal_kepler_all['omegas'] = trilegal_kepler_all['omegas'].apply(literal_eval_w_exceptions)

    # remove cdpp=0
    trilegal_kepler_all = trilegal_kepler_all.loc[trilegal_kepler_all['rrmscdpp06p0']>0]

    #trilegal_kepler_all = trilegal_kepler_all.loc[(trilegal_kepler_all['height'] <= 1000) & (trilegal_kepler_all['age'] <= 13.7)] 
    print("FINAL SAMPLE COUNT: ", len(trilegal_kepler_all))

    #utils.plot_properties(berger_kepler_all['iso_teff'], berger_kepler_all['iso_age'])
    heights.append(np.array(trilegal_kepler_all['height']))
    ages.append(np.array(trilegal_kepler_all['age']))

    # RESULT PLOT STUFF
    trilegal_kepler_all['height_bins'] = pd.cut(trilegal_kepler_all['height'], bins=height_bins, include_lowest=True)
    trilegal_kepler_counts = np.array(trilegal_kepler_all.groupby(['height_bins']).count().reset_index()['index'])
    utils.plot_age_vs_height(trilegal_kepler_all, label='TRI')
    quit()

    # isolate planet hosts and bin them by galactic height
    trilegal_kepler_planets = trilegal_kepler_all.loc[trilegal_kepler_all['num_planets'] > 0]
    trilegal_kepler_planets = trilegal_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas']).reset_index(drop=True)

    trilegal_kepler_planets = trilegal_kepler_planets.loc[(trilegal_kepler_planets['periods'] <= 40) & (trilegal_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
    trilegal_kepler_planets = trilegal_kepler_planets.loc[trilegal_kepler_planets['planet_radii'] <= 4.] # limit radii to fairly compare with SEs in Zink+ 2023 (2)...or how about include SNs too (4)?
    trilegal_kepler_planets_counts = np.array(trilegal_kepler_planets.groupby(['height_bins']).count().reset_index()['index'])

    physical_planet_occurrence = 100 * trilegal_kepler_planets_counts/trilegal_kepler_counts 
    physical_planet_occurrences.append(physical_planet_occurrence)

    detected_planet_occurrences = []
    adjusted_planet_occurrences = []
    transit_multiplicities = []
    geom_transit_multiplicities = []
    piv_physicals = []
    piv_detecteds = []

    height_adjusted_planet_occurrences = []

    #completeness_all = []
    for k in range(10): 

        #berger_kepler_planets_temp = berger_kepler_planets
        ### Simulate detections from these synthetic systems
        prob_detections, transit_statuses, sn, geom_transit_statuses = simulate_transit.calculate_transit_vectorized(trilegal_kepler_planets.periods, 
                                        trilegal_kepler_planets.stellar_radius, trilegal_kepler_planets.planet_radii,
                                        trilegal_kepler_planets.eccs, 
                                        trilegal_kepler_planets.incls, 
                                        trilegal_kepler_planets.omegas, trilegal_kepler_planets.stellar_mass,
                                        trilegal_kepler_planets.rrmscdpp06p0, angle_flag=True) 

        trilegal_kepler_planets['transit_status'] = transit_statuses[0]
        trilegal_kepler_planets['prob_detections'] = prob_detections[0]
        trilegal_kepler_planets['sn'] = sn
        trilegal_kepler_planets['geom_transit_status'] = geom_transit_statuses

        # need index to be str or tuple, else unhashable type when groupby.count()
        trilegal_kepler_planets['index'] = trilegal_kepler_planets['index'].apply(str) 

        # isolate detected transiting planets
        trilegal_kepler_transiters = trilegal_kepler_planets.loc[trilegal_kepler_planets['transit_status']==1]
        geom = trilegal_kepler_planets.loc[trilegal_kepler_planets['geom_transit_status']==1]
        non_geom = trilegal_kepler_planets.loc[trilegal_kepler_planets['geom_transit_status']!=1]
        
        ### Completeness
        # Calculate completeness map
        completeness_map, piv_physical, piv_detected = simulate_helpers.completeness(trilegal_kepler_planets, trilegal_kepler_transiters)
        completeness_threshold = 0.0025 # completeness threshold under which period/radius cell is not counted; 0.5% results in full recovery, but let's round up to 1%
        #completeness_threshold = 0.0005
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
        transit_multiplicity = trilegal_kepler_transiters.groupby('index').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index()['index']
        transit_multiplicity = transit_multiplicity.to_list()
        transit_multiplicity += [0.] * (6 - len(transit_multiplicity)) # pad with zeros to match length of k
        transit_multiplicities_all.append(transit_multiplicity)

        # also calculate the geometric transit multiplicity
        geom_transiters_trilegal_kepler = trilegal_kepler_planets.loc[trilegal_kepler_planets['geom_transit_status']==1]
        geom_transit_multiplicity = geom_transiters_trilegal_kepler.groupby('index').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index()['index']
        geom_transit_multiplicity = geom_transit_multiplicity.to_list()
        geom_transit_multiplicity += [0.] * (6 - len(geom_transit_multiplicity)) # pad with zeros to match length of k
        geom_transit_multiplicities_all.append(geom_transit_multiplicity)

        # calculate detected occurrence rate 
        trilegal_kepler_transiters_counts = np.array(trilegal_kepler_transiters.groupby(['height_bins']).count().reset_index()['index'])
        detected_planet_occurrence = trilegal_kepler_transiters_counts/trilegal_kepler_counts
        detected_planet_occurrences_all.append(detected_planet_occurrence)

        # same, but adjust for period- and radius-dependent completeness 
        trilegal_kepler_transiters1 = trilegal_kepler_transiters.loc[(trilegal_kepler_transiters['height'] > 100) & (trilegal_kepler_transiters['height'] <= np.logspace(2,3,6)[1])]
        trilegal_kepler_transiters2 = trilegal_kepler_transiters.loc[(trilegal_kepler_transiters['height'] > np.logspace(2,3,6)[1]) & (trilegal_kepler_transiters['height'] <= np.logspace(2,3,6)[2])]
        trilegal_kepler_transiters3 = trilegal_kepler_transiters.loc[(trilegal_kepler_transiters['height'] > np.logspace(2,3,6)[2]) & (trilegal_kepler_transiters['height'] <= np.logspace(2,3,6)[3])]
        trilegal_kepler_transiters4 = trilegal_kepler_transiters.loc[(trilegal_kepler_transiters['height'] > np.logspace(2,3,6)[3]) & (trilegal_kepler_transiters['height'] <= np.logspace(2,3,6)[4])]
        trilegal_kepler_transiters5 = trilegal_kepler_transiters.loc[(trilegal_kepler_transiters['height'] > np.logspace(2,3,6)[4]) & (trilegal_kepler_transiters['height'] <= 1000)]
        #print(trilegal_kepler_transiters1) # 70 (0.039)
        #print(trilegal_kepler_transiters2) # 150 (0.038)
        #print(trilegal_kepler_transiters5) # 31 (0.009)
        #quit()

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

        cdpp1_mean.append(np.mean(trilegal_kepler_transiters1['rrmscdpp06p0']))
        cdpp1_std.append(np.std(trilegal_kepler_transiters1['rrmscdpp06p0']))
        cdpp3_mean.append(np.mean(trilegal_kepler_transiters3['rrmscdpp06p0']))
        cdpp3_std.append(np.std(trilegal_kepler_transiters3['rrmscdpp06p0']))
        cdpp5_mean.append(np.mean(trilegal_kepler_transiters5['rrmscdpp06p0']))
        cdpp5_std.append(np.std(trilegal_kepler_transiters5['rrmscdpp06p0']))

        stellar_radius1_mean.append(np.mean(trilegal_kepler_transiters1['stellar_radius']))
        stellar_radius1_std.append(np.std(trilegal_kepler_transiters1['stellar_radius']))
        stellar_radius3_mean.append(np.mean(trilegal_kepler_transiters3['stellar_radius']))
        stellar_radius3_std.append(np.std(trilegal_kepler_transiters3['stellar_radius']))
        stellar_radius5_mean.append(np.mean(trilegal_kepler_transiters5['stellar_radius']))
        stellar_radius5_std.append(np.std(trilegal_kepler_transiters5['stellar_radius']))

        sn1_mean.append(np.mean(trilegal_kepler_transiters1['sn']))
        sn1_std.append(np.std(trilegal_kepler_transiters1['sn']))
        sn3_mean.append(np.mean(trilegal_kepler_transiters3['sn']))
        sn3_std.append(np.std(trilegal_kepler_transiters3['sn']))
        sn5_mean.append(np.mean(trilegal_kepler_transiters5['sn']))
        sn5_std.append(np.std(trilegal_kepler_transiters5['sn']))

        age1_mean.append(np.mean(trilegal_kepler_transiters1['age']))
        age1_std.append(np.std(trilegal_kepler_transiters1['age']))
        age3_mean.append(np.mean(trilegal_kepler_transiters3['age']))
        age3_std.append(np.std(trilegal_kepler_transiters3['age']))
        age5_mean.append(np.mean(trilegal_kepler_transiters5['age']))
        age5_std.append(np.std(trilegal_kepler_transiters5['age']))

        len1.append(len(trilegal_kepler_transiters1))
        len3.append(len(trilegal_kepler_transiters3))
        len5.append(len(trilegal_kepler_transiters5))

        #"""
        len_trilegal_kepler_transiters1, _ = simulate_helpers.adjust_for_completeness2(trilegal_kepler_transiters1, completeness_map, radius_grid, period_grid) #completeness_map_np vs completeness_map
        len_trilegal_kepler_transiters2, _ = simulate_helpers.adjust_for_completeness2(trilegal_kepler_transiters2, completeness_map, radius_grid, period_grid)
        len_trilegal_kepler_transiters3, _ = simulate_helpers.adjust_for_completeness2(trilegal_kepler_transiters3, completeness_map, radius_grid, period_grid)
        len_trilegal_kepler_transiters4, _ = simulate_helpers.adjust_for_completeness2(trilegal_kepler_transiters4, completeness_map, radius_grid, period_grid)
        len_trilegal_kepler_transiters5, _ = simulate_helpers.adjust_for_completeness2(trilegal_kepler_transiters5, completeness_map, radius_grid, period_grid)
        len_trilegal_kepler_transiters = np.array([len_trilegal_kepler_transiters1, len_trilegal_kepler_transiters2, len_trilegal_kepler_transiters3, len_trilegal_kepler_transiters4, len_trilegal_kepler_transiters5])
        #len_trilegal_kepler_recovered, recovered_piv = simulate_helpers.adjust_for_completeness(trilegal_kepler_transiters, completeness_map, radius_grid, period_grid)
        adjusted_planet_occurrence = np.array(len_trilegal_kepler_transiters/trilegal_kepler_counts)

        ### adjust by height bins too! 
        height_adjusted_planet_occurrence = simulate_helpers.completeness_height(100 * adjusted_planet_occurrence, physical_planet_occurrence) 
        adjusted_planet_occurrences_all.append(adjusted_planet_occurrence) 
        height_adjusted_planet_occurrences_all.append(height_adjusted_planet_occurrence)

        ### this is for testing why the later height bins have much smaller completeness
        """
        trilegal_kepler_planets1 = trilegal_kepler_planets.loc[(trilegal_kepler_planets['height'] > 100) & (trilegal_kepler_planets['height'] <= np.logspace(2,3,6)[1])]
        trilegal_kepler_planets2 = trilegal_kepler_planets.loc[(trilegal_kepler_planets['height'] > np.logspace(2,3,6)[1]) & (trilegal_kepler_planets['height'] <= np.logspace(2,3,6)[2])]
        trilegal_kepler_planets3 = trilegal_kepler_planets.loc[(trilegal_kepler_planets['height'] > np.logspace(2,3,6)[2]) & (trilegal_kepler_planets['height'] <= np.logspace(2,3,6)[3])]
        trilegal_kepler_planets4 = trilegal_kepler_planets.loc[(trilegal_kepler_planets['height'] > np.logspace(2,3,6)[3]) & (trilegal_kepler_planets['height'] <= np.logspace(2,3,6)[4])]
        trilegal_kepler_planets5 = trilegal_kepler_planets.loc[(trilegal_kepler_planets['height'] > np.logspace(2,3,6)[4]) & (trilegal_kepler_planets['height'] <= 1000)]
        print(trilegal_kepler_planets1) # 1807
        print(trilegal_kepler_planets2) # 3975
        print(trilegal_kepler_planets5) # 3390

        print(np.nanmedian(trilegal_kepler_planets1['rrmscdpp06p0']))
        print(np.nanmedian(trilegal_kepler_planets2['rrmscdpp06p0']))
        print(np.nanmedian(trilegal_kepler_planets3['rrmscdpp06p0']))
        print(np.nanmedian(trilegal_kepler_planets4['rrmscdpp06p0']))
        print(np.nanmedian(trilegal_kepler_planets5['rrmscdpp06p0']))

        print(np.nanmedian(trilegal_kepler_transiters1['rrmscdpp06p0']))
        print(np.nanmedian(trilegal_kepler_transiters2['rrmscdpp06p0']))
        print(np.nanmedian(trilegal_kepler_transiters3['rrmscdpp06p0']))
        print(np.nanmedian(trilegal_kepler_transiters4['rrmscdpp06p0']))
        print(np.nanmedian(trilegal_kepler_transiters5['rrmscdpp06p0']))

        completeness_map1, piv_physical1, piv_detected1 = simulate_helpers.completeness(trilegal_kepler_planets1, trilegal_kepler_transiters1)
        completeness_map2, piv_physical2, piv_detected2 = simulate_helpers.completeness(trilegal_kepler_planets2, trilegal_kepler_transiters2)
        completeness_map3, piv_physical3, piv_detected3 = simulate_helpers.completeness(trilegal_kepler_planets3, trilegal_kepler_transiters3)
        completeness_map4, piv_physical4, piv_detected4 = simulate_helpers.completeness(trilegal_kepler_planets4, trilegal_kepler_transiters4)
        completeness_map5, piv_physical5, piv_detected5 = simulate_helpers.completeness(trilegal_kepler_planets5, trilegal_kepler_transiters5)
        completeness_threshold1 = 0.0005
        completeness_threshold2 = 0.0005
        completeness_threshold3 = 0.0005
        completeness_threshold4 = 0.0001
        completeness_threshold5 = 0.000001
        completeness_map1 = completeness_map1.mask(completeness_map1 < completeness_threshold1) 
        completeness_map2 = completeness_map2.mask(completeness_map2 < completeness_threshold2) 
        completeness_map3 = completeness_map3.mask(completeness_map3 < completeness_threshold3) 
        completeness_map4 = completeness_map4.mask(completeness_map4 < completeness_threshold4) 
        completeness_map5 = completeness_map5.mask(completeness_map5 < completeness_threshold5) 

        len_trilegal_kepler_transiters1, _ = simulate_helpers.adjust_for_completeness(trilegal_kepler_transiters1, completeness_map1, radius_grid, period_grid) #completeness_map_np vs completeness_map
        len_trilegal_kepler_transiters2, _ = simulate_helpers.adjust_for_completeness(trilegal_kepler_transiters2, completeness_map2, radius_grid, period_grid)
        len_trilegal_kepler_transiters3, _ = simulate_helpers.adjust_for_completeness(trilegal_kepler_transiters3, completeness_map3, radius_grid, period_grid)
        len_trilegal_kepler_transiters4, _ = simulate_helpers.adjust_for_completeness(trilegal_kepler_transiters4, completeness_map4, radius_grid, period_grid)
        len_trilegal_kepler_transiters5, _ = simulate_helpers.adjust_for_completeness(trilegal_kepler_transiters5, completeness_map5, radius_grid, period_grid)
        len_trilegal_kepler_transiters = np.array([len_trilegal_kepler_transiters1, len_trilegal_kepler_transiters2, len_trilegal_kepler_transiters3, len_trilegal_kepler_transiters4, len_trilegal_kepler_transiters5])
        adjusted_planet_occurrence = len_trilegal_kepler_transiters/trilegal_kepler_counts
        print(100 * adjusted_planet_occurrence)
        print(physical_planet_occurrence)
        quit()
        adjusted_planet_occurrences_all.append(adjusted_planet_occurrence) 
        """
        #"""
    
    adjusted_planet_occurrences.append(adjusted_planet_occurrence)
    #height_adjusted_detection = 100 * np.array(adjusted_planet_occurrences) / np.array(physical_planet_occurrences)
    #height_adjusted_planet_occurrence = 100 * np.array(adjusted_planet_occurrences) / height_adjusted_detection
    #height_adjusted_planet_occurrences.append(height_adjusted_planet_occurrence)

# compare key sensitivity parameters with B20 across diagnostic bins, to see where the big differences are
#"""
print("cdpps")
print(np.mean(cdpp1_mean), np.mean(cdpp1_std))
print(np.mean(cdpp3_mean), np.mean(cdpp3_std))
print(np.mean(cdpp5_mean), np.mean(cdpp5_std))
#plt.hist(trilegal_kepler_transiters1['rrmscdpp06p0'], label='bin 1', alpha=0.5, density=True)
#plt.hist(trilegal_kepler_transiters3['rrmscdpp06p0'], label='bin 3', alpha=0.5, density=True)
#plt.hist(trilegal_kepler_transiters5['rrmscdpp06p0'], label='bin 5', alpha=0.5, density=True)
#plt.xlabel('CDPP (6 hr) [ppm]')
#plt.legend()
#plt.show()

print("stellar radius")
print(np.mean(stellar_radius1_mean), np.mean(stellar_radius1_std))
print(np.mean(stellar_radius3_mean), np.mean(stellar_radius3_std))
print(np.mean(stellar_radius5_mean), np.mean(stellar_radius5_std))
#plt.hist(trilegal_kepler_transiters1['stellar_radius'], label='bin 1', alpha=0.5, density=True)
#plt.hist(trilegal_kepler_transiters3['stellar_radius'], label='bin 3', alpha=0.5, density=True)
#plt.hist(trilegal_kepler_transiters5['stellar_radius'], label='bin 5', alpha=0.5, density=True)
#plt.xlabel('stellar radius [Solar radius]')
#plt.legend()
#plt.show()

print("s/n")
print(np.mean(sn1_mean), np.mean(sn1_std))
print(np.mean(sn3_mean), np.mean(sn3_std))
print(np.mean(sn5_mean), np.mean(sn5_std))
#plt.hist(trilegal_kepler_transiters1['sn'], label='bin 1', alpha=0.5, density=True)
#plt.hist(trilegal_kepler_transiters3['sn'], label='bin 3', alpha=0.5, density=True)
#plt.hist(trilegal_kepler_transiters5['sn'], label='bin 5', alpha=0.5, density=True)
#plt.xlabel('s/n')
#plt.legend()
#plt.show()

print("age")
print(np.mean(age1_mean), np.mean(age1_std))
print(np.mean(age3_mean), np.mean(age3_std))
print(np.mean(age5_mean), np.mean(age5_std))
#plt.hist(trilegal_kepler_transiters1['age'], label='bin 1', alpha=0.5, density=True)
#plt.hist(trilegal_kepler_transiters3['age'], label='bin 3', alpha=0.5, density=True)
#plt.hist(trilegal_kepler_transiters5['age'], label='bin 5', alpha=0.5, density=True)
#plt.xlabel('age [Gyr]')
#plt.legend()
#plt.show()

print("bin sizes")
print(np.mean(len1), np.std(len1))
print(np.mean(len3), np.std(len3))
print(np.mean(len5), np.std(len5))
#"""

# one-time creation of empirical completeness map, using the first detection of each of the 30 Populations and averaging them
#print("COMPLETENESS MAPS")
#print(completeness_all)

mean_completeness = np.nanmean(completeness_all, axis=0)
std_completeness = np.nanstd(completeness_all, axis=0)
#print(mean_completeness)
#print(std_completeness)
#utils.plot_completeness(mean_completeness, std_completeness, radius_grid, period_grid)
#pd.DataFrame(np.nanmean(completeness_all, axis=0)).to_csv(path+'data/completeness_map_empirical'+name+'.csv', index=False)

print("")
print("f: ", np.mean(fs))
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

### adjust for decreasing sensitivity with increasing galactic height
"""
print(100 * adjusted_planet_occurrences_all)
print(physical_planet_occurrences)
#print(np.array(adjusted_planet_occurrences_all).reshape((3,3,5)), np.array(physical_planet_occurrences).shape)
height_adjusted_detection = 100 * np.array(adjusted_planet_occurrences_all).reshape((3,3,5)) / np.array(physical_planet_occurrences)
height_adjusted_planet_occurrences_all = np.array(adjusted_planet_occurrences_all).reshape((3,3,5)) / height_adjusted_detection
print(height_adjusted_planet_occurrences_all)
"""
#mean_recovered_planet_occurrences = np.nanmean(height_adjusted_planet_occurrences_all, axis=0)
#yerr_recovered = np.std(height_adjusted_planet_occurrences_all, axis=0)
#print(height_adjusted_planet_occurrences_all)
#print("height-adjusted recovered planet occurrences, and yerr: ", mean_recovered_planet_occurrences, yerr_recovered)

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

    tau = numpyro.sample("tau", dist.Uniform(-1., 1.)) # U(-1, 0)
    occurrence = numpyro.sample("occurrence", dist.Uniform(0.01, 1.)) # U(0.01, 1)

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
m = (frac2 - frac1)/(x[-1] - threshold)
y = np.where(x < threshold, frac1, frac1 + m * (x-threshold))

ax2.plot(x, y, color='powderblue')
ax2.set_xlabel('cosmic age [Gyr]')
ax2.set_ylabel('planet host fraction')
ax2.set_ylim([0,1])

fig.tight_layout()
plt.savefig(path+'plots/trilegal/trilegal_model_vs_zink_'+name+'_empirical_completeness.png', format='png', bbox_inches='tight')

#plt.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='Zink+ 2023 Kepler data')
#plt.scatter(x=zink_kepler['scale_height'], y=physical_planet_occurrence, c='red', label='model')
#plt.xlabel(r'$Z_{max}$ [pc]')
#plt.ylabel('planets per 100 stars')
#plt.legend()
#plt.tight_layout()
#plt.savefig(path+'plots/trilegal/'+name)
plt.show()

"""
# planet host fractions
step functions
model 1, (12, 20, 80): 0.30, or (12, 25, 80): 0.36, or (12, 25, 75): 0.35, or (12, 25, 70): 0.34, or (12, 25, 50): 0.30
model 2, (11.5, 10, 80): 0.31, or (11.5, 15, 70): 0.31 (bad both ways), or (11.5, 15, 65): 0.30 (nope, overshot it), or (11.5, 20, 65): 0.33
model 3, (11, 15, 60): 0.30, or (11, 10, 65): 0.32 
model 4, (9.5, 5, 45): 0.30, or (9.5, 15, 75): 0.34
model 5, (7.5, 1, 40): 0.33
model 6, (5.5, 1, 35): 0.34, or (5.5, 1, 30): 0.29

piecewise functions
model 1, (7.5, 1, 40): 0.33, -0.36 +/- 0.19, normalization a hair high but slope is good
model 2, (7, 10, 55): 0.32, -0.37 +/- 0.17, the most perfect model I have ever seen
model 3 (5, 15, 45): 0.33, -0.31 +/- 0.17, good fit

stars: ~66000
"""