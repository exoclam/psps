##################################################
### Testing individual models for Paper III ######
##################################################

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
import numpyro
from numpyro import distributions as dist, infer
import numpyro_ext
from numpyro_ext.distributions import MixtureGeneral
from tqdm import tqdm
from ast import literal_eval
import seaborn as sns

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
import arviz as az
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

berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell

# Berger+ 2020 sample has lots of stellar params we need, but no source_id
berger = Table.read(path+'data/berger_kepler_stellar_fgk.csv', data_end=20000) # turn off data_end when not testing heights
# Bedell cross-match has the Gaia DR3 source_id we need to calculate Zmax with Gala
megan = Table.read(path+'data/kepler_dr3_good.fits')

# cross-match the cross-matches (we only lose ~100 stars)
merged = join(berger, megan, keys='kepid')
merged.rename_column('parallax_2', 'parallax')

berger_kepler = berger_kepler.loc[berger_kepler['kepid'].isin(merged['kepid'])]
berger_kepler['height'] = pd.read_csv(path+'data/zmaxes.csv')#[:len(berger_kepler)]

"""
### PLOT COMPLETENESS MAP
#completeness_map = utils.completeness(berger_kepler).T # I did this on HPG. It took 11 hours.
completeness_map = pd.read_csv(path+'data/completeness_map_empirical.csv') # _empirical was done on the first detected iterations of each of the 30 Populations, averaged together
completeness_map_np = completeness_map.to_numpy()
completeness_map_np = np.flip(completeness_map_np, axis=0)
print(completeness_map_np)
#completeness_map_df = pd.DataFrame(completeness_map, columns=np.logspace(np.log10(2), np.log10(300), 9), index=np.linspace(1, 4, 9))
#print(completeness_map_df)

plt.figure(figsize=(10, 5)) 
a = plt.imshow(completeness_map_np, cmap='Purples', interpolation='none', extent=[2,300,1,4])
plt.xlabel('period [days]')
plt.ylabel(r'$R_p$ [$R_{\oplus}$]') 
plt.xscale('log')
plt.colorbar(a, label='completeness')
plt.savefig(path+'plots/completeness_map_empirical.png', format='png', bbox_inches='tight')
plt.show()
quit()
"""

completeness_map = pd.read_csv(path+'data/completeness_map.csv')
completeness_map_np = completeness_map.to_numpy()
completeness_map_np = np.flip(completeness_map_np, axis=0)

"""
### TEST F1 AND F2 PER THRESHOLD (this is not a useful exercise)
kois = pd.read_csv(path+'data/cumulative_2021.03.04_20.04.43.csv')
kois = kois.loc[kois.koi_disposition != 'FALSE POSITIVE']

positives_kepler = pd.merge(berger_kepler, kois, how='inner', right_on='kepid', left_on='KIC') 

# hmm, shouldn't we already be able to back out the fraction of Kepler hosts per age threshold, to rough order?
thresholds = np.linspace(1, 10, 19)

for thresh in thresholds:
    berger_kepler_old = berger_kepler.loc[berger_kepler['iso_age'] >= thresh]
    berger_kepler_young = berger_kepler.loc[berger_kepler['iso_age'] < thresh]

    planets_old = positives_kepler.loc[positives_kepler['iso_age'] >= thresh]
    planets_young = positives_kepler.loc[positives_kepler['iso_age'] < thresh]

    #print(planets_old.drop_duplicates(subset='kepid_x'))
    #print(berger_kepler_old.drop_duplicates(subset='kepid'))

    f1 = len(planets_old.drop_duplicates(subset='kepid_x'))/len(berger_kepler_old.drop_duplicates(subset='kepid'))
    f2 = len(planets_young.drop_duplicates(subset='kepid_x'))/len(berger_kepler_young.drop_duplicates(subset='kepid'))

    print("age: ", thresh, ", threshold: ", 13.7 - thresh, ", f1: ", f1, ", f2: ", f2)

quit()
"""

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

# planet formation history model parameters
threshold = 9.5 # cosmic age in Gyr; 13.7 minus stellar age, then round
frac1 = 0.3 # frac1 must be < frac2 if comparing cosmic ages
frac2 = 0.3 # 0.55 led to f=0.3, high Z being too high, low Z being a bit low; yet, 0.6 led to f=0.23...?
# 10, 0.15, 0.55 led to f=0.28, high Z being fine, low Z being a bit low; 10, 0.15, 0.6, f=0.3, low Z too high, high Z too low --> 10, 0.25, 0.5, led to f=0.34, low Z too low, high Z too high (basically flat) --> 10, 0.2, 0.6, led to f=0.34 and basically flat --> 10, 0.1, 0.7, f=0.31, low Z is good, high Z is a bit high --> 10, 0.05, 0.7, f=0.28, perfect match tho
# 5.5, 0.01, 0.4 led to f=0.3, flat line with high Z just above, low Z being way too low
# 12, 0.15, 0.7 led to f=0.21, low Z is way too low --> 12, 0.2, 0.85 led to f=0.28, low Z almost there --> 12, 0.25, 0.9 led to f=0.33, perfect low Z, high Z a bit high --> 12, 0.2, 0.9 led to f=0.28, low Z a bit low again.
# 11, 0.2, 0.9 led to f=0.36, everything too high --> 11, 0.15, 0.85, led to f=0.31, low Z a bit high --> 11, 0.15, 0.8, led to f=0.3, as close as I can get

# monotonic: 0, 0.8 --> f=0.47, medium and high Z way too high (basically flat); 0, 0.6 --> f=0.36, still flat, low Z a bit under, high Z still high; 0.05, 0.5 --> f=0.32, flat, low Z low, high Z high; 0.01, 0.55 --> f=0.32, flat as usual; 0.1, 0.4 --> f=0.28 (flat)

# piecewise: 0.1, 0.6, 10 --> f=0.18, much too low; 0.1, 0.6, 5 --> f=0.3, low Z a bit low, high Z a bit high; 0.15, 0.55, 6 --> f=0.29, basically flat; 0.05, 0.65, 5 --> f=0.29, still flat; 0.05, 0.7, 5 --> f=0.31, low Z a bit low, high Z a bit high, the closest we've gotten  
# piecewise: 7.5, 0.1, 0.8 --> f=0.3; 7, 0.05, 0.85 --> f=0.3 (good match, low Z a tad low);  7, 0.01, 0.9 --> f=0.28 (low Z a bit low, high Z a bit high); 8, 0.01, 0.9 --> f=0.24 (model fit was perfect, but low Z was actually too low)
# piecewise: 7.5, 0.01, 0.9 --> f=0.27 (lowest Z is low, all else perfect); 7.5, 0.01, 0.95 --> f=0.28 (flat, way off); 7.5, 0.03, 0.9 --> f=0.28 (flat, way off); 7.5, 0.05, 0.9 --> f=0.30 (better, if only lowest Z was a tad higher); 7.5, 0.1, 0.9 --> f=0.33 (flat, but bc the middle point is the only one that's too high)
# piecewise: 5, 0.05, 0.95 --> f=

#"""
# make Fig 3 for Paper III, in order to show a sample platter of step function models
thresholds = np.array([12, 11.5, 11, 9.5, 7.5, 5.5, 5.5])
f1s = np.array([0.2, 0.2, 0.15, 0.1, 0.01, 0.01, 0.33])
f2s = np.array([0.95, 0.85, 0.8, 0.65, 0.5, 0.4, 0.33])
utils.plot_models(thresholds, f1s, f2s, ax=None, lookback=True)
quit()
#"""

name_thresh = 115
name_f1 = 20
name_f2 = 85
name = 'step_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2)
name = 'piecewise_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2)

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
height_bins = np.array([0., 150, 250, 400, 650, 3000])
height_bins = np.linspace(0, 1500, 10)
height_bins = np.array([0., 120, 200, 300, 500, 800, 1500]) # the actual Zink Fig 12 height bins

"""
# histogram of heights
berger_kepler['height'] = berger_kepler['height'] * 1000
berger_kepler = berger_kepler.loc[berger_kepler['height'] <= 1500]
plt.hist(berger_kepler['height'], bins=height_bins)
plt.show()
quit()
"""

#height_bins = np.linspace(0, 1500, 31)
# for each model, draw around stellar age errors 10 times
for j in tqdm(range(2)): 

    #berger_kepler['iso_age_err1'] = berger_kepler['iso_age_err1'] * 2
    #berger_kepler['iso_age_err2'] = berger_kepler['iso_age_err2'] * 2

    # draw stellar radius, mass, and age using asymmetric errors 
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'iso_rad', 'iso_rad_err1', 'iso_rad_err2', 'stellar_radius')
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_age', 'iso_age_err1', 'iso_age_err2', 'age')
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'stellar_mass')

    # enrich berger_kepler with z_maxes using gala
    z_maxes = simulate_helpers.gala_galactic_heights(merged, output=False)
    berger_kepler_temp['height'] = z_maxes # kpc

    # I need to plot Figs 1 & 2; usually don't turn this on
    #print("before dropping heights: ", len(berger_kepler_temp))
    #berger_kepler_temp = berger_kepler_temp.dropna(subset='height')
    #print("after dropping heights: ", len(berger_kepler_temp))
    #utils.plot_properties(berger_kepler['iso_teff'], berger_kepler['iso_age'])

    ### create a Population object to hold information about the occurrence law governing that specific population
    # THIS IS WHERE YOU CHOOSE THE PLANET FORMATION HISTORY MODEL YOU WANT TO FORWARD MODEL
    pop = Population(berger_kepler_temp['age'], threshold, frac1, frac2)
    frac_hosts = pop.galactic_occurrence_step(threshold, frac1, frac2)
    #frac_hosts = pop.galactic_occurrence_monotonic(frac1, frac2)
    #frac_hosts = pop.galactic_occurrence_piecewise(frac1, frac2, threshold)

    alpha_se = np.random.normal(-1., 0.2)
    alpha_sn = np.random.normal(-1.5, 0.1)

    # create Star objects, with their planetary systems
    star_data = []
    for i in tqdm(range(len(berger_kepler_temp))): # 100
        star = Star(berger_kepler_temp['age'][i], berger_kepler_temp['stellar_radius'][i], berger_kepler_temp['stellar_mass'][i], berger_kepler_temp['rrmscdpp06p0'][i], frac_hosts[i], berger_kepler_temp['height'][i], alpha_se, alpha_sn, berger_kepler_temp['kepid'][i])
        star_update = {
            'kepid': star.kepid,
            'age': star.age,
            'stellar_radius': star.stellar_radius,
            'stellar_mass': star.stellar_mass,
            'rrmscdpp06p0': star.rrmscdpp06p0,
            'frac_host': star.frac_host,
            'height': star.height,
            'midplane': star.midplane,
            'prob_intact': star.prob_intact,
            'status': star.status,
            'sigma_incl': star.sigma_incl,
            'num_planets': star.num_planets,
            'periods': star.periods,
            'incls': star.incls,
            'mutual_incls': star.mutual_incls,
            'eccs': star.eccs,
            'omegas': star.omegas,
            'planet_radii': star.planet_radii
        }
        star_data.append(star_update)
        pop.add_child(star)

    # convert back to DataFrame
    berger_kepler_all = pd.DataFrame.from_records(star_data)
    
    num_hosts = berger_kepler_all.loc[berger_kepler_all['num_planets']>0]
    f = len(num_hosts)/len(berger_kepler_all)
    fs.append(f)

    berger_kepler_all = berger_kepler_all.dropna(subset=['height'])
    berger_kepler_all['age'] = berger_kepler_all['age']
    berger_kepler_all['height'] = berger_kepler_all['height'] * 1000 # to pc
    berger_kepler_all['periods'] = berger_kepler_all['periods'].apply(literal_eval_w_exceptions)
    berger_kepler_all['planet_radii'] = berger_kepler_all['planet_radii'].apply(literal_eval_w_exceptions)
    berger_kepler_all['incls'] = berger_kepler_all['incls'].apply(literal_eval_w_exceptions)
    berger_kepler_all['mutual_incls'] = berger_kepler_all['mutual_incls'].apply(literal_eval_w_exceptions)
    berger_kepler_all['eccs'] = berger_kepler_all['eccs'].apply(literal_eval_w_exceptions)
    berger_kepler_all['omegas'] = berger_kepler_all['omegas'].apply(literal_eval_w_exceptions)

    #print(sim[i], berger_kepler_all['height'])

    berger_kepler_all = berger_kepler_all.loc[(berger_kepler_all['height'] <= 1500) & (berger_kepler_all['age'] <= 13.5)] 
    print("FINAL SAMPLE COUNT: ", len(berger_kepler_all))

    #utils.plot_properties(berger_kepler_all['iso_teff'], berger_kepler_all['iso_age'])
    heights.append(np.array(berger_kepler_all['height']))
    ages.append(np.array(berger_kepler_all['age']))

    # RESULT PLOT STUFF
    berger_kepler_all['height_bins'] = pd.cut(berger_kepler_all['height'], bins=height_bins, include_lowest=True)
    berger_kepler_counts = np.array(berger_kepler_all.groupby(['height_bins']).count().reset_index()['kepid'])

    # isolate planet hosts and bin them by galactic height
    berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all['num_planets'] > 0]
    berger_kepler_planets = berger_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas']).reset_index(drop=True)
    #print("data frame: ", berger_kepler_planets)
    #print(np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid']))
    berger_kepler_planets_counts_precut = np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid'])
    #print(berger_kepler_planets_counts_precut)
    #quit()
    berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
    berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 4.] # limit radii to fairly compare with SEs in Zink+ 2023 (2)...or how about include SNs too (4)?
    berger_kepler_planets_counts = np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid'])

    physical_planet_occurrence = 100 * berger_kepler_planets_counts/berger_kepler_counts # normally yes
    physical_planet_occurrences.append(physical_planet_occurrence)
    print("physical planet occurrences: ", physical_planet_occurrences)

detected_planet_occurrences = []
adjusted_planet_occurrences = []
transit_multiplicities = []
geom_transit_multiplicities = []

for i in range(5):  # 10

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

    # isolate detected transiting planets
    berger_kepler_transiters = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]
    #print("physical planets: ", len((berger_kepler_planets['kepid'])))
    #print("detected planets: ", len((berger_kepler_transiters['kepid'])))
    
    # read out detected yields
    #berger_kepler_transiters.to_csv(path+'galactic-occurrence/systems/berger_kepler_planets_detected_'+str(i)+'.csv', index=False)

    # Read in pre-generated population
    #transiters_berger_kepler = pd.read_csv(path+'galactic-occurrence/systems/berger_kepler_planets_detected_'+str(i)+'.csv')

    berger_kepler_planets1 = berger_kepler_planets.loc[berger_kepler_planets['height'] <= 150]
    berger_kepler_planets2 = berger_kepler_planets.loc[(berger_kepler_planets['height'] > 150) & (berger_kepler_planets['height'] <= 250)]
    berger_kepler_planets3 = berger_kepler_planets.loc[(berger_kepler_planets['height'] > 250) & (berger_kepler_planets['height'] <= 400)]
    berger_kepler_planets4 = berger_kepler_planets.loc[(berger_kepler_planets['height'] > 400) & (berger_kepler_planets['height'] <= 650)]
    berger_kepler_planets5 = berger_kepler_planets.loc[berger_kepler_planets['height'] > 650]

    ### Completeness
    # Calculate completeness map(s)
    completeness_map, piv_physical, piv_detected = simulate_helpers.completeness(berger_kepler_planets, berger_kepler_transiters)
    completeness_threshold = 0.0025 # completeness threshold under which period/radius cell is not counted; 0.5% results in full recovery, but let's round up to 1%
    completeness_map = completeness_map.mask(completeness_map < completeness_threshold) # assert that completeness fractions lower than 1% are statistically insignificant
    print(completeness_map)

    # new, experimental way of applying completeness map
    berger_kepler_transiters['radius_bins'] = pd.cut(berger_kepler_transiters['planet_radii'], bins=radius_grid, include_lowest=True)
    berger_kepler_transiters['period_bins'] = pd.cut(berger_kepler_transiters['periods'], bins=period_grid, include_lowest=True)
    df_small = berger_kepler_transiters[['radius_bins', 'period_bins', 'transit_status']]
    
    df_small = df_small.groupby(['radius_bins','period_bins']).sum(['transit_status']).reset_index()
    df_piv = df_small.pivot(index='radius_bins', columns='period_bins', values='transit_status')
    df_piv = df_piv.to_numpy()/completeness_map_np
    len_berger_kepler_transiters = np.nansum(df_piv)

    #len_berger_kepler_transiters, df_piv = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters, pd.DataFrame(completeness_map_np), radius_grid, period_grid)
    print(len_berger_kepler_transiters, df_piv)
    print(len_berger_kepler_transiters/berger_kepler_counts)

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
    berger_kepler_transiters1 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 0) & (berger_kepler_transiters['height'] <= 120)]
    berger_kepler_transiters2 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 120) & (berger_kepler_transiters['height'] <= 200)]
    berger_kepler_transiters3 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 200) & (berger_kepler_transiters['height'] <= 300)]
    berger_kepler_transiters4 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 300) & (berger_kepler_transiters['height'] <= 500)]
    berger_kepler_transiters5 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 500) & (berger_kepler_transiters['height'] <= 800)]
    berger_kepler_transiters6 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 800) & (berger_kepler_transiters['height'] <= 1500)]

    #berger_kepler_transiters1 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 0) & (berger_kepler_transiters['height'] <= 150)]
    #berger_kepler_transiters2 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 150) & (berger_kepler_transiters['height'] <= 300)]
    #berger_kepler_transiters3 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 450) & (berger_kepler_transiters['height'] <= 600)]
    #berger_kepler_transiters4 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 600) & (berger_kepler_transiters['height'] <= 750)]
    #berger_kepler_transiters5 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 750) & (berger_kepler_transiters['height'] <= 900)]
    #berger_kepler_transiters6 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 900) & (berger_kepler_transiters['height'] <= 1050)]
    #berger_kepler_transiters7 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 1050) & (berger_kepler_transiters['height'] <= 1200)]
    #berger_kepler_transiters8 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 1200) & (berger_kepler_transiters['height'] <= 1350)]
    #berger_kepler_transiters9 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 1350) & (berger_kepler_transiters['height'] <= 1500)]

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
    len_berger_kepler_transiters6, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters6, completeness_map, radius_grid, period_grid)
    len_berger_kepler_transiters = np.array([len_berger_kepler_transiters1, len_berger_kepler_transiters2, len_berger_kepler_transiters3, len_berger_kepler_transiters4, len_berger_kepler_transiters5, len_berger_kepler_transiters6])

    #berger_kepler_transiters1['radius_bins'] = pd.cut(berger_kepler_transiters1['planet_radii'], bins=radius_grid, include_lowest=True)
    #berger_kepler_transiters1['period_bins'] = pd.cut(berger_kepler_transiters1['periods'], bins=period_grid, include_lowest=True)
    #df_small = berger_kepler_transiters1[['radius_bins', 'period_bins', 'transit_status']]
    #df_small = df_small.groupby(['radius_bins','period_bins']).sum(['transit_status']).reset_index()
    #df_piv = df_small.pivot(index='radius_bins', columns='period_bins', values='transit_status')
    #print(df_piv)
    #print(completeness_map)
    #quit()

    """
    len_berger_kepler_transiters1, df_piv1 = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters1, completeness_map_np, radius_grid, period_grid)
    len_berger_kepler_transiters2, df_piv2 = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters2, completeness_map_np, radius_grid, period_grid)
    len_berger_kepler_transiters3, df_piv3 = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters3, completeness_map_np, radius_grid, period_grid)
    len_berger_kepler_transiters4, df_piv4 = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters4, completeness_map_np, radius_grid, period_grid)
    len_berger_kepler_transiters5, df_piv5 = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters5, completeness_map_np, radius_grid, period_grid)
    len_berger_kepler_transiters6, df_piv6 = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters6, completeness_map_np, radius_grid, period_grid)
    len_berger_kepler_transiters7, df_piv7 = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters7, completeness_map_np, radius_grid, period_grid)
    len_berger_kepler_transiters8, df_piv8 = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters8, completeness_map_np, radius_grid, period_grid)
    len_berger_kepler_transiters9, df_piv9 = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters9, completeness_map_np, radius_grid, period_grid)
    len_berger_kepler_transiters = np.array([len_berger_kepler_transiters1, len_berger_kepler_transiters2, len_berger_kepler_transiters3, len_berger_kepler_transiters4, len_berger_kepler_transiters5, len_berger_kepler_transiters6, len_berger_kepler_transiters7, len_berger_kepler_transiters8, len_berger_kepler_transiters9])
    """

    ### what if we split height bins into 10 evenly-spaced bins
    berger_kepler_transiters['height_bins'] = pd.cut(berger_kepler_transiters['height'], bins=height_bins, include_lowest=True)

    adjusted_planet_occurrence = len_berger_kepler_transiters/berger_kepler_counts
    adjusted_planet_occurrences_all.append(adjusted_planet_occurrence)

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
#plt.show()
"""

### MAKE RESULT PLOT
zink_sn_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([38, 29, 23, 24, 17]), 'occurrence_err1': np.array([5, 3, 2, 2, 4]), 'occurrence_err2': np.array([6, 3, 2, 4, 4])})
zink_se_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([28, 29, 25, 27, 18]), 'occurrence_err1': np.array([5, 3, 3, 4, 4]), 'occurrence_err2': np.array([5, 3, 3, 3, 4])})
zink_kepler_occurrence = np.array([38, 29, 23, 24, 17])+np.array([28, 29, 25, 27, 18])
zink_kepler_occurrence_err1 = np.round(np.sqrt((zink_sn_kepler['occurrence_err1'])**2 + (zink_se_kepler['occurrence_err1']**2)), 2)
zink_kepler_occurrence_err2 = np.round(np.sqrt((zink_sn_kepler['occurrence_err2'])**2 + (zink_se_kepler['occurrence_err2']**2)), 2)
zink_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': zink_kepler_occurrence, 'occurrence_err1': zink_kepler_occurrence_err1, 'occurrence_err2': zink_kepler_occurrence_err2})

print("")
print("fs: ", fs)
print("")
#quit()

mean_physical_planet_occurrences = np.mean(physical_planet_occurrences, axis=0)
yerr = np.std(physical_planet_occurrences, axis=0)
print("mean physical planet occurrences, and yerr: ", mean_physical_planet_occurrences, yerr)

mean_detected_planet_occurrences = np.mean(detected_planet_occurrences_all, axis=0)
yerr_detected = np.std(detected_planet_occurrences_all, axis=0)

print("recovered planet occurrences: ", adjusted_planet_occurrences_all)

mean_recovered_planet_occurrences = 100 * np.mean(adjusted_planet_occurrences_all, axis=0)
yerr_recovered = 100 * np.std(adjusted_planet_occurrences_all, axis=0)
print("mean recovered planet occurrences, and yerr: ", mean_recovered_planet_occurrences, yerr_recovered)

print("transit multiplicities: ", transit_multiplicities_all)

plt.errorbar(np.array(height_bins[1:]), mean_recovered_planet_occurrences, yerr_recovered, fmt='o', label='recovered')
plt.errorbar(np.array(height_bins[1:]), mean_detected_planet_occurrences*100, yerr_detected, fmt='o', label='detected')
plt.errorbar(np.array(height_bins[1:]), mean_physical_planet_occurrences, yerr, fmt='o', label='true')
plt.legend()
plt.show()

z_max = np.logspace(2, 3, 100)
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
#opt_params = run_optim(jax.random.PRNGKey(5), np.array(height_bins[1:]), yerr, y=mean_physical_planet_occurrences)
opt_params = run_optim(jax.random.PRNGKey(5), np.array(height_bins[1:]), yerr, y=mean_recovered_planet_occurrences)
print("opt params: ", opt_params)

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

#sampler.run(jax.random.PRNGKey(0), np.array(zink_kepler['scale_height']), yerr, y=mean_physical_planet_occurrences)
sampler.run(jax.random.PRNGKey(0), np.array(zink_kepler['scale_height']), yerr, y=mean_recovered_planet_occurrences)
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
ax1.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', color='red', alpha=0.5, capsize=3, elinewidth=1, markeredgewidth=1, label='Zink+ 2023 Kepler data')

# our simulated data
#ax1.errorbar(x=zink_kepler['scale_height'], y=mean_physical_planet_occurrences, yerr=yerr, fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, color='#03acb1', alpha=0.5, label='model yield')
ax1.errorbar(x=zink_kepler['scale_height'], y=mean_recovered_planet_occurrences, yerr=yerr, fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, color='#03acb1', alpha=0.5, label='model yield')

#"""
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
print("OUR YIELD: ", our_models)
print(len(our_models))
ax1.fill_between(z_max, our_yield_max, our_yield_min, color='#03acb1', alpha=0.3, label='model best-fit posteriors') 
#"""

ax1.set_xlim([100, 1000])
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

# FOR MONOTONIC MDOEL, PASS AX OBJECT TO UTILS.PLOT_MODELS()
#utils.plot_models(frac1, frac2, ax2)
b = frac1
m = (frac2 - frac1)/(x[-1] - x[0])
y = b + m * x

# piecewise model
m = (frac2 - frac1)/(x[-1] - threshold)
y = np.where(x < threshold, frac1, frac1 + m * (x-threshold))

ax2.plot(x, y, color='powderblue')
ax2.set_xlabel('cosmic age [Gyr]')
ax2.set_ylabel('planet host fraction')
ax2.set_ylim([0,1])

fig.tight_layout()
#plt.savefig(path+'plots/model_vs_zink_'+name+'_recovered.png', format='png', bbox_inches='tight')
plt.show()