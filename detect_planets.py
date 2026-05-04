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

from itertools import zip_longest
import numpy.ma as ma # for masked arrays

from astropy.table import Table, join

from psps.transit_class import Population, Star, GeneralStar
import psps.simulate_helpers as simulate_helpers
import psps.simulate_transit as simulate_transit
import psps.utils as utils

# these packages are for fitting with numpyro
import numpyro
from numpyro import distributions as dist, infer
import numpyro_ext
import arviz as az
import jax

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
#path = '/home/c.lam/blue/psps/'

hu25_b20_kepler_b25_k2_kepmag = pd.read_csv(path+'data/joint/hu25_b20_kepler_b25_k2.csv')

def literal_eval_w_exceptions(x):
    try:
        return literal_eval(str(x))   
    except Exception as e:
        pass

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

# planet formation history model parameters
threshold = 11 # cosmic age in Gyr; 13.7 minus stellar age, then round
frac1 = 0.33 # frac1 must be < frac2 if comparing cosmic ages
frac2 = 0.33

name_thresh = 11
name_f1 = 33
name_f2 = 33
name = 'step_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2)
#name = 'monotonic_'+str(name_f1)+'_'+str(name_f2)
#name = 'piecewise_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2)

sim = sorted(glob(path+'data/joint/stellar_samples/' + name + '/' + name + '*'))

# define grids
period_grid = np.logspace(np.log10(1), np.log10(40), 10)
radius_grid = np.linspace(1, 4, 10)
period_grid_k2 = np.array([0.5, 5, 10, 20, 30, 40])
radius_grid_k2 = np.array([0,2,4])
height_bins = np.logspace(2, 3, 6) # ah, so the above are the midpoints of the actual bins they used, I guess
height_bin_midpoints = 0.5 * (np.logspace(2,3,6)[1:] + np.logspace(2,3,6)[:-1])
#print("height bins: ", height_bins)

# load sensitivity maps, as calculated in build_completeness.py
kepler_sensitivity = np.load(path+'data/joint/kepler_sensitivity.npy')
k2_sensitivity = np.load(path+'data/joint/k2_sensitivity.npy')
kepler_sensitivity_list = np.load(path+'data/joint/kepler_sensitivity_list.npy')
k2_sensitivity_list = np.load(path+'data/joint/k2_sensitivity_list.npy')
kepler_sensitivity_height_list = np.array([0.932, 0.928, 0.935, 0.933, 0.934])
k2_sensitivity_height_list = np.array([0.306, 0.296, 0.293, 0.284, 0.277])

# Kepler completeness: Fig 10 in https://pure-oai.bham.ac.uk/ws/files/54320976/Thompson_et_al_Planetary_candidates_observed_The_Astrophysical_Journal_Supplement_Series_2018.pdf
pmesh, rmesh = np.meshgrid(period_grid, radius_grid, indexing='ij')
kepler_completeness = np.full_like(pmesh, 1., dtype=float)  
kepler_completeness[(rmesh > 2) & (pmesh > 5)] = 0.90
kepler_completeness[(rmesh <= 2) & (pmesh <= 5)] = 0.96
kepler_completeness[(rmesh <= 2) & (pmesh > 5)] = 0.93

# Kepler reliability: Fig 11 in https://pure-oai.bham.ac.uk/ws/files/54320976/Thompson_et_al_Planetary_candidates_observed_The_Astrophysical_Journal_Supplement_Series_2018.pdf
pmesh, rmesh = np.meshgrid(period_grid, radius_grid, indexing='ij')
kepler_reliability = np.full_like(pmesh, 1., dtype=float)  
kepler_reliability[rmesh <= 2] = 0.98
kepler_reliability[(rmesh > 2) & (pmesh > 5)] = 0.99
#print(kepler_reliability)

# K2 reliability: https://iopscience.iop.org/article/10.3847/1538-3881/ac2309#ajac2309s7, Fig 13
pmesh_k2, rmesh_k2 = np.meshgrid(period_grid_k2, radius_grid_k2, indexing='ij')
k2_reliability = np.full_like(pmesh_k2, 1., dtype=float)
k2_reliability[(rmesh_k2 > 2) & (pmesh_k2 <= 5)] = 0.98
k2_reliability[(rmesh_k2 > 2) & (pmesh_k2 > 5) & (pmesh_k2 <= 10)] = 0.96
k2_reliability[(rmesh_k2 > 2) & (pmesh_k2 > 10) & (pmesh_k2 <= 20)] = 0.96
k2_reliability[(rmesh_k2 > 2) & (pmesh_k2 > 20) & (pmesh_k2 <= 30)] = 0.85
k2_reliability[(rmesh_k2 > 2) & (pmesh_k2 > 30) & (pmesh_k2 <= 40)] = 0.88
k2_reliability[(rmesh_k2 <= 2) & (pmesh_k2 <= 5)] = 0.99
k2_reliability[(rmesh_k2 <= 2) & (pmesh_k2 > 5) & (pmesh_k2 <= 10)] = 0.94
k2_reliability[(rmesh_k2 <= 2) & (pmesh_k2 > 10) & (pmesh_k2 <= 20)] = 0.86
k2_reliability[(rmesh_k2 <= 2) & (pmesh_k2 > 20) & (pmesh_k2 <= 30)] = 0.87
k2_reliability[(rmesh_k2 <= 2) & (pmesh_k2 > 30) & (pmesh_k2 <= 40)] = 0.75

def prob_geom_transit(a, R_star, R_planet):
    """Calculate geometric transit probability.
    Assume, for this study of close-in planets, that eccentricity is zero.

    Args:
        a (float): semi-major axis in AU
        R_star (float): stellar radius in solar radii
        R_planet (float): planetary radius in solar radii
    Returns:
        float: geometric transit probability
    """
    return (simulate_helpers.solar_radius_to_au(R_star) + simulate_helpers.earth_radius_to_au(R_planet)) / a 

def bin_and_count(df, pgrid, rgrid):
    r = df['planet_radii'].values
    p = df['periods'].values
    H, _, _ = np.histogram2d(p, r, bins=[pgrid, rgrid])

    return H  

detected_se_list = []
detected_sn_list = []
physical_se_list = []
physical_sn_list = []
adjusted_se_list = []
adjusted_sn_list = []
kepler_stars_list = []
k2_stars_list = []
adjusted_se_kepler_list = []
adjusted_sn_kepler_list = []
adjusted_se_k2_list = []
adjusted_sn_k2_list = []
# calculate detected planet yields for each synthetic stellar-planetary population
for i in tqdm(range(len(sim))):

    berger_kepler_all = pd.read_csv(sim[i], sep=',') #, on_bad_lines='skip'
    #berger_kepler_all = pd.read_csv(path+'data/berger_gala/'+name+'.csv')
    
    num_hosts = berger_kepler_all.loc[berger_kepler_all['num_planets']>0]
    #print("f: ", len(num_hosts)/len(berger_kepler_all))
    f = len(num_hosts)/len(berger_kepler_all)

    berger_kepler_all = berger_kepler_all.dropna(subset=['height'])

    berger_kepler_all['height'] = berger_kepler_all['height'] * 1000 # kpc --> pc
    berger_kepler_all['periods'] = berger_kepler_all['periods'].apply(literal_eval_w_exceptions)
    berger_kepler_all['planet_radii'] = berger_kepler_all['planet_radii'].apply(literal_eval_w_exceptions)
    berger_kepler_all['incls'] = berger_kepler_all['incls'].apply(literal_eval_w_exceptions)
    berger_kepler_all['mutual_incls'] = berger_kepler_all['mutual_incls'].apply(literal_eval_w_exceptions)
    berger_kepler_all['eccs'] = berger_kepler_all['eccs'].apply(literal_eval_w_exceptions)
    berger_kepler_all['omegas'] = berger_kepler_all['omegas'].apply(literal_eval_w_exceptions)
    berger_kepler_all['se_or_sn'] = berger_kepler_all['se_or_sn'].str.replace(' ', ', ')
    berger_kepler_all['se_or_sn'] = berger_kepler_all['se_or_sn'].apply(literal_eval_w_exceptions)

    # assign height bins so we can split
    berger_kepler_all['height_bin'] = pd.cut(berger_kepler_all['height'], bins=height_bins, labels=False)
    berger_kepler_all = berger_kepler_all.dropna(subset=['height_bin'])
    berger_kepler_all['height_bin'] = berger_kepler_all['height_bin'].astype(int)

    detected_se = []
    detected_sn = []
    physical_se = []
    physical_sn = []
    adjusted_se = []
    adjusted_sn = []
    kepler_stars = []
    k2_stars = []
    detected_se_kepler = []
    detected_sn_kepler = []
    physical_se_kepler = []
    physical_sn_kepler = []
    adjusted_se_kepler = []
    adjusted_sn_kepler = []
    detected_se_k2 = []
    detected_sn_k2 = []
    physical_se_k2 = []
    physical_sn_k2 = []
    adjusted_se_k2 = []
    adjusted_sn_k2 = []
    for i in range(len(height_bins[:-1])):	
        # isolate by height bin
        temp = berger_kepler_all.loc[berger_kepler_all['height_bin']==i]

        # split into Kepler and K2, since they have different detection efficiencies (bc different baselines and different pointings)
        kepler_temp = temp.loc[temp['Kepler_ID'] > 0]
        k2_temp = temp.loc[temp['EPIC_ID'] > 0]
        kepler_stars.append(len(kepler_temp))
        k2_stars.append(len(k2_temp))

        # denominators
        temp['height_bins'] = pd.cut(temp['height'], bins=height_bins, include_lowest=True)
        berger_kepler_counts = np.array(temp.groupby(['height_bins']).count().reset_index()['GaiaDR3'])

        kepler_temp['height_bins'] = pd.cut(kepler_temp['height'], bins=height_bins, include_lowest=True)
        kepler_temp_counts = np.array(kepler_temp.groupby(['height_bins']).count().reset_index()['GaiaDR3'])

        k2_temp['height_bins'] = pd.cut(k2_temp['height'], bins=height_bins, include_lowest=True)
        k2_temp_counts = np.array(k2_temp.groupby(['height_bins']).count().reset_index()['GaiaDR3'])

        ### isolate planet hosts
        berger_kepler_planets = temp.loc[temp['num_planets'] > 0]
        kepler_planets_temp = kepler_temp.loc[kepler_temp['num_planets'] > 0]
        k2_planets_temp = k2_temp.loc[k2_temp['num_planets'] > 0]

        ### EXPLODE
        berger_kepler_planets = berger_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas', 'se_or_sn']).reset_index(drop=True)
        kepler_planets_temp_explode = kepler_planets_temp.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas', 'se_or_sn']).reset_index(drop=True)
        k2_planets_temp_explode = k2_planets_temp.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas', 'se_or_sn']).reset_index(drop=True)

        ### select planets that are relevant to this study 
        berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
        berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 4.] # limit radii to fairly compare with SEs in Zink+ 2023 (2)...or how about include SNs too (4)?
        kepler_planets_temp_explode = kepler_planets_temp_explode.loc[(kepler_planets_temp_explode['periods'] <= 40) & (kepler_planets_temp_explode['periods'] > 1)] 
        kepler_planets_temp_explode = kepler_planets_temp_explode.loc[kepler_planets_temp_explode['planet_radii'] <= 4.] 
        k2_planets_temp_explode = k2_planets_temp_explode.loc[(k2_planets_temp_explode['periods'] <= 40) & (k2_planets_temp_explode['periods'] > 1)] 
        k2_planets_temp_explode = k2_planets_temp_explode.loc[k2_planets_temp_explode['planet_radii'] <= 4.] 
        
        ### split into SEs and SNs
        berger_kepler_se = berger_kepler_planets.loc[berger_kepler_planets['se_or_sn']=='se']
        berger_kepler_sn = berger_kepler_planets.loc[berger_kepler_planets['se_or_sn']=='sn']
        kepler_planets_se = kepler_planets_temp_explode.loc[kepler_planets_temp_explode['se_or_sn']=='se']
        kepler_planets_sn = kepler_planets_temp_explode.loc[kepler_planets_temp_explode['se_or_sn']=='sn']
        k2_planets_se = k2_planets_temp_explode.loc[k2_planets_temp_explode['se_or_sn']=='se']
        k2_planets_sn = k2_planets_temp_explode.loc[k2_planets_temp_explode['se_or_sn']=='sn']

        #plt.hist(berger_kepler_se.loc[berger_kepler_se['num_planets']>2]['mutual_incls'], density=True)
        #plt.hist(berger_kepler_se.loc[berger_kepler_se['num_planets']<2]['mutual_incls'], density=True)
        #plt.show()

        ### Simulate detections from these synthetic systems
        _, transit_statuses_kepler_se, _, geom_transit_statuses_kepler_se = simulate_transit.kepler_detection(kepler_planets_se, angle_flag=True) 
        kepler_planets_se['transit_status'] = transit_statuses_kepler_se[0]
        kepler_planets_se['geom_transit_status'] = geom_transit_statuses_kepler_se
        _, transit_statuses_kepler_sn, _, geom_transit_statuses_kepler_sn = simulate_transit.kepler_detection(kepler_planets_sn, angle_flag=True) 
        kepler_planets_sn['transit_status'] = transit_statuses_kepler_sn[0]
        kepler_planets_sn['geom_transit_status'] = geom_transit_statuses_kepler_sn
        transit_statuses_k2_se, _, _, geom_transit_statuses_k2_se = simulate_transit.k2_detection(k2_planets_se, angle_flag=True) 
        k2_planets_se['transit_status'] = transit_statuses_k2_se
        k2_planets_se['geom_transit_status'] = geom_transit_statuses_k2_se
        transit_statuses_k2_sn, _, _, geom_transit_statuses_k2_sn = simulate_transit.k2_detection(k2_planets_sn, angle_flag=True) 
        k2_planets_sn['transit_status'] = transit_statuses_k2_sn
        k2_planets_sn['geom_transit_status'] = geom_transit_statuses_k2_sn

        kepler_detected_se = kepler_planets_se.loc[kepler_planets_se['transit_status']==1]
        kepler_detected_sn = kepler_planets_sn.loc[kepler_planets_sn['transit_status']==1]
        k2_detected_se = k2_planets_se.loc[k2_planets_se['transit_status']==1]
        k2_detected_sn = k2_planets_sn.loc[k2_planets_sn['transit_status']==1]

        kepler_geom_transiters_se = kepler_planets_se.loc[kepler_planets_se['geom_transit_status']==1]
        kepler_geom_transiters_sn = kepler_planets_sn.loc[kepler_planets_sn['geom_transit_status']==1]
        k2_geom_transiters_se = k2_planets_se.loc[k2_planets_se['geom_transit_status']==1]
        k2_geom_transiters_sn = k2_planets_sn.loc[k2_planets_sn['geom_transit_status']==1]
        print("height: ", height_bins[i], "Kepler stars: ", len(kepler_temp), "K2 stars: ", len(k2_temp), "Kepler planets: ", len(kepler_planets_temp_explode), "K2 planets: ", len(k2_planets_temp_explode), "Kepler detected planets: ", len(kepler_detected_se)+len(kepler_detected_sn), "K2 detected planets: ", len(k2_detected_se)+len(k2_detected_sn))
        
        """
        print("Kepler geometric transit detection efficiency, SEs: ", len(kepler_geom_transiters_se)/len(kepler_planets_se))
        print("Kepler geometric transit detection efficiency, SNs: ", len(kepler_geom_transiters_sn)/len(kepler_planets_sn))
        try:
            print("K2 geometric transit detection efficiency, SEs: ", len(k2_geom_transiters_se)/len(k2_planets_se))
        except:
            print("no K2 SEs in this height bin")
        try:
            print("K2 geometric transit detection efficiency, SNs: ", len(k2_geom_transiters_sn)/len(k2_planets_sn))
        except:
            print("no K2 SNs in this height bin")
        """

        # Check whether angle_flag is working as intended
        #intact_se = kepler_planets_se.loc[kepler_planets_se['status']=='intact']
        #disrupted_se = kepler_planets_se.loc[kepler_planets_se['status']=='disrupted']
        #print("SE intact: ", len(intact_se.loc[intact_se['transit_status']==1])/len(intact_se))
        #print("SE disrupted: ", len(disrupted_se.loc[disrupted_se['transit_status']==1])/len(disrupted_se))
        #intact_sn = kepler_planets_sn.loc[kepler_planets_sn['status']=='intact']
        #disrupted_sn = kepler_planets_sn.loc[kepler_planets_sn['status']=='disrupted']

        ### Put Kepler and K2 together for completeness and reliability calculations
        kepler_detected = pd.concat([kepler_detected_se, kepler_detected_sn])
        k2_detected = pd.concat([k2_detected_se, k2_detected_sn])
        kepler_planets = pd.concat([kepler_planets_se, kepler_planets_sn])
        k2_planets = pd.concat([k2_planets_se, k2_planets_sn])
        kepler_geom_transiters = pd.concat([kepler_geom_transiters_se, kepler_geom_transiters_sn])
        k2_geom_transiters = pd.concat([k2_geom_transiters_se, k2_geom_transiters_sn])
        #print("Kepler detected planets: ", len(kepler_detected), "K2 detected planets: ", len(k2_detected))
        #print("")
        detected_se.append(len(kepler_detected_se)+len(k2_detected_se))
        detected_sn.append(len(kepler_detected_sn)+len(k2_detected_sn))
        physical_se.append(len(kepler_planets_se)+len(k2_planets_se))
        physical_sn.append(len(kepler_planets_sn)+len(k2_planets_sn))

        ### ADJUST
        ### old way: don't consider reliability; completeness is literally just the inverse of the detection efficiency, so this just captures Poisson noise

        ### new way: use reliability from literature, sensitivity from injection/recovery tests, and geometric completeness from either analytic formula or empirically
        kepler_se_geom_completeness = len(prob_geom_transit(simulate_helpers.p_to_a(kepler_detected_se['periods'], kepler_detected_se['stellar_mass']), kepler_detected_se['stellar_radius'], kepler_detected_se['planet_radii']))/len(kepler_planets_se)
        kepler_sn_geom_completeness = len(prob_geom_transit(simulate_helpers.p_to_a(kepler_detected_sn['periods'], kepler_detected_sn['stellar_mass']), kepler_detected_sn['stellar_radius'], kepler_detected_sn['planet_radii']))/len(kepler_planets_sn)
        try:
            k2_se_geom_completeness = len(prob_geom_transit(simulate_helpers.p_to_a(k2_detected_se['periods'], k2_detected_se['stellar_mass']), k2_detected_se['stellar_radius'], k2_detected_se['planet_radii']))/len(k2_planets_se)
        except: # if no K2 planets, doesn't matter what geometric completeness is because 0 times anything is 0. So I can set to 1 to avoid dividing by 0. 
            k2_se_geom_completeness = 1
        try:
            k2_sn_geom_completeness = len(prob_geom_transit(simulate_helpers.p_to_a(k2_detected_sn['periods'], k2_detected_sn['stellar_mass']), k2_detected_sn['stellar_radius'], k2_detected_sn['planet_radii']))/len(k2_planets_sn)
        except:
            k2_sn_geom_completeness = 1

        # reliability
        kepler_detected_se_binned_reliability_adjusted = bin_and_count(kepler_detected_se, period_grid, radius_grid) * kepler_reliability[:-1, :-1]
        kepler_detected_sn_binned_reliability_adjusted = bin_and_count(kepler_detected_sn, period_grid, radius_grid) * kepler_reliability[:-1, :-1]
        k2_detected_se_binned_reliability_adjusted = bin_and_count(k2_detected_se, period_grid_k2, radius_grid_k2) * k2_reliability[:-1, :-1]
        k2_detected_sn_binned_reliability_adjusted = bin_and_count(k2_detected_sn, period_grid_k2, radius_grid_k2) * k2_reliability[:-1, :-1]

        kepler_adjusted_se = np.sum(kepler_detected_se_binned_reliability_adjusted) / (kepler_sensitivity_height_list[i] * kepler_se_geom_completeness)
        kepler_adjusted_sn = np.sum(kepler_detected_sn_binned_reliability_adjusted) / (kepler_sensitivity_height_list[i] * kepler_sn_geom_completeness)
        k2_adjusted_se = np.sum(k2_detected_se_binned_reliability_adjusted) / (k2_sensitivity_height_list[i] * k2_se_geom_completeness)
        k2_adjusted_sn = np.sum(k2_detected_sn_binned_reliability_adjusted) / (k2_sensitivity_height_list[i] * k2_sn_geom_completeness)

        adjusted_se_kepler.append(kepler_adjusted_se)
        adjusted_sn_kepler.append(kepler_adjusted_sn)
        adjusted_se_k2.append(k2_adjusted_se)
        adjusted_sn_k2.append(k2_adjusted_sn)

        if k2_adjusted_se > 0: 
            adjusted_se.append(kepler_adjusted_se + k2_adjusted_se)
        else: # if there weren't detected K2 planets
            adjusted_se.append(kepler_adjusted_se)
        if k2_adjusted_sn > 0:
            adjusted_sn.append(kepler_adjusted_sn + k2_adjusted_sn)
        else:
            adjusted_sn.append(kepler_adjusted_sn)

    detected_se_list.append(detected_se)
    detected_sn_list.append(detected_sn)
    physical_se_list.append(physical_se)
    physical_sn_list.append(physical_sn)
    adjusted_se_kepler_list.append(adjusted_se_kepler)
    adjusted_sn_kepler_list.append(adjusted_sn_kepler)
    adjusted_se_k2_list.append(adjusted_se_k2)
    adjusted_sn_k2_list.append(adjusted_sn_k2)
    kepler_stars_list.append(kepler_stars)
    k2_stars_list.append(k2_stars)
    adjusted_se_list.append(adjusted_se)
    adjusted_sn_list.append(adjusted_sn)

denominators = np.mean(kepler_stars_list, axis=0) + np.mean(k2_stars_list, axis=0)
kepler_denominators = np.mean(kepler_stars_list, axis=0)
k2_denominators = np.mean(k2_stars_list, axis=0)
print("detected SEs: ", np.mean(detected_se_list, axis=0)/denominators)
print("detected SNs: ", np.mean(detected_sn_list, axis=0)/denominators)
print("physical SEs: ", np.mean(physical_se_list, axis=0)/denominators)
print("physical SNs: ", np.mean(physical_sn_list, axis=0)/denominators)
print("adjusted SEs: ", np.mean(adjusted_se_list, axis=0)/denominators)
print("adjusted SNs: ", np.mean(adjusted_sn_list, axis=0)/denominators)

print("adjusted Kepler SEs: ", np.mean(adjusted_se_kepler_list, axis=0)/kepler_denominators)
print("adjusted K2 SEs: ", np.mean(adjusted_se_k2_list, axis=0)/k2_denominators)

### PLOTTING
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
left, bottom, width, height = [0.16, 0.25, 0.25, 0.25]
ax2 = fig.add_axes([left, bottom, width, height])
z_max = np.logspace(2, 3.02, 100)
height_bin_midpoints1 = 0.5 * (np.logspace(2.02,3.02,6)[1:] + np.logspace(2.02,3.02,6)[:-1])
def model(x, tau, occurrence):

    dln = 0.0011
    scaleMax= 1000
    scaleMin = 100
    const = (scaleMax)**(tau+1)/(tau+1) - ((scaleMin)**(tau+1)/(tau+1))
    planet_yield = occurrence * x**(tau)/const/dln * 100
    
    return planet_yield

def power_model(x, yerr, y=None):

    tau = numpyro.sample("tau", dist.Uniform(-1., 1.))
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
    #quit()
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
opt_params = run_optim(jax.random.PRNGKey(5), height_bin_midpoints, 100*np.std(adjusted_se_list, axis=0)/(kepler_denominators+k2_denominators), y=100*np.mean(adjusted_se_list, axis=0)/(kepler_denominators+k2_denominators))
#print("opt params: ", opt_params)

# sample posteriors for best-fit model to simulated data
sampler = infer.MCMC(
    infer.NUTS(power_model, dense_mass=True,
        regularize_mass_matrix=False,
        init_strategy=numpyro.infer.init_to_value(values=opt_params)), 
    num_warmup=5000,
    num_samples=10000,
    num_chains=4,
    progress_bar=True,
)

print("yerr: ", 100*np.std(adjusted_se_list, axis=0)/(kepler_denominators+k2_denominators))
print("y: ", 100*np.mean(adjusted_se_list, axis=0)/(kepler_denominators+k2_denominators))
sampler.run(jax.random.PRNGKey(0), height_bin_midpoints, 100*np.std(adjusted_se_list, axis=0)/(kepler_denominators+k2_denominators), y=100*np.mean(adjusted_se_list, axis=0)/(kepler_denominators+k2_denominators))
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
sum_model = zink_csv['model'] #+ zink_csv_sn['model']
for temp_list in zip_longest(*sum_model):
    yield_max.append(np.percentile(temp_list, 84)) # plus one sigma
    yield_min.append(np.percentile(temp_list, 16)) # minus one sigma
ax1.fill_between(z_max, yield_max, yield_min, color='red', alpha=0.3, label='Zink+ 2023 posteriors') #03acb1

zink_sn_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([38, 29, 23, 24, 17]), 'occurrence_err1': np.array([5, 3, 2, 2, 4]), 'occurrence_err2': np.array([6, 3, 2, 4, 4])})
zink_se_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([28, 29, 25, 27, 18]), 'occurrence_err1': np.array([5, 3, 3, 4, 4]), 'occurrence_err2': np.array([5, 3, 3, 3, 4])})
zink_kepler_occurrence = np.array([38, 29, 23, 24, 17])+np.array([28, 29, 25, 27, 18])
zink_kepler_occurrence_err1 = np.round(np.sqrt((zink_sn_kepler['occurrence_err1'])**2 + (zink_se_kepler['occurrence_err1']**2)), 2)
zink_kepler_occurrence_err2 = np.round(np.sqrt((zink_sn_kepler['occurrence_err2'])**2 + (zink_se_kepler['occurrence_err2']**2)), 2)
zink_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': zink_kepler_occurrence, 'occurrence_err1': zink_kepler_occurrence_err1, 'occurrence_err2': zink_kepler_occurrence_err2})

# zink+ 2023 data
ax1.errorbar(x=height_bin_midpoints, y=zink_se_kepler['occurrence'], yerr=(zink_se_kepler['occurrence_err1'], zink_se_kepler['occurrence_err2']), fmt='o', color='red', alpha=0.5, capsize=3, elinewidth=1, markeredgewidth=1, label='Zink+ 2023 Kepler data')

# our physical data
se_yield_max = 100 * np.nanmax(physical_se_list, axis=0) / (kepler_denominators+k2_denominators)
se_yield_min = 100 * np.nanmin(physical_se_list, axis=0) / (kepler_denominators+k2_denominators)
mean_physical_planet_occurrences = 100 * np.mean(physical_se_list, axis=0) / (kepler_denominators+k2_denominators)
std_physical_planet_occurrences = 100 * np.std(physical_se_list, axis=0) / (kepler_denominators+k2_denominators)
ax1.errorbar(x=height_bin_midpoints1, y=mean_physical_planet_occurrences, yerr=std_physical_planet_occurrences, fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, color='purple', alpha=0.5, label='model physical yield')
#ax1.fill_between(z_max, se_yield_max, se_yield_min, color='#03acb1', alpha=0.3, label='model best-fit posteriors') 

# our adjusted data
mean_adjusted_planet_occurrences = 100 * np.mean(adjusted_se_list, axis=0) / (kepler_denominators+k2_denominators)
std_adjusted_planet_occurrences = 100 * np.std(adjusted_se_list, axis=0) / (kepler_denominators+k2_denominators)
print("mean adjusted planet occurrences: ", mean_adjusted_planet_occurrences)
print("std adjusted planet occurrences: ", std_adjusted_planet_occurrences)
ax1.errorbar(x=height_bin_midpoints1, y=mean_adjusted_planet_occurrences, yerr=std_adjusted_planet_occurrences, fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, color='blue', alpha=0.5, label='model adjusted yield')
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
ax1.fill_between(z_max, our_yield_max, our_yield_min, color='#03acb1', alpha=0.3, label='best-fit model') 

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
#x = np.linspace(0, 14, 1000)
#y = np.where(x <= threshold, frac1, frac2)
x = np.linspace(14, 0, 1000)
y = np.where(x <= 13.7 - threshold, frac2, frac1)

# monotonic model
#b = frac1
#m = (frac2 - frac1)/(x[-1] - x[0])
#y = b + m * x

# piecewise model
#m = (frac2 - frac1)/(x[-1] - threshold)
#y = np.where(x < threshold, frac1, frac1 + m * (x-threshold))

ax2.plot(x, y, color='powderblue', linewidth=2)
ax2.set_xlabel('lookback time [Gyr]')
ax2.set_ylabel('planet host fraction')
ax2.set_ylim([0,1])
fig.tight_layout()
plt.show()

quit()




### ALL THIS USED TO BE INSIDE THE LOOP

### Adjust based on geometric transit, sensitivity, and reliability.
# no relation between period and height, which means reliability can be applied as a flat correction
# apply completeness (sensitivity times geometric transit completeness) over height, rather than period (the old way)
kepler_planets_height_binned, _ = np.histogram(kepler_planets['height'], bins=height_bins)
kepler_detected_height_binned, _ = np.histogram(kepler_detected_se['height'], bins=height_bins)
completeness_height_binned = kepler_detected_height_binned / kepler_planets_height_binned
print("Kepler completeness by height: ", completeness_height_binned)

def _bin_and_count(df, pgrid, rgrid):
    r = df['planet_radii'].values
    p = df['periods'].values
    H, _, _ = np.histogram2d(p, r, bins=[pgrid, rgrid])

    return H  

### What is the geometric transit detection efficiency? 
# Since injected planets are subject only to the randomly oriented midplane (since mutual incl is chosen to be 0), only period matters
# In practice, geometric completeness depends on the model (and nature's preferred) mutual inclination distribution.
# But! For this study, we assume that the mutual inclination distribution is the same across all heights, so the geometric transit detection efficiency should be the same across all heights, and thus not bias our results.
# So, we use the formula above, (R_star + R_planet)/a, to analytically back out the expected value of true planets. 
#print(len(kepler_geom_transiters_se)/len(kepler_planets_se))
#print(len(kepler_geom_transiters_sn)/len(kepler_planets_sn))
#print(len(k2_geom_transiters_se)/len(k2_planets_se))
#print(len(k2_geom_transiters_sn)/len(k2_planets_sn))
kepler_planets['prob_transit'] = prob_geom_transit(simulate_helpers.p_to_a(kepler_planets['periods'], kepler_planets['stellar_mass']), kepler_planets['stellar_radius'], kepler_planets['planet_radii'])
period_bins = pd.cut(kepler_planets['periods'], bins=period_grid)
radius_bins = pd.cut(kepler_planets['planet_radii'], bins=radius_grid)
prob_geom_transit_kepler =(
kepler_planets.groupby([period_bins, radius_bins])['prob_transit']
.mean()
.unstack()
).to_numpy()

# average P(transit) across radius bins, since geometric transit probability doesn't depend strongly on radius
# use this for both Kepler and K2, under the assumption that their stellar radius distributions are similar
prob_geom_transit_kepler_row_means = prob_geom_transit_kepler.mean(axis=1, keepdims=True)
kepler_geom_transit_completeness = np.repeat(prob_geom_transit_kepler_row_means, prob_geom_transit_kepler.shape[1], axis=1)
#print("Kepler geometric transit completeness: ", kepler_geom_transit_completeness)

# empirical geometric transit completeness: painting it on from Winn+10 analytically assumes perfectly coplanar orbits, which is not an assumption I'm willing to relax, given the spread of architectures I've intentionally painted onto our systems
kepler_geom_transit_completeness_numerator, _ = np.histogram(kepler_geom_transiters['periods'].values, bins=period_grid) 
kepler_geom_transit_completeness_denominator, _ = np.histogram(kepler_planets['periods'].values, bins=period_grid)
kepler_geom_transit_completeness = kepler_geom_transit_completeness_numerator / kepler_geom_transit_completeness_denominator
#print("Kepler geometric transit completeness: ", kepler_geom_transit_completeness)
k2_geom_transit_completeness_numerator, _ = np.histogram(k2_geom_transiters['periods'].values, bins=period_grid)
k2_geom_transit_completeness_denominator, _ = np.histogram(k2_planets['periods'].values, bins=period_grid)
k2_geom_transit_completeness = k2_geom_transit_completeness_numerator / k2_geom_transit_completeness_denominator
print("K2 geometric transit completeness: ", k2_geom_transit_completeness)

kepler_planets_se_binned = _bin_and_count(kepler_planets_se, period_grid, radius_grid)
kepler_planets_sn_binned = _bin_and_count(kepler_planets_sn, period_grid, radius_grid)
kepler_geom_transiters_se_binned = _bin_and_count(kepler_geom_transiters_se, period_grid, radius_grid) # swap out grids depending on sensitivity or reliability map
kepler_geom_transiters_sn_binned = _bin_and_count(kepler_geom_transiters_sn, period_grid, radius_grid) 
kepler_detected_se_binned = _bin_and_count(kepler_detected_se, period_grid, radius_grid)
kepler_detected_sn_binned = _bin_and_count(kepler_detected_sn, period_grid, radius_grid)

# kepler_planets_binned = kepler_planets_se_binned + kepler_planets_sn_binned
# kepler_geom_transiters_binned = kepler_geom_transiters_se_binned + kepler_geom_transiters_sn_binned
# kepler_geom_transit_completeness = kepler_geom_transiters_binned / kepler_planets_binned
# print("Kepler geometric transit completeness: ", kepler_geom_transit_completeness)
print("painted-on reliability, sensitivity, and geometric transit completeness: ", kepler_reliability[1:, 1:].mean(), kepler_sensitivity[1:, 1:].mean(), kepler_geom_transit_completeness.mean())
print("empirical sensitivity, given geometric transit: ", (kepler_detected_se_binned.sum() + kepler_detected_sn_binned.sum())/(kepler_geom_transiters_se_binned.sum() + kepler_geom_transiters_sn_binned.sum()))
print("empirical geometric transit completeness: ", (kepler_geom_transiters_se_binned.sum() + kepler_geom_transiters_sn_binned.sum())/(kepler_planets_se_binned.sum() + kepler_planets_sn_binned.sum()))

kepler_adjusted_se = kepler_detected_se_binned * kepler_reliability[1:, 1:] / (kepler_sensitivity[1:, 1:] * kepler_geom_transit_completeness) 
kepler_adjusted_sn = kepler_detected_sn_binned * kepler_reliability[1:, 1:] / (kepler_sensitivity[1:, 1:] * kepler_geom_transit_completeness)
kepler_adjusted = kepler_adjusted_se + kepler_adjusted_sn
kepler_detected_binned = kepler_detected_se_binned + kepler_detected_sn_binned
kepler_geom_transiters_binned = kepler_geom_transiters_se_binned + kepler_geom_transiters_sn_binned
print("Kepler sensitivity map: ", kepler_sensitivity[1:, 1:])
print("detected Kepler planets: ", kepler_detected_se_binned + kepler_detected_sn_binned)
print("reliability andsensitivity-adjusted detected Kepler planets: ", (kepler_detected_se_binned+kepler_detected_sn_binned) * kepler_reliability[1:, 1:] / kepler_sensitivity[1:, 1:])
print("Kepler geometric transit probability map: ", kepler_geom_transit_completeness)
print("Kepler completeness-adjusted occurrence: ", kepler_adjusted)
print("Kepler occurrence rate adjusted for geometric transit and sensitivity: ", kepler_adjusted.sum()/kepler_temp_counts.sum())
print("TRUE Kepler occurrence rate: ", (kepler_planets_se_binned + kepler_planets_sn_binned).sum()/kepler_temp_counts.sum())

empirical_sensitivity = kepler_detected_binned.sum() / kepler_geom_transiters_binned.sum()
print("Empirical sensitivity: ", empirical_sensitivity)
print("sensitivity from injection test: ", kepler_sensitivity[1:, 1:].mean())
print("Thompson+18 sensitivity: ", kepler_completeness.mean())
print("")

### bin by height
kepler_planets_height_binned, _ = np.histogram(kepler_planets['height'], bins=height_bins)
kepler_detected_height_binned, _ = np.histogram(kepler_detected_se['height'], bins=height_bins)
kepler_temp_height_binned, _ = np.histogram(kepler_temp['height'], bins=height_bins)
print("Kepler detected occurrence rate by height: ", 100 * kepler_detected_height_binned/kepler_temp_height_binned)
print("Kepler true occurrence rate by height: ", 100 * kepler_planets_height_binned/kepler_temp_height_binned)


k2_planets_se_binned = _bin_and_count(k2_planets_se, period_grid_k2, radius_grid_k2)
k2_planets_sn_binned = _bin_and_count(k2_planets_sn, period_grid_k2, radius_grid_k2)
k2_geom_transiters_se_binned = _bin_and_count(k2_geom_transiters_se, period_grid_k2, radius_grid_k2) # swap out grids depending on sensitivity or reliability map
k2_geom_transiters_sn_binned = _bin_and_count(k2_geom_transiters_sn, period_grid_k2, radius_grid_k2) 
k2_detected_se_binned = _bin_and_count(k2_detected_se, period_grid_k2, radius_grid_k2)
k2_detected_sn_binned = _bin_and_count(k2_detected_sn, period_grid_k2, radius_grid_k2)
#print(k2_planets_se_binned)
#print(k2_planets_sn_binned)
#print(k2_detected_se_binned)
#print(k2_detected_sn_binned)

k2_planets['prob_transit'] = prob_geom_transit(simulate_helpers.p_to_a(k2_planets['periods'], k2_planets['stellar_mass']), k2_planets['stellar_radius'], k2_planets['planet_radii'])
period_bins_k2 = pd.cut(k2_planets['periods'], bins=period_grid_k2)
radius_bins_k2 = pd.cut(k2_planets['planet_radii'], bins=radius_grid_k2)
prob_geom_transit_k2 =(
k2_planets.groupby([period_bins_k2, radius_bins_k2])['prob_transit']
.mean()
.unstack()
).to_numpy()
prob_geom_transit_k2_row_means = prob_geom_transit_k2.mean(axis=1, keepdims=True)
k2_geom_transit_completeness = np.repeat(prob_geom_transit_k2_row_means, prob_geom_transit_k2.shape[1], axis=1)
print("K2 geometric transit completeness: ", k2_geom_transit_completeness)
print(k2_detected_se_binned.shape, k2_reliability[1:, 1:].shape, k2_sensitivity[1:, 1:].shape, k2_geom_transit_completeness.shape)

k2_adjusted_se = k2_detected_se_binned * k2_reliability[1:, 1:] / (k2_sensitivity[1:, 1:] * k2_geom_transit_completeness) 
k2_adjusted_sn = k2_detected_sn_binned * k2_reliability[1:, 1:] / (k2_sensitivity[1:, 1:] * k2_geom_transit_completeness)
k2_adjusted = k2_adjusted_se + k2_adjusted_sn
print("K2 real planets: ", k2_planets_se_binned + k2_planets_sn_binned)
print("K2 sensitivity map: ", k2_sensitivity[1:, 1:])
print("detected K2 planets: ", k2_detected_se_binned + k2_detected_sn_binned)
print("reliability andsensitivity-adjusted detected K2 planets: ", (k2_detected_se_binned+k2_detected_sn_binned)  * k2_reliability[1:, 1:] / k2_sensitivity[1:, 1:])
print("K2 geometric transit probability map: ", k2_geom_transit_completeness)
print("K2 completeness-adjusted occurrence: ", k2_adjusted)
print("K2 occurrence rate adjusted for geometric transit and sensitivity: ", k2_adjusted.sum()/k2_temp_counts.sum())
quit()

#print(_bin_and_count(kepler_planets_se, period_grid, radius_grid))
#print(kepler_geom_transiters_se_binned)
#print(kepler_sensitivity[:-1, :-1])
#print(kepler_geom_transiters_se_binned / kepler_sensitivity[:-1, :-1])
# a subtle distinction: the sensitivity map is 10x10, since each element is evaluated as is, while binning reduces the dim to 9x9
# so, we need to do something like cut off the last row/column of the sensitivity map and live with the slight mismatch (center vs edge)
#kepler_detected_transiters_se = kepler_geom_transiters_se / 

### Compute completeness maps
### WHAT IF I CALCULATED A UNIVERSAL KEPLER AND K2 COMPLETENESS MAP USING THE CONTROL MODEL? Then I can just use that one map for everything
### How would I use it? Detected planets, times reliability, divided by completeness. Is it fair to use a map where 
#print(kepler_planets.loc[(kepler_planets['periods']>=20)&(kepler_planets['planet_radii']<=2.)])
#print(kepler_transiters.loc[(kepler_transiters['periods']>=20)&(kepler_transiters['planet_radii']<=2.)])
#quit()

completeness_map_kepler, piv_physical_kepler, piv_detected_kepler = simulate_helpers.completeness(kepler_planets, kepler_transiters)
completeness_threshold = 0.003 # completeness threshold under which period/radius cell is not counted; 0.5% results in full recovery, but let's round up to 1%
completeness_map_kepler = completeness_map_kepler.mask(completeness_map_kepler < completeness_threshold) # assert that completeness fractions lower than 1% are statistically insignificant
print(completeness_map_kepler)

completeness_map_k2, piv_physical_k2, piv_detected_k2 = simulate_helpers.completeness(k2_planets, k2_transiters)
completeness_threshold = 0.003 # completeness threshold under which period/radius cell is not counted; 0.5% results in full recovery, but let's round up to 1%
completeness_map_k2 = completeness_map_k2.mask(completeness_map_k2 < completeness_threshold) # assert that completeness fractions lower than 1% are statistically insignificant
print(completeness_map_k2)
quit()

### Introduce reliability maps and divide completeness maps from them to get "recovered" occurrence

### count by height bins for raw planets

### count by height bins for detected planets




# count by height bin, for all, Kepler only, and K2 only
berger_kepler_planets_counts = np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['age'])
physical_planet_occurrence = 100 * berger_kepler_planets_counts/berger_kepler_counts 
physical_planet_occurrences_all.append(physical_planet_occurrence)

kepler_temp_planets_counts = np.array(kepler_temp_planets.groupby(['height_bins']).count().reset_index()['age'])
physical_planet_occurrence_kepler = 100 * kepler_temp_planets_counts/kepler_temp_counts 
physical_planet_occurrences_kepler.append(physical_planet_occurrence_kepler)

k2_temp_planets_counts = np.array(k2_temp_planets.groupby(['height_bins']).count().reset_index()['age'])
physical_planet_occurrence_k2 = 100 * k2_temp_planets_counts/k2_temp_counts 
physical_planet_occurrences_k2.append(physical_planet_occurrence_k2)

# do this for SEs and SNs, separately
berger_kepler_se_counts = np.array(berger_kepler_se.groupby(['height_bins']).count().reset_index()['age'])
physical_planet_occurrence_se = 100 * berger_kepler_se_counts/berger_kepler_counts
physical_planet_occurrences_se_all.append(physical_planet_occurrence_se)
berger_kepler_sn_counts = np.array(berger_kepler_sn.groupby(['height_bins']).count().reset_index()['age'])
physical_planet_occurrence_sn = 100 * berger_kepler_sn_counts/berger_kepler_counts
physical_planet_occurrences_sn_all.append(physical_planet_occurrence_sn)
kepler_temp_se_counts = np.array(kepler_temp_se.groupby(['height_bins']).count().reset_index()['age'])
physical_planet_occurrence_se_kepler = 100 * kepler_temp_se_counts/kepler_temp_counts
physical_planet_occurrences_se_kepler.append(physical_planet_occurrence_se_kepler)
kepler_temp_sn_counts = np.array(kepler_temp_sn.groupby(['height_bins']).count().reset_index()['age'])
physical_planet_occurrence_sn_kepler = 100 * kepler_temp_sn_counts/kepler_temp_counts
physical_planet_occurrences_sn_kepler.append(physical_planet_occurrence_sn_kepler)
k2_temp_se_counts = np.array(k2_temp_se.groupby(['height_bins']).count().reset_index()['age'])
physical_planet_occurrence_se_k2 = 100 * k2_temp_se_counts/k2_temp_counts
physical_planet_occurrences_se_k2.append(physical_planet_occurrence_se_k2)
k2_temp_sn_counts = np.array(k2_temp_sn.groupby(['height_bins']).count().reset_index()['age'])
physical_planet_occurrence_sn_k2 = 100 * k2_temp_sn_counts/k2_temp_counts
physical_planet_occurrences_sn_k2.append(physical_planet_occurrence_sn_k2)

### there aren't enough K2 planets to make a straight line
print("Kepler SEs: ", physical_planet_occurrence_se_kepler)
print("Kepler SNs: ", physical_planet_occurrence_sn_kepler)
print("K2 SEs: ", physical_planet_occurrence_se_k2)
print("K2 SNs: ", physical_planet_occurrence_sn_k2)

print("raw Kepler SEs, numerator: ", kepler_temp_se_counts)
print("raw Kepler SNs, numerator: ", kepler_temp_sn_counts)
print("raw K2 SEs, numerator: ", k2_temp_se_counts)
print("raw K2 SNs, numerator: ", k2_temp_sn_counts)

print("raw Kepler denominator: ", kepler_temp_counts)
print("raw K2 denominator: ", k2_temp_counts)

### Simulate detections from these synthetic systems
prob_detections, transit_statuses, sn, geom_transit_statuses = simulate_transit.kepler_detection(berger_kepler_planets, angle_flag=True) 

prob_detections_kepler, transit_statuses_kepler, sn_kepler, geom_transit_statuses_kepler = simulate_transit.calculate_transit_vectorized(kepler_temp_planets.periods, 
                                kepler_temp_planets.stellar_radius, kepler_temp_planets.planet_radii,
                                kepler_temp_planets.eccs, 
                                kepler_temp_planets.incls, 
                                kepler_temp_planets.omegas, kepler_temp_planets.stellar_mass,
                                kepler_temp_planets.rrmscdpp06p0, angle_flag=True) 

prob_detections_kepler_se, transit_statuses_kepler_se, sn_kepler_se, geom_transit_statuses_kepler_se = simulate_transit.calculate_transit_vectorized(kepler_temp_se.periods, 
                                kepler_temp_se.stellar_radius, kepler_temp_se.planet_radii,
                                kepler_temp_se.eccs, 
                                kepler_temp_se.incls, 
                                kepler_temp_se.omegas, kepler_temp_se.stellar_mass,
                                kepler_temp_se.rrmscdpp06p0, angle_flag=True) 

prob_detections_kepler_sn, transit_statuses_kepler_sn, sn_kepler_sn, geom_transit_statuses_kepler_sn = simulate_transit.calculate_transit_vectorized(kepler_temp_sn.periods, 
                                kepler_temp_sn.stellar_radius, kepler_temp_sn.planet_radii,
                                kepler_temp_sn.eccs, 
                                kepler_temp_sn.incls, 
                                kepler_temp_sn.omegas, kepler_temp_sn.stellar_mass,
                                kepler_temp_sn.rrmscdpp06p0, angle_flag=True) 

prob_detections_k2, transit_statuses_k2, sn_k2, geom_transit_statuses_k2 = simulate_transit.calculate_transit_vectorized_k2(k2_temp_planets.periods, 
                                k2_temp_planets.stellar_radius, k2_temp_planets.planet_radii,
                                k2_temp_planets.eccs, 
                                k2_temp_planets.incls, 
                                k2_temp_planets.omegas, k2_temp_planets.stellar_mass,
                                k2_temp_planets.rrmscdpp06p0, k2_temp_planets.baseline, angle_flag=True) 
print(transit_statuses_k2)

prob_detections_k2_se, transit_statuses_k2_se, sn_k2, geom_transit_statuses_k2_se = simulate_transit.calculate_transit_vectorized_k2(k2_temp_se.periods, 
                                k2_temp_se.stellar_radius, k2_temp_se.planet_radii,
                                k2_temp_se.eccs, 
                                k2_temp_se.incls, 
                                k2_temp_se.omegas, k2_temp_se.stellar_mass,
                                k2_temp_se.rrmscdpp06p0, k2_temp_se.baseline, angle_flag=True) 
quit()


berger_kepler_planets['transit_status'] = transit_statuses[0]
berger_kepler_planets['prob_detections'] = prob_detections[0]
berger_kepler_planets['sn'] = sn
berger_kepler_planets['geom_transit_status'] = geom_transit_statuses[0]

kepler_temp_planets['transit_status'] = transit_statuses_kepler[0]
kepler_temp_planets['prob_detections'] = prob_detections_kepler[0]
kepler_temp_planets['sn'] = sn_kepler
kepler_temp_planets['geom_transit_status'] = geom_transit_statuses_kepler[0]

kepler_temp_se['transit_status'] = transit_statuses_kepler_se[0]
kepler_temp_se['prob_detections'] = prob_detections_kepler_se[0]
kepler_temp_se['sn'] = sn_kepler_se
kepler_temp_se['geom_transit_status'] = geom_transit_statuses_kepler_se[0]

kepler_temp_sn['transit_status'] = transit_statuses_kepler_sn[0]
kepler_temp_sn['prob_detections'] = prob_detections_kepler_sn[0]
kepler_temp_sn['sn'] = sn_kepler_sn
kepler_temp_sn['geom_transit_status'] = geom_transit_statuses_kepler_sn[0]

k2_temp_planets['transit_status'] = transit_statuses_k2[0]
k2_temp_planets['prob_detections'] = prob_detections_k2[0]
k2_temp_planets['sn'] = sn_k2



# need kepid to be str or tuple, else unhashable type when groupby.count()
berger_kepler_planets['GaiaDR3'] = berger_kepler_planets['GaiaDR3'].apply(str) 
kepler_temp_planets['GaiaDR3'] = kepler_temp_planets['GaiaDR3'].apply(str) 
kepler_temp_se['GaiaDR3'] = kepler_temp_se['GaiaDR3'].apply(str)
kepler_temp_sn['GaiaDR3'] = kepler_temp_sn['GaiaDR3'].apply(str)
k2_temp_planets['GaiaDR3'] = k2_temp_planets['GaiaDR3'].apply(str) 

# isolate detected transiting planets
berger_kepler_transiters = berger_kepler_planets.loc[berger_kepler_planets['transit_status']==1]
kepler_temp_transiters = kepler_temp_planets.loc[kepler_temp_planets['transit_status']==1]
se_temp_transiters = kepler_temp_se.loc[kepler_temp_se['transit_status']==1]
sn_temp_transiters = kepler_temp_sn.loc[kepler_temp_sn['transit_status']==1]
#k2_temp_transiters = k2_temp_planets.loc[k2_temp_planets['transit_status']==1]

### Completeness
# Calculate completeness map
#if (k==0) or (k==1) or (k==2) or (k==3) or (k==4):
completeness_map, piv_physical, piv_detected = simulate_helpers.completeness(berger_kepler_planets, berger_kepler_transiters)
completeness_threshold = 0.003 # completeness threshold under which period/radius cell is not counted; 0.5% results in full recovery, but let's round up to 1%
completeness_map = completeness_map.mask(completeness_map < completeness_threshold) # assert that completeness fractions lower than 1% are statistically insignificant
completeness_all.append(completeness_map)

completeness_map_kepler, piv_physical_kepler, piv_detected_kepler = simulate_helpers.completeness(kepler_temp_planets, kepler_temp_transiters)
completeness_threshold = 0.003 # completeness threshold under which period/radius cell is not counted; 0.5% results in full recovery, but let's round up to 1%
completeness_map_kepler = completeness_map_kepler.mask(completeness_map_kepler < completeness_threshold) # assert that completeness fractions lower than 1% are statistically insignificant
completeness_kepler.append(completeness_map_kepler)

### Reliability
print(reliability_k2(k2_temp_sn))
quit()

# calculate detected occurrence rate 
berger_kepler_transiters_counts = np.array(berger_kepler_transiters.groupby(['height_bins']).count().reset_index()['GaiaDR3'])
detected_planet_occurrence = berger_kepler_transiters_counts/berger_kepler_counts
detected_planet_occurrences_all.append(detected_planet_occurrence)

kepler_temp_transiters_counts = np.array(kepler_temp_transiters.groupby(['height_bins']).count().reset_index()['GaiaDR3'])
detected_planet_occurrence_kepler = kepler_temp_transiters_counts/kepler_temp_counts
#print("detected: ", detected_planet_occurrence_kepler)
#print("completeness: ", completeness_map_kepler)
detected_planet_occurrences_kepler.append(detected_planet_occurrence_kepler)

berger_kepler_transiters1 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > 100) & (berger_kepler_transiters['height'] <= np.logspace(2,3,6)[1])]
berger_kepler_transiters2 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > np.logspace(2,3,6)[1]) & (berger_kepler_transiters['height'] <= np.logspace(2,3,6)[2])]
berger_kepler_transiters3 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > np.logspace(2,3,6)[2]) & (berger_kepler_transiters['height'] <= np.logspace(2,3,6)[3])]
berger_kepler_transiters4 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > np.logspace(2,3,6)[3]) & (berger_kepler_transiters['height'] <= np.logspace(2,3,6)[4])]
berger_kepler_transiters5 = berger_kepler_transiters.loc[(berger_kepler_transiters['height'] > np.logspace(2,3,6)[4]) & (berger_kepler_transiters['height'] <= 1000)]

#"""
len_berger_kepler_transiters1, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters1, completeness_map, radius_grid, period_grid) #completeness_map_np vs completeness_map
len_berger_kepler_transiters2, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters2, completeness_map, radius_grid, period_grid)
len_berger_kepler_transiters3, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters3, completeness_map, radius_grid, period_grid)
len_berger_kepler_transiters4, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters4, completeness_map, radius_grid, period_grid)
len_berger_kepler_transiters5, _ = simulate_helpers.adjust_for_completeness2(berger_kepler_transiters5, completeness_map, radius_grid, period_grid)
len_berger_kepler_transiters = np.array([len_berger_kepler_transiters1, len_berger_kepler_transiters2, len_berger_kepler_transiters3, len_berger_kepler_transiters4, len_berger_kepler_transiters5])

#len_berger_kepler_recovered, recovered_piv = simulate_helpers.adjust_for_completeness(berger_kepler_transiters, completeness_map, radius_grid, period_grid)
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
#print("COMPLETENESS MAPS")
#print(completeness_all)

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

heights = np.concatenate(heights)
ages = np.concatenate(ages)

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

    tau = numpyro.sample("tau", dist.Uniform(-1., 1.))
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
    #quit()
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
    num_warmup=5000,
    num_samples=10000,
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
left, bottom, width, height = [0.16, 0.25, 0.25, 0.25]
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

"""
### get combined SE and SN tau and occurrence for Zink23
print(yerr_recovered, mean_recovered_planet_occurrences)
print(0.5 * (np.array(zink_kepler['occurrence_err1']) + np.array(zink_kepler['occurrence_err2'])), np.array(zink_kepler['occurrence']))
opt_params = run_optim(jax.random.PRNGKey(5), height_bin_midpoints, 0.5 * (np.array(zink_kepler['occurrence_err1']) + np.array(zink_kepler['occurrence_err2'])), y=np.array(zink_kepler['occurrence']))
#print("opt params: ", opt_params)

# sample posteriors for best-fit model to simulated data
sampler = infer.MCMC(
    infer.NUTS(power_model, dense_mass=True,
        regularize_mass_matrix=False,
        init_strategy=numpyro.infer.init_to_value(values=opt_params)), 
    num_warmup=10000,
    num_samples=10000,
    num_chains=8,
    progress_bar=True,
)
sampler.run(jax.random.PRNGKey(0), height_bin_midpoints, 0.5 * (np.array(zink_kepler['occurrence_err1']) + np.array(zink_kepler['occurrence_err2'])), y=np.array(zink_kepler['occurrence']))
inf_data = az.from_numpyro(sampler)
print(az.summary(inf_data))
quit()
"""

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
#x = np.linspace(0, 14, 1000)
#y = np.where(x <= threshold, frac1, frac2)
x = np.linspace(14, 0, 1000)
y = np.where(x <= 13.7 - threshold, frac2, frac1)

# monotonic model
#b = frac1
#m = (frac2 - frac1)/(x[-1] - x[0])
#y = b + m * x

# piecewise model
#m = (frac2 - frac1)/(x[-1] - threshold)
#y = np.where(x < threshold, frac1, frac1 + m * (x-threshold))

ax2.plot(x, y, color='powderblue', linewidth=2)
#ax2.invert_xaxis()
ax2.set_xlabel('lookback time [Gyr]')
ax2.set_ylabel('planet host fraction')
ax2.set_ylim([0,1])

fig.tight_layout()
#plt.savefig(path+'plots2/results'+name+'_lookback.png', format='png', bbox_inches='tight')

#plt.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='Zink+ 2023 Kepler data')
#plt.scatter(x=zink_kepler['scale_height'], y=physical_planet_occurrence, c='red', label='model')
#plt.xlabel(r'$Z_{max}$ [pc]')
#plt.ylabel('planets per 100 stars')
#plt.legend()
#plt.tight_layout()
#plt.savefig(path+'plots/'+name)
plt.show()
