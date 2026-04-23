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

# K2 campaign pointings, from https://archive.stsci.edu/missions-and-data/k2/campaign-fields
ras = [173.939610, 246.1264, 336.66534641439, 59.0759116, 130.1576478, 204.8650344, 287.82850661, 16.3379975, 270.3544823, 186.7794430,
	   260.3880064, 351.6588124, 72.7971166, 160.6824762, 233.6175730, 133.7099689, 202.5496152, 
	   130.1610170, 347.2590265]
decs = [1.4172989, -22.4473, -11.096663792177, 18.6605794, 16.8296140, -11.2953585, -23.36001815, 5.2623459, -21.7798098, -4.0271572,
		-23.9759578, -5.1023328, 20.7870759, 6.8509316, -20.0792397, 18.5253931, -7.7210759, 
		16.8278629, -4.2027029]
campaigns = np.arange(19)+1
highs = [1, 3, 6, 8, 10, 12, 14, 17, 19]
lows = [2, 4, 5, 7, 9, 11, 13, 15, 16, 18]
bs = []
for i in range(19):
    bs.append(simulate_helpers.convert_ra_dec_to_b(ras[i], decs[i]))
baselines = [83, 82, 81, 75, 74, 78, 83, 80, 71, 76, 75, 79, 80, 80, 89, 80, 68, 51, 28]
# campaign-wise recovery fraction parameters for the logistic function Eqn 6 in https://iopscience.iop.org/article/10.3847/1538-3881/ac2309#ajac2309t2. Params available for C1-C8 and C10-C18, with C9 and C19 being just the AFGK dwarf value used
logistic_a = [0.3923, 0.6430, 0.7462, 0.6734, 0.4425, 0.7654, 0.3941, 0.6669, 0.6095, 0.5572, 0.2171, 0.6192, 0.6853, 0.7505, 0.6067, 0.6809, 0.5848, 0.6116, 0.6095]
logistic_k = [0.7654, 0.7173, 0.6689, 0.6344, 0.5923, 0.5759, 0.6052, 0.5726, 0.6088, 0.6469, 0.4759, 0.7341, 0.5698, 0.6596, 0.6480, 0.7256, 0.6633, 0.4676, 0.6088]
logistic_l = [11.3914, 10.8544, 10.5701, 11.1443, 11.3923, 10.8772, 11.7002, 10.0560, 10.8986, 10.0056, 12.3882, 10.6272, 11.3878, 10.9776, 10.4673, 10.5453, 10.3635, 11.5783, 10.8986]
k2_pointings = dict({'campaign': campaigns, 'ra': ras, 'dec': decs, 'b': bs, 'baseline': baselines, 'logistic_a': logistic_a, 'logistic_k': logistic_k, 'logistic_l': logistic_l})

# prep lists
heights = []
ages = []
fs = []
physical_planet_occurrences_all = []
detected_planet_occurrences_all = []
adjusted_planet_occurrences_all = []
transit_multiplicities_all = []
geom_transit_multiplicities_all = []
physical_planet_occurrences_kepler = []
detected_planet_occurrences_kepler = []
adjusted_planet_occurrences_kepler = []
physical_planet_occurrences_k2 = []
detected_planet_occurrences_k2 = []
adjusted_planet_occurrences_k2 = []
completeness_all = []
completeness_kepler = []
completeness_k2 = []

# lists for SEs and SNs
physical_planet_occurrences_se_all = []
physical_planet_occurrences_sn_all = []
physical_planet_occurrences_se_kepler = []
physical_planet_occurrences_sn_kepler = []
physical_planet_occurrences_se_k2 = []
physical_planet_occurrences_sn_k2 = []

# define grids
period_grid = np.logspace(np.log10(1), np.log10(40), 10)
radius_grid = np.linspace(1, 4, 10)
#height_bins = np.array([0., 150, 250, 400, 650, 3000]) 
height_bins = np.array([0., 120, 200, 300, 500, 800, 1500]) # the actual Zink Fig 12 height bins
height_bins = np.logspace(2, 3, 6) # ah, so the above are the midpoints of the actual bins they used, I guess
height_bin_midpoints = 0.5 * (np.logspace(2,3,6)[1:] + np.logspace(2,3,6)[:-1])

# define reliability for Kepler and K2
# Kepler: Fig 8 in https://iopscience.iop.org/article/10.3847/1538-4365/aab4f9

# K2: Fig 13 in https://iopscience.iop.org/article/10.3847/1538-3881/ac2309; code block modified from https://github.com/jonzink/ExoMult/blob/master/ScalingK2VIII/ExoMult_Teff_Tutorial.ipynb
def reliability_k2(k2_sn):
    """Build reliability map for K2 using Zink+21

    Args:
        k2_sn (Pandas DF): Pandas DataFrame of K2 sub-Neptunes

    Returns:
        _type_: _description_
    """

    reliability=np.ones(len(k2_sn))
    reli=np.where(k2_sn.Period<5,.99,reli)
    reli=np.where((k2_sn.Period>=5) & (k2_sn.Period<10),.94,reli)
    reli=np.where((k2_sn.Period>=10) & (k2_sn.Period<20),.86,reli)
    reli=np.where((k2_sn.Period>=20) & (k2_sn.Period<30),.87,reli)
    reli=np.where((k2_sn.Period>=30) & (k2_sn.Period<40),.75,reli)
    k2_sn["reli"]=reli*1

    return 

# calculate detected planet yields for each synthetic stellar-planetary population
for i in tqdm(range(len(sim))):

    berger_kepler_all = pd.read_csv(sim[i], sep=',') #, on_bad_lines='skip'
    #berger_kepler_all = pd.read_csv(path+'data/berger_gala/'+name+'.csv')
    
    num_hosts = berger_kepler_all.loc[berger_kepler_all['num_planets']>0]
    #print("f: ", len(num_hosts)/len(berger_kepler_all))
    f = len(num_hosts)/len(berger_kepler_all)
    fs.append(f)

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

    # split into Kepler and K2, since they have different detection efficiencies (bc different baselines and different pointings)
    kepler_temp = berger_kepler_all.loc[berger_kepler_all['kepler_or_k2'] == 'Kepler']
    k2_temp = berger_kepler_all.loc[berger_kepler_all['kepler_or_k2'] == 'K2']

    heights.append(np.array(berger_kepler_all['height']))
    ages.append(np.array(berger_kepler_all['age']))

    # denominators
    berger_kepler_all['height_bins'] = pd.cut(berger_kepler_all['height'], bins=height_bins, include_lowest=True)
    berger_kepler_counts = np.array(berger_kepler_all.groupby(['height_bins']).count().reset_index()['source_id_dr3'])

    kepler_temp['height_bins'] = pd.cut(kepler_temp['height'], bins=height_bins, include_lowest=True)
    kepler_temp_counts = np.array(kepler_temp.groupby(['height_bins']).count().reset_index()['source_id_dr3'])

    k2_temp['height_bins'] = pd.cut(k2_temp['height'], bins=height_bins, include_lowest=True)
    print(k2_temp)
    print(list(k2_temp.columns))
    quit()
    k2_temp_counts = np.array(k2_temp.groupby(['height_bins']).count().reset_index()['source_id_dr3'])

    # isolate planet hosts and bin them by galactic height
    berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all['num_planets'] > 0]

    # EXPLODE
    berger_kepler_planets = berger_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas', 'se_or_sn']).reset_index(drop=True)

    # split into SEs and SNs
    berger_kepler_se = berger_kepler_planets.loc[berger_kepler_planets['se_or_sn']=='se']
    berger_kepler_sn = berger_kepler_planets.loc[berger_kepler_planets['se_or_sn']=='sn']

    # select planets that are relevant to this study 
    berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
    berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 4.] # limit radii to fairly compare with SEs in Zink+ 2023 (2)...or how about include SNs too (4)?

    # split into Kepler and K2 for numerators
    kepler_temp_planets = berger_kepler_planets.loc[berger_kepler_planets['kepler_or_k2']=='Kepler']
    k2_temp_planets = berger_kepler_planets.loc[berger_kepler_planets['kepler_or_k2']=='K2']
    kepler_temp_se = berger_kepler_planets.loc[(berger_kepler_planets['kepler_or_k2']=='Kepler') & (berger_kepler_planets['se_or_sn']=='se')]
    k2_temp_se = berger_kepler_planets.loc[(berger_kepler_planets['kepler_or_k2']=='K2') & (berger_kepler_planets['se_or_sn']=='se')]
    kepler_temp_sn = berger_kepler_planets.loc[(berger_kepler_planets['kepler_or_k2']=='Kepler') & (berger_kepler_planets['se_or_sn']=='sn')]   
    k2_temp_sn = berger_kepler_planets.loc[(berger_kepler_planets['kepler_or_k2']=='K2') & (berger_kepler_planets['se_or_sn']=='sn')]

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
    prob_detections, transit_statuses, sn, geom_transit_statuses = simulate_transit.calculate_transit_vectorized(berger_kepler_planets.periods, 
                                    berger_kepler_planets.stellar_radius, berger_kepler_planets.planet_radii,
                                    berger_kepler_planets.eccs, 
                                    berger_kepler_planets.incls, 
                                    berger_kepler_planets.omegas, berger_kepler_planets.stellar_mass,
                                    berger_kepler_planets.rrmscdpp06p0, angle_flag=True) 

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
    berger_kepler_planets['source_id_dr3'] = berger_kepler_planets['source_id_dr3'].apply(str) 
    kepler_temp_planets['source_id_dr3'] = kepler_temp_planets['source_id_dr3'].apply(str) 
    kepler_temp_se['source_id_dr3'] = kepler_temp_se['source_id_dr3'].apply(str)
    kepler_temp_sn['source_id_dr3'] = kepler_temp_sn['source_id_dr3'].apply(str)
    k2_temp_planets['source_id_dr3'] = k2_temp_planets['source_id_dr3'].apply(str) 

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
    berger_kepler_transiters_counts = np.array(berger_kepler_transiters.groupby(['height_bins']).count().reset_index()['source_id_dr3'])
    detected_planet_occurrence = berger_kepler_transiters_counts/berger_kepler_counts
    detected_planet_occurrences_all.append(detected_planet_occurrence)

    kepler_temp_transiters_counts = np.array(kepler_temp_transiters.groupby(['height_bins']).count().reset_index()['source_id_dr3'])
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
