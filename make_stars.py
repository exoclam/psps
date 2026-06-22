#######################################
# MAKE STARS AND PLANETS FROM HU+ 2025
#######################################

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
from scipy.stats import truncnorm

from itertools import zip_longest
import numpy.ma as ma # for masked arrays

from astropy.table import Table, join

from psps.transit_class import Population, Star, GeneralStar, K2Star, StarZ23, K2StarZ23 # Star is for Kepler stars
import psps.simulate_helpers as simulate_helpers
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

age_bins = np.linspace(0, 8, 20)
teff_bins = np.linspace(4500, 7000, 20)

sample_quality = 'bad_new' # bad, meh, good
# whichever one doesn't get the _bad_ages or _good_ages qualifier gets used
if (sample_quality=='bad') or (sample_quality=='bad_new'):
    hu25_b20_kepler_b25_k2 = pd.read_csv(path+'data/joint/hu25_b20_kepler_b25_k2.csv') # aka bad ages (no fractional age uncertainty cut)
elif sample_quality=='meh':
    hu25_b20_kepler_b25_k2 = pd.read_csv(path+'data/joint/hu25_b20_kepler_b25_k2_good_ages.csv') 
elif sample_quality=='good':
    #hu25_b20_kepler_b25_k2 = pd.read_csv(path+'data/joint/hu25_b20_kepler_b25_k2_soft_age_uncertainty_cut.csv') # aka meh ages (fractional age uncertainty cut at 1.)
    hu25_b20_kepler_b25_k2 = pd.read_csv(path+'data/joint/hu25_b20_kepler_b25_k2_meh_ages.csv')
kepler = pd.read_csv(path+'data/kepler/kepler_stars_bootstrapped.csv')
kepler_stars = pd.read_csv(path+'data/joint/kepler_only_bootstrap.csv') # result of real_kepler_only_planets.py
kepler_bad = pd.read_csv(path+'data/kepler/kepler_stars_bootstrapped_bad.csv')
kepler_only = pd.read_csv('/Users/chrislam/Desktop/psps/data/joint/kepler_only/piecewise_95_5_100/piecewise_95_5_100_kepler_only_0.csv')
bad = pd.read_csv(path+'data/joint/hu25_b20_kepler_b25_k2.csv') # aka bad ages (no fractional age uncertainty cut)
good = pd.read_csv(path+'data/joint/hu25_b20_kepler_b25_k2_good_ages.csv') 
meh = pd.read_csv(path+'data/joint/hu25_b20_kepler_b25_k2_meh_ages.csv')

keep_zmax = pd.read_csv(path+'data/joint/stellar_samples/keep_zmax.csv')
bad = pd.merge(bad, keep_zmax, left_on='source_id_dr3', right_on='GaiaDR3', how='inner')
good = pd.merge(good, keep_zmax, left_on='source_id_dr3', right_on='GaiaDR3', how='inner')
meh = pd.merge(meh, keep_zmax, left_on='source_id_dr3', right_on='GaiaDR3', how='inner')

#plt.hist(kepler['age'], bins=np.linspace(0, 8, 20), density=True, alpha=0.3, label='Kepler-only')
#plt.hist(kepler_bad['age'], bins=np.linspace(0, 8, 20), density=True, alpha=0.3, label='Kepler-only')
plt.hist(kepler_only['age'], bins=np.linspace(0, 8, 20), density=True, alpha=0.3, label='Kepler-only')
plt.hist(bad['age'], bins=np.linspace(0, 8, 20), density=True, alpha=0.3, label='joint Kepler-K2')
#plt.hist(meh['age'], bins=np.linspace(0, 8, 20), density=True, alpha=0.3, label='meh')
#plt.hist(good['age'], bins=np.linspace(0, 8, 20), density=True, alpha=0.3, label='good')
plt.xlabel('age [Gyr]')
plt.legend()
plt.savefig(path+'plots/joint/age_kepler_only_vs_joint.png')
plt.show()

# plt.hist(kepler['height']*1000, bins=np.logspace(2, 3, 20), density=True, alpha=0.3, label='kepler-only')
# plt.hist(kepler_bad['height']*1000, bins=np.logspace(2, 3, 20), density=True, alpha=0.3, label='kepler-bad')
# plt.hist(bad['height']*1000, bins=np.logspace(2, 3, 20), density=True, alpha=0.3, label='bad')
# plt.hist(meh['height']*1000, bins=np.logspace(2, 3, 20), density=True, alpha=0.3, label='meh')
# plt.hist(good['height']*1000, bins=np.logspace(2, 3, 20), density=True, alpha=0.3, label='good')
# plt.xlabel('height [pc]')
# plt.legend()
# #plt.savefig(path+'plots/joint/zmax_kepler_vs_bad_vs_meh_vs_good_keep_zmax.png')
# plt.show()
# quit()

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

# planet formation history model parameters
threshold = 6.5 # cosmic age in Gyr; 13.7 minus stellar age, then round
frac1 = 0.01 # frac1 must be < frac2 if comparing cosmic ages
frac2 = 0.65

name_thresh = 65
name_f1 = 1
name_f2 = 65
#name = 'step_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2)
#name = 'monotonic_'+str(name_f1)+'_'+str(name_f2)
name = 'piecewise_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2)

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
k2_pointings = pd.DataFrame(dict({'Campaign': campaigns, 'ra': ras, 'dec': decs, 'b': bs, 'baseline': baselines, 'logistic_a': logistic_a, 'logistic_k': logistic_k, 'logistic_l': logistic_l}))

# for each model, draw around stellar age errors 10 times
kepler_ages = []
k2_ages = []
total_ages = []
kepler_planets_ages = []
k2_planets_ages = []
total_planets_ages = []
for j in tqdm(range(10)): 

    temp_df = hu25_b20_kepler_b25_k2.copy() # copy the original DataFrame to avoid modifying it in place during each iteration

    # draw stellar radius, mass, and age using asymmetric errors 
    a, b = (0 - temp_df['Rad']) / temp_df['e_Rad'], np.inf
    temp_df['stellar_radius'] = truncnorm.rvs(a, b, loc=temp_df['Rad'], scale=temp_df['e_Rad'], size=len(temp_df)) # to avoid negative values
        
    a, b = (0 - temp_df['Mass']) / temp_df['e_Mass'], np.inf
    temp_df['stellar_mass'] = truncnorm.rvs(a, b, loc=temp_df['Mass'], scale=temp_df['e_Mass'], size=len(temp_df)) # to avoid negative values
    
    #a, b = (0 - temp_df['Teff']) / temp_df['e_Teff'], np.inf
    #temp_df['teff_drawn'] = truncnorm.rvs(a, b, loc=temp_df['Teff'], scale=temp_df['e_Teff'], size=len(temp_df))
    temp_df['teff_drawn'] = np.random.normal(temp_df['Teff'], temp_df['e_Teff'])

    temp_df['age_err2'] = -1 * np.abs(temp_df['age_err2'])
    temp_df = simulate_helpers.draw_asymmetrically(temp_df, 'age', 'age_err1', 'age_err2', 'age_drawn') # use this bc the psps version uses the same name as our input column
    #print(temp_df[['age','age_err1','age_err2','age_drawn']])

    temp_df = temp_df.loc[(temp_df['age_drawn'] <= 8.) & (temp_df['age_drawn'] >= 0.5)]
    print("age cut: ", len(temp_df))

    # assign, rather than compute, zmax
    temp_df = pd.merge(temp_df, keep_zmax, left_on='source_id_dr3', right_on='GaiaDR3', how='inner')
    temp_df['height'] = np.random.normal(temp_df['zmax_mean'], temp_df['zmax_std'])

    # re-assign heights
    #temp_df['height'] = simulate_helpers.gala_galactic_heights(Table.from_pandas(temp_df))
    # plt.hist(temp_df['teff_drawn'], bins=teff_bins, density=True, alpha=0.3, label='joint bad')

    # temp_df['feh_drawn'] = np.random.normal(temp_df['[Fe/H]'], temp_df['e_[Fe/H]'])
    # temp_df = temp_df.loc[(temp_df['feh_drawn'] <= 0.25) & (temp_df['feh_drawn'] >= -0.25)]
    # print("Fe/H cut: ", len(temp_df))

    # plt.hist(kepler['iso_teff'], bins=teff_bins, density=True, alpha=0.3, label='Kepler only')
    # plt.xlabel('age [Gyr]')
    # plt.legend()
    # plt.show()
    # quit()

    # split between Kepler and K2
    temp_df['kepler_or_k2'] = np.where(temp_df['Kepler_ID'] > 0, 'Kepler', 'K2')
    temp_kepler = temp_df.loc[temp_df['kepler_or_k2']=='Kepler']
    temp_k2 = temp_df.loc[temp_df['kepler_or_k2']=='K2']
    temp_k2 = pd.merge(temp_k2, k2_pointings[['Campaign','baseline','logistic_a','logistic_k','logistic_l']], on='Campaign', how='left')

    ### create a Population object to hold information about the occurrence law governing that specific population
    # THIS IS WHERE YOU CHOOSE THE PLANET FORMATION HISTORY MODEL 
    pop = Population(temp_df['age_drawn'], threshold, frac1, frac2)
    pop_kepler = Population(temp_kepler['age_drawn'], threshold, frac1, frac2)
    pop_k2 = Population(temp_k2['age_drawn'], threshold, frac1, frac2)

    if name[:4]=='step':
        frac_hosts = pop.galactic_occurrence_step(threshold, frac1, frac2)
        frac_hosts_kepler = pop_kepler.galactic_occurrence_step(threshold, frac1, frac2)
        frac_hosts_k2 = pop_k2.galactic_occurrence_step(threshold, frac1, frac2)
    elif name[:4]=='mono':
        frac_hosts = pop.galactic_occurrence_monotonic(frac1, frac2)
        frac_hosts_kepler = pop_kepler.galactic_occurrence_monotonic(frac1, frac2)
        frac_hosts_k2 = pop_k2.galactic_occurrence_monotonic(frac1, frac2)
    elif name[:4]=='piec':
        frac_hosts = pop.galactic_occurrence_piecewise(frac1, frac2, threshold)
        frac_hosts_kepler = pop_kepler.galactic_occurrence_piecewise(frac1, frac2, threshold)
        frac_hosts_k2 = pop_k2.galactic_occurrence_piecewise(frac1, frac2, threshold)

    print("f: ", np.nanmean(frac_hosts))

    intact_fracs = scipy.stats.truncnorm.rvs(0, 1, loc=0.18, scale=0.1, size=len(temp_df))  
    intact_fracs_kepler = scipy.stats.truncnorm.rvs(0, 1, loc=0.18, scale=0.1, size=len(temp_kepler))  
    intact_fracs_k2 = scipy.stats.truncnorm.rvs(0, 1, loc=0.18, scale=0.1, size=len(temp_k2))  

    alpha_se = np.random.normal(-1., 0.2)
    alpha_sn = np.random.normal(-1.5, 0.1)

    alpha_se_kepler = np.random.normal(-1.0, 0.2)
    alpha_se_k2 = np.random.normal(-1.9, 0.7)
    alpha_sn_kepler = np.random.normal(-1.5, 0.1)
    alpha_sn_k2 = np.random.normal(-2., 0.3)

    beta1_se_kepler = np.random.normal(1.9, 0.3)
    beta1_se_k2 = np.random.normal(1.7, 0.4)
    beta1_sn_kepler = np.random.normal(2.6, 0.5)
    beta1_sn_k2 = np.random.normal(2.8, 0.6)

    pbr_se_kepler = np.random.normal(5.9, 1.2)
    pbr_se_k2 = np.random.normal(8.9, 7.4)
    pbr_sn_kepler = np.random.normal(8.5, 2.4)
    pbr_sn_k2 = np.random.normal(10.9, 5.)

    beta2_se_kepler = np.random.normal(0.2, 0.2)
    beta2_se_k2 = np.random.normal(0.5, 1.2)
    beta2_sn_kepler = np.random.normal(0.6, 0.2)
    beta2_sn_k2 = np.random.normal(0.9, 0.7)

    # actually asymmetric, but for simplicity's sake, we will draw using the smaller uncertainty 
    gamma = np.random.normal(-0.09, 0.02) 
    a_low = np.random.normal(0.29, 0.03)
    a_upp = np.random.normal(0.44, 0.03)

    #"""
    # create Star objects, with their planetary systems
    # star_data = []
    # kepler_or_k2 = []
    # for i in tqdm(range(len(temp_df))): # 100
    #     star = GeneralStar(temp_df['source_id_dr3'].iloc[i], temp_df['age'].iloc[i], temp_df['stellar_radius'].iloc[i], temp_df['stellar_mass'].iloc[i], temp_df['teff_drawn'].iloc[i], temp_df['CDPP6'].iloc[i], temp_df['height'].iloc[i], alpha_se, alpha_sn, frac_hosts[i], intact_fracs[i], temp_df['kepler_or_k2'].iloc[i])
    #     star_update = {
    #         'source_id_dr3': star.GaiaDR3,
    #         'age': star.age,
    #         'stellar_radius': star.stellar_radius,
    #         'stellar_mass': star.stellar_mass,
    #         'Teff': star.Teff,
    #         'rrmscdpp06p0': star.rrmscdpp06p0,
    #         'frac_host': star.frac_host,
    #         'height': star.height,
    #         'midplane': star.midplane,
    #         'prob_intact': star.prob_intact,
    #         'status': star.status,
    #         'sigma_incl': star.sigma_incl,
    #         'num_planets': star.num_planets,
    #         'periods': star.periods,
    #         'incls': star.incls,
    #         'mutual_incls': star.mutual_incls,
    #         'eccs': star.eccs,
    #         'omegas': star.omegas,
    #         'planet_radii': star.planet_radii,
    #         'kepler_or_k2': star.kepler_or_k2,
    #         'se_or_sn': star.se_or_sn
    #     }
    #     star_data.append(star_update)
    #     pop.add_child(star)

        # if temp_df['Kepler_ID'].iloc[i] > 0:
        #     kepler_or_k2.append('Kepler')
        # elif temp_df['EPIC_ID'].iloc[i] > 0:
        #     kepler_or_k2.append('K2')

    star_data_k2 = []
    for i in tqdm(range(len(temp_k2))): # 100
        #star = K2StarZ23(temp_k2['age_drawn'].iloc[i], temp_k2['stellar_radius'].iloc[i], temp_k2['stellar_mass'].iloc[i], temp_k2['teff_drawn'].iloc[i], temp_k2['CDPP6'].iloc[i], temp_k2['height'].iloc[i], alpha_se_k2, alpha_sn_k2, beta1_se_k2, beta1_sn_k2, beta2_se_k2, beta2_sn_k2, pbr_se_k2, pbr_sn_k2, frac_hosts_k2[i], intact_fracs_k2[i], temp_k2['Campaign'].iloc[i], temp_k2['baseline'].iloc[i], temp_k2['source_id_dr3'].iloc[i], temp_k2['EPIC_ID'].iloc[i], temp_k2['logistic_a'].iloc[i], temp_k2['logistic_k'].iloc[i], temp_k2['logistic_l'].iloc[i])
        star = K2Star(temp_k2['age_drawn'].iloc[i], temp_k2['stellar_radius'].iloc[i], temp_k2['stellar_mass'].iloc[i], temp_k2['teff_drawn'].iloc[i], temp_k2['CDPP6'].iloc[i], temp_k2['height'].iloc[i], alpha_se_k2, alpha_sn_k2, frac_hosts_k2[i], intact_fracs_k2[i], temp_k2['Campaign'].iloc[i], temp_k2['baseline'].iloc[i], temp_k2['source_id_dr3'].iloc[i], temp_k2['EPIC_ID'].iloc[i], temp_k2['logistic_a'].iloc[i], temp_k2['logistic_k'].iloc[i], temp_k2['logistic_l'].iloc[i])        
        star_update = {
            'GaiaDR3': star.GaiaDR3,
            'age': star.age,
            'stellar_radius': star.stellar_radius,
            'stellar_mass': star.stellar_mass,
            'Teff': star.Teff,
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
            'planet_radii': star.planet_radii,
            #'kepler_or_k2': star.kepler_or_k2,
            'se_or_sn': star.se_or_sn,
            'EPIC_ID': star.EPIC_ID,
            'campaign': star.campaign,
            'baseline': star.baseline,
            'logistic_a': star.logistic_a,
            'logistic_k': star.logistic_k,
            'logistic_l': star.logistic_l
        }
        star_data_k2.append(star_update)
        pop_k2.add_child(star)

    # ### this is for generating that single kepler-only dataset, for comparison
    # temp_kepler = kepler.copy() 
    # temp_kepler['age_drawn'] = temp_kepler['age']
    # temp_kepler['teff_drawn'] = temp_kepler['Teff']
    # temp_kepler['CDPP6'] = temp_kepler['rrmscdpp06p0']
    # temp_kepler['Kepler_ID'] = temp_kepler['KIC']
    # temp_kepler['source_id_dr3'] = 555*np.ones(len(temp_kepler))
    # pop_kepler = Population(temp_kepler['age_drawn'], threshold, frac1, frac2)
    # frac_hosts_kepler = pop_kepler.galactic_occurrence_piecewise(frac1, frac2, threshold)
    # intact_fracs_kepler = scipy.stats.truncnorm.rvs(0, 1, loc=0.18, scale=0.1, size=len(temp_kepler))  

    star_data_kepler = []
    for i in tqdm(range(len(temp_kepler))): # 100
        #star = StarZ23(temp_kepler['age_drawn'].iloc[i], temp_kepler['stellar_radius'].iloc[i], temp_kepler['stellar_mass'].iloc[i], temp_kepler['teff_drawn'].iloc[i], temp_kepler['CDPP6'].iloc[i], temp_kepler['height'].iloc[i], alpha_se_kepler, alpha_sn_kepler, beta1_se_kepler, beta1_sn_kepler, beta2_se_kepler, beta2_sn_kepler, pbr_se_kepler, pbr_sn_kepler, frac_hosts_kepler[i], intact_fracs_kepler[i], temp_kepler['source_id_dr3'].iloc[i], temp_kepler['Kepler_ID'].iloc[i])
        star = Star(temp_kepler['age_drawn'].iloc[i], temp_kepler['stellar_radius'].iloc[i], temp_kepler['stellar_mass'].iloc[i], temp_kepler['teff_drawn'].iloc[i], temp_kepler['CDPP6'].iloc[i], temp_kepler['height'].iloc[i], alpha_se_kepler, alpha_sn_kepler, frac_hosts_kepler[i], intact_fracs_kepler[i], temp_kepler['source_id_dr3'].iloc[i], temp_kepler['Kepler_ID'].iloc[i])
        star_update = {
            'GaiaDR3': star.GaiaDR3,
            'age': star.age,
            'stellar_radius': star.stellar_radius,
            'stellar_mass': star.stellar_mass,
            'Teff': star.Teff,
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
            'planet_radii': star.planet_radii,
            #'kepler_or_k2': star.kepler_or_k2,
            'se_or_sn': star.se_or_sn,
            'Kepler_ID': star.Kepler_ID
        }
        star_data_kepler.append(star_update)
        pop_kepler.add_child(star)

    # convert back to DataFrame
    #berger_kepler_all = pd.DataFrame.from_records(star_data)
    berger_k2 = pd.DataFrame.from_records(star_data_k2)
    berger_kepler = pd.DataFrame.from_records(star_data_kepler)
    kepler_ages.append(berger_kepler['age'])
    k2_ages.append(berger_k2['age'])
    ###berger_kepler.to_csv(path+'data/joint/good_ages_stellar_samples/'+name+'_kepler_only.csv', index=False) # make a single file that uses only Kepler (pre-HU25), for comparison
    ###quit()
    #"""

    ### figure out whether to draw uniformly in linear space and then arcsin it, or the OG way of drawing uniformly in angle space
    # plt.hist(berger_kepler['midplane'], bins=np.linspace(-np.pi/2, np.pi/2, 100), density=True, alpha=0.3, label='midplane')
    # old_way = np.random.uniform(-np.pi/2, np.pi/2, 1000)
    # plt.hist(old_way, bins=np.linspace(-np.pi/2, np.pi/2, 10), density=True, alpha=0.3, label='old way')
    # u = np.random.uniform(-1, 1, 1000)
    # i_mid = np.arcsin(u)
    # plt.hist(i_mid, bins=np.linspace(-np.pi/2, np.pi/2, 10), density=True, alpha=0.3, label='new way')
    # plt.legend()
    # plt.show()
    # quit()

    #berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all.num_planets > 0]
    berger_kepler_planets = berger_kepler.loc[berger_kepler.num_planets > 0]
    berger_k2_planets = berger_k2.loc[berger_k2.num_planets > 0]
    kepler_planets_ages.append(berger_kepler_planets['age'])
    k2_planets_ages.append(berger_k2_planets['age'])

    planets = pd.concat([berger_kepler_planets, berger_k2_planets])
    planets_explode = planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas', 'se_or_sn']).reset_index(drop=True)
    planets_explode = planets_explode.loc[(planets_explode['periods'] <= 40) & (planets_explode['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
    planets_explode = planets_explode.loc[planets_explode['planet_radii'] <= 4.]
    planets_se = planets_explode.loc[planets_explode['se_or_sn']=='se']
    planets_sn = planets_explode.loc[planets_explode['se_or_sn']=='sn']
    plt.scatter(planets_se['periods'], planets_se['planet_radii'], s=5, c='purple', alpha=0.3, label='SE')
    plt.scatter(planets_sn['periods'], planets_sn['planet_radii'], s=5, c='green', alpha=0.3, label='SN')
    plt.xlabel('period [day]')
    plt.ylabel(r'radius [$R_{\oplus}$]')
    plt.xscale('log')
    plt.legend()
    plt.show()
    quit()

    #berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
    #berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 4.]

    # ### temp check that height age relation (and expected scatter) remains
    # bins2d = [np.linspace(0, 8, 10), np.logspace(2, 3, 10)]
    # label='HU25'
    # ages = berger_kepler_planets['age']
    # heights = berger_kepler_planets['height']*1000
    # norm = 10
    # hist, xedges, yedges = np.histogram2d(ages, heights, bins=bins2d)
    # hist = hist.T
    # if label=='TRI':
    #     ax = plt.pcolormesh(xedges, yedges, hist, cmap='Blues')
    #     plt.xlabel('TRILEGAL age [Gyr]')
    #     plt.ylabel('TRILEGAL height [pc]')
    # elif label=='HU25':
    #     ax = plt.pcolormesh(xedges, yedges, hist, cmap='Blues')
    #     plt.xlabel('Kepler SE age [Gyr]')
    #     plt.ylabel('Kepler SE height [pc]')
    # plt.tight_layout()
    # plt.show()
    # quit()

    berger_stars_all = pd.concat([berger_kepler, berger_k2])
    berger_planets_all = pd.concat([berger_kepler_planets, berger_k2_planets])
    total_ages.append(berger_stars_all['age'])
    total_planets_ages.append(berger_planets_all['age'])
    f = len(berger_planets_all)/len(berger_stars_all)
    print("f: ", f)

    # # age distribution
    # age_bins = np.linspace(0, 8, 20)
    # plt.hist(berger_stars_all['age'], bins=age_bins, density=True, alpha=0.4, label='field')
    # plt.hist(berger_planets_all['age'], bins=age_bins, density=True, alpha=0.5, label='host')
    # plt.xlabel('age [Gyr]')
    # plt.legend()
    # #plt.savefig(path+'plots/joint/age_'+name+'.png')
    # plt.show()
    # quit()

    if sample_quality=='meh':
        try:
            os.mkdir(path+'data/joint/meh_ages_stellar_samples/'+name)
            print(f"Directory '{name}' created")
        except FileExistsError:
            print(f"Directory '{name}' already exists")
        berger_stars_all.to_csv(path+'data/joint/meh_ages_stellar_samples/'+name+'/'+name+'_'+str(j)+'.csv', index=False)

    elif sample_quality=='good':
        try:
            os.mkdir(path+'data/joint/good_ages_stellar_samples/'+name)
            print(f"Directory '{name}' created")
        except FileExistsError:
            print(f"Directory '{name}' already exists")
        berger_stars_all.to_csv(path+'data/joint/good_ages_stellar_samples/'+name+'/'+name+'_'+str(j)+'.csv', index=False)

    elif sample_quality=='bad': # OG
        try:
            os.mkdir(path+'data/joint/stellar_samples/'+name)
            print(f"Directory '{name}' created")
        except FileExistsError:
            print(f"Directory '{name}' already exists")
        berger_stars_all.to_csv(path+'data/joint/stellar_samples/'+name+'/'+name+'_'+str(j)+'.csv', index=False)

    elif sample_quality=='bad_new': # OG
        try:
            os.mkdir(path+'data/joint/bad_stellar_samples/'+name)
            print(f"Directory '{name}' created")
        except FileExistsError:
            print(f"Directory '{name}' already exists")
        berger_stars_all.to_csv(path+'data/joint/bad_stellar_samples/'+name+'/'+name+'_'+str(j)+'.csv', index=False)
    
# age_bins = np.linspace(0, 14, 20)
# for i in range(5):
#     plt.hist(kepler_ages[i], bins=age_bins, alpha=0.1, density=True, color='blue', label='field')
#     plt.hist(kepler_planets_ages[i], bins=age_bins, alpha=0.1, density=True, color='orange', label='hosts')
# plt.xlabel('age [Gyr]')
# plt.legend()
# plt.savefig(path+'plots/joint/ages_'+name+'.png')
# plt.show()
