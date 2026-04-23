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

from itertools import zip_longest
import numpy.ma as ma # for masked arrays

from astropy.table import Table, join

from psps.transit_class import Population, Star, GeneralStar, K2Star # Star is for Kepler stars
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

hu25_b20_kepler_b25_k2_kepmag = pd.read_csv(path+'data/joint/hu25_b20_kepler_b25_k2.csv')

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
try:
    os.mkdir(path+'data/joint/stellar_samples/'+name)
    print(f"Directory '{name}' created")
except FileExistsError:
    print(f"Directory '{name}' already exists")

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
for j in tqdm(range(5)): 

    temp_df = hu25_b20_kepler_b25_k2_kepmag.copy() # copy the original DataFrame to avoid modifying it in place during each iteration

    # draw stellar radius, mass, and age using asymmetric errors 
    temp_df['stellar_radius'] = np.random.normal(temp_df['Rad'], temp_df['e_Rad'])
    temp_df = simulate_helpers.draw_asymmetrically(temp_df, 'age', 'age_err1', 'age_err2', 'age') # use this bc the psps version uses the same name as our input column
    temp_df['stellar_mass'] = np.random.normal(temp_df['Mass'], temp_df['e_Mass'])
    temp_df['teff_drawn'] = np.random.normal(temp_df['Teff'], temp_df['e_Teff'])

    temp_df['kepler_or_k2'] = np.where(temp_df['Kepler_ID'] > 0, 'Kepler', 'K2')
    temp_kepler = temp_df.loc[temp_df['kepler_or_k2']=='Kepler']
    temp_k2 = temp_df.loc[temp_df['kepler_or_k2']=='K2']
    temp_k2 = pd.merge(temp_k2, k2_pointings[['Campaign','baseline','logistic_a','logistic_k','logistic_l']], on='Campaign', how='left')

    ### create a Population object to hold information about the occurrence law governing that specific population
    # THIS IS WHERE YOU CHOOSE THE PLANET FORMATION HISTORY MODEL 
    pop = Population(temp_df['age'], threshold, frac1, frac2)
    pop_kepler = Population(temp_kepler['age'], threshold, frac1, frac2)
    pop_k2 = Population(temp_k2['age'], threshold, frac1, frac2)

    frac_hosts = pop.galactic_occurrence_step(threshold, frac1, frac2)
    #frac_hosts = pop.galactic_occurrence_monotonic(frac1, frac2)
    #frac_hosts = pop.galactic_occurrence_piecewise(frac1, frac2, threshold)
    frac_hosts_kepler = pop_kepler.galactic_occurrence_step(threshold, frac1, frac2)
    #frac_hosts_kepler = pop_kepler.galactic_occurrence_monotonic(frac1, frac2)
    #frac_hosts_kepler = pop_kepler.galactic_occurrence_piecewise(frac1, frac2, threshold)
    frac_hosts_k2 = pop_k2.galactic_occurrence_step(threshold, frac1, frac2)
    #frac_hosts_k2 = pop_k2.galactic_occurrence_monotonic(frac1, frac2)
    #frac_hosts_k2 = pop_k2.galactic_occurrence_piecewise(frac1, frac2, threshold)

    intact_fracs = scipy.stats.truncnorm.rvs(0, 1, loc=0.18, scale=0.1, size=len(temp_df))  
    intact_fracs_kepler = scipy.stats.truncnorm.rvs(0, 1, loc=0.18, scale=0.1, size=len(temp_kepler))  
    intact_fracs_k2 = scipy.stats.truncnorm.rvs(0, 1, loc=0.18, scale=0.1, size=len(temp_k2))  

    alpha_se = np.random.normal(-1., 0.2)
    alpha_sn = np.random.normal(-1.5, 0.1)

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
        star = K2Star(temp_k2['age'].iloc[i], temp_k2['stellar_radius'].iloc[i], temp_k2['stellar_mass'].iloc[i], temp_k2['teff_drawn'].iloc[i], temp_k2['CDPP6'].iloc[i], temp_k2['height'].iloc[i], alpha_se, alpha_sn, frac_hosts_k2[i], intact_fracs_k2[i], temp_k2['Campaign'].iloc[i], temp_k2['baseline'].iloc[i], temp_k2['source_id_dr3'].iloc[i], temp_k2['EPIC_ID'].iloc[i], temp_k2['logistic_a'].iloc[i], temp_k2['logistic_k'].iloc[i], temp_k2['logistic_l'].iloc[i])
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

    star_data_kepler = []
    for i in tqdm(range(len(temp_kepler))): # 100
        star = Star(temp_kepler['age'].iloc[i], temp_kepler['stellar_radius'].iloc[i], temp_kepler['stellar_mass'].iloc[i], temp_kepler['teff_drawn'].iloc[i], temp_kepler['CDPP6'].iloc[i], temp_kepler['height'].iloc[i], alpha_se, alpha_sn, frac_hosts_kepler[i], intact_fracs_kepler[i], temp_kepler['source_id_dr3'].iloc[i], temp_kepler['Kepler_ID'].iloc[i])
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
    #print(berger_kepler_all[['num_planets', 'periods', 'planet_radii', 'se_or_sn']])
    #"""

    #berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all.num_planets > 0]
    berger_kepler_planets = berger_kepler.loc[berger_kepler.num_planets > 0]
    berger_k2_planets = berger_k2.loc[berger_k2.num_planets > 0]
    #berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
    #berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 4.]

    berger_kepler_all = pd.concat([berger_kepler, berger_k2])
    #print(berger_kepler_all)
    #f = len(berger_kepler_planets)/len(berger_kepler_all)
    #print("f: ", f)

    berger_kepler_all.to_csv(path+'data/joint/stellar_samples/'+name+'/'+name+'_'+str(j)+'.csv')