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
from tqdm import tqdm

from itertools import zip_longest
import numpy.ma as ma # for masked arrays

from astropy.table import Table, join

from psps.transit_class import Population, Star
import psps.simulate_helpers as simulate_helpers

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

name = 'step_11_20_85'
#"""
# diagnostic plotting age vs height
berger_kepler_all = pd.read_csv(path+'data/berger_gala/'+name+'.csv')
num_hosts = berger_kepler_all.loc[berger_kepler_all['num_planets']>0]
print("f: ", len(num_hosts)/len(berger_kepler_all))
berger_kepler_all = berger_kepler_all.dropna(subset=['height'])
berger_kepler_all['height'] = berger_kepler_all['height'] * 1000 # to pc

berger_kepler_all['periods'] = berger_kepler_all['periods'].apply(literal_eval_w_exceptions)
berger_kepler_all['planet_radii'] = berger_kepler_all['planet_radii'].apply(literal_eval_w_exceptions)
berger_kepler_all['incls'] = berger_kepler_all['incls'].apply(literal_eval_w_exceptions)
berger_kepler_all['mutual_incls'] = berger_kepler_all['mutual_incls'].apply(literal_eval_w_exceptions)
berger_kepler_all['eccs'] = berger_kepler_all['eccs'].apply(literal_eval_w_exceptions)
berger_kepler_all['omegas'] = berger_kepler_all['omegas'].apply(literal_eval_w_exceptions)

plt.hist2d(berger_kepler_all['height'], berger_kepler_all['age'], bins=40)
plt.xlabel('height [pc]')
plt.ylabel('age [Gyr]')
plt.xscale('log')
plt.xlim([0, 1500])
plt.ylim([0, 14])
#plt.savefig(path+'plots/trilegal_height_age.png')
plt.show()

zink_sn_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([38, 29, 23, 24, 17]), 'occurrence_err1': np.array([5, 3, 2, 2, 4]), 'occurrence_err2': np.array([6, 3, 2, 4, 4])})
zink_se_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': np.array([28, 29, 25, 27, 18]), 'occurrence_err1': np.array([5, 3, 3, 4, 4]), 'occurrence_err2': np.array([5, 3, 3, 3, 4])})
zink_kepler_occurrence = np.array([38, 29, 23, 24, 17])+np.array([28, 29, 25, 27, 18])
zink_kepler_occurrence_err1 = np.round(np.sqrt((zink_sn_kepler['occurrence_err1'])**2 + (zink_se_kepler['occurrence_err1']**2)), 2)
zink_kepler_occurrence_err2 = np.round(np.sqrt((zink_sn_kepler['occurrence_err2'])**2 + (zink_se_kepler['occurrence_err2']**2)), 2)
zink_kepler = pd.DataFrame({'scale_height': np.array([120., 200., 300., 500., 800.]), 'occurrence': zink_kepler_occurrence, 'occurrence_err1': zink_kepler_occurrence_err1, 'occurrence_err2': zink_kepler_occurrence_err2})

height_bins = np.array([0., 150, 250, 400, 650, 3000])
berger_kepler_all['height_bins'] = pd.cut(berger_kepler_all['height'], bins=height_bins, include_lowest=True)
berger_kepler_counts = np.array(berger_kepler_all.groupby(['height_bins']).count().reset_index()['kepid'])

# isolate planet hosts and bin them by galactic height
berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all['num_planets'] > 0]
berger_kepler_planets = berger_kepler_planets.explode(['periods', 'planet_radii', 'incls', 'mutual_incls', 'eccs', 'omegas']).reset_index(drop=True)
berger_kepler_planets_counts_precut = np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid'])

berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 4.] # limit radii to fairly compare with SEs in Zink+ 2023 (2)...or how about include SNs too (4)?
berger_kepler_planets_counts = np.array(berger_kepler_planets.groupby(['height_bins']).count().reset_index()['kepid'])

physical_planet_occurrence = 100 * berger_kepler_planets_counts/berger_kepler_counts # normally yes
print("physical planet occurrence: ", physical_planet_occurrence)

plt.errorbar(x=zink_kepler['scale_height'], y=zink_kepler['occurrence'], yerr=(zink_kepler['occurrence_err1'], zink_kepler['occurrence_err2']), fmt='o', capsize=3, elinewidth=1, markeredgewidth=1, label='Zink+ 2023 Kepler data')
plt.scatter(x=zink_kepler['scale_height'], y=physical_planet_occurrence, c='red', label='model')
plt.xlabel(r'$Z_{max}$ [pc]')
plt.ylabel('planets per 100 stars')
plt.legend()
plt.tight_layout()
plt.savefig(path+'plots/'+name)
plt.show()

quit()
#"""

berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell
# Berger+ 2020 sample has lots of stellar params we need, but no source_id
berger = Table.read(path+'data/berger_kepler_stellar_fgk.csv')
# Bedell cross-match has the Gaia DR3 source_id we need to calculate Zmax with Gala
megan = Table.read(path+'data/kepler_dr3_good.fits')

# cross-match the cross-matches (we only lose ~100 stars)
merged = join(berger, megan, keys='kepid')
merged.rename_column('parallax_2', 'parallax')
berger_kepler = berger_kepler.loc[berger_kepler['kepid'].isin(merged['kepid'])]

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

# planet formation history model parameters
threshold = 11. # cosmic age in Gyr; 13.7 minus stellar age, then round
frac1 = 0.2 # frac1 must be < frac2 if comparing cosmic ages
frac2 = 0.85

# send da video
physical_planet_occurrences = []
physical_planet_occurrences_precut = []
detected_planet_occurrences_all = []
adjusted_planet_occurrences_all = []
transit_multiplicities_all = []
geom_transit_multiplicities_all = []
completeness_all = []
# for each model, draw around stellar age errors 10 times
for j in tqdm(range(1)): 

    # draw stellar radius, mass, and age using asymmetric errors 
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'iso_rad', 'iso_rad_err1', 'iso_rad_err2', 'stellar_radius')
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_age', 'iso_age_err1', 'iso_age_err2', 'age')
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'stellar_mass')

    # enrich berger_kepler with z_maxes using gala
    z_maxes = simulate_helpers.gala_galactic_heights(merged, output=False)
    berger_kepler_temp['height'] = z_maxes # kpc

    ### create a Population object to hold information about the occurrence law governing that specific population
    # THIS IS WHERE YOU CHOOSE THE PLANET FORMATION HISTORY MODEL YOU WANT TO FORWARD MODEL
    pop = Population(berger_kepler_temp['age'], threshold, frac1, frac2)
    frac_hosts = pop.galactic_occurrence_step(threshold, frac1, frac2)

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
    berger_kepler_all.to_csv(path+'data/berger_gala/'+name+'.csv')






