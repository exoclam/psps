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

from psps.transit_class import Population, Star
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

"""
berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell

# Berger+ 2020 sample has lots of stellar params we need, but no source_id
berger = Table.read(path+'data/berger_kepler_stellar_fgk.csv')
# Bedell cross-match has the Gaia DR3 source_id we need to calculate Zmax with Gala
megan = Table.read(path+'data/kepler_dr3_good.fits')

# cross-match the cross-matches (we only lose ~100 stars)
merged = join(berger, megan, keys='kepid')
merged.rename_column('parallax_2', 'parallax')
berger_kepler = berger_kepler.loc[berger_kepler['kepid'].isin(merged['kepid'])]
"""

# actually, I just ran gala once and that's all I needed
berger_kepler = pd.read_csv(path+'data/berger_kepler_bedell_gala.csv') # now with gala heights!
berger_kepler = berger_kepler.dropna(subset=['height'])

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

# planet formation history model parameters
threshold = 5.5 # cosmic age in Gyr; 13.7 minus stellar age, then round
frac1 = 0.05 # frac1 must be < frac2 if comparing cosmic ages
frac2 = 0.55

name_thresh = 55
name_f1 = 5
name_f2 = 55
name = 'step_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2)
#name = 'monotonic_'+str(name_f1)+'_'+str(name_f2)
name = 'piecewise_'+str(name_thresh)+'_'+str(name_f1)+'_'+str(name_f2)

physical_planet_occurrences = []
physical_planet_occurrences_precut = []
detected_planet_occurrences_all = []
adjusted_planet_occurrences_all = []
transit_multiplicities_all = []
geom_transit_multiplicities_all = []
completeness_all = []
# for each model, draw around stellar age errors 10 times
for j in tqdm(range(5)): 

    # draw stellar radius, mass, and age using asymmetric errors 
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'iso_rad', 'iso_rad_err1', 'iso_rad_err2', 'stellar_radius')
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_age', 'iso_age_err1', 'iso_age_err2', 'age')
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'stellar_mass')

    # enrich berger_kepler with z_maxes using gala
    #z_maxes = simulate_helpers.gala_galactic_heights(merged, output=False)
    #berger_kepler_temp['height'] = z_maxes # kpc

    # I need to plot Figs 1 & 2; usually don't turn this on
    #print("before dropping heights: ", len(berger_kepler_temp))
    #berger_kepler_temp = berger_kepler_temp.dropna(subset='height')
    #print("after dropping heights: ", len(berger_kepler_temp))
    #utils.plot_properties(berger_kepler['iso_teff'], berger_kepler['iso_age'])

    ### create a Population object to hold information about the occurrence law governing that specific population
    # THIS IS WHERE YOU CHOOSE THE PLANET FORMATION HISTORY MODEL YOU WANT TO FORWARD MODEL
    pop = Population(berger_kepler_temp['age'], threshold, frac1, frac2)
    #frac_hosts = pop.galactic_occurrence_step(threshold, frac1, frac2)
    #frac_hosts = pop.galactic_occurrence_monotonic(frac1, frac2)
    frac_hosts = pop.galactic_occurrence_piecewise(frac1, frac2, threshold)
    intact_fracs = scipy.stats.truncnorm.rvs(0, 1, loc=0.18, scale=0.1, size=len(berger_kepler_temp))  # np vs JAX bc of random key issues

    alpha_se = np.random.normal(-1., 0.2)
    alpha_sn = np.random.normal(-1.5, 0.1)

    # create Star objects, with their planetary systems
    star_data = []
    for i in tqdm(range(len(berger_kepler_temp))): # 100
        star = Star(berger_kepler_temp['age'][i], berger_kepler_temp['stellar_radius'][i], berger_kepler_temp['stellar_mass'][i], berger_kepler_temp['rrmscdpp06p0'][i], berger_kepler_temp['height'][i], alpha_se, alpha_sn, frac_hosts[i], intact_fracs[i], berger_kepler_temp['kepid'][i])
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

    # do this thing where I make B20 look like TRI, instead of the other way around
    berger_kepler_all = berger_kepler_all.loc[berger_kepler_all['age'] <= 8.]

    berger_kepler_planets = berger_kepler_all.loc[berger_kepler_all.num_planets > 0]
    #berger_kepler_planets = berger_kepler_planets.loc[(berger_kepler_planets['periods'] <= 40) & (berger_kepler_planets['periods'] > 1)] # limit periods to fairly compare with Zink+ 2023
    #berger_kepler_planets = berger_kepler_planets.loc[berger_kepler_planets['planet_radii'] <= 4.]

    #print(berger_kepler_all)
    f = len(berger_kepler_planets)/len(berger_kepler_all)
    print("f: ", f)
    #quit()
    berger_kepler_all.to_csv(path+'data/berger_gala2/'+name+'/'+name+'_'+str(j)+'.csv')
    #quit()





