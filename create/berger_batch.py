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

path = '/Users/chrislam/Desktop/mastrangelo/' # new computer has different username

# we're gonna need this for reading in the initial Berger+ 2020 data
def literal_eval_w_exceptions(x):
    try:
        return literal_eval(str(x))   
    except Exception as e:
        pass

berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell
# Berger+ 2020 sample has lots of stellar params we need, but no source_id
berger = Table.read(path+'data/berger_kepler_stellar_fgk.csv')
# Bedell cross-match has the Gaia DR3 source_id we need to calculate Zmax with Gala
megan = Table.read(path+'data/kepler_dr3_good.fits')

# cross-match the cross-matches (we only lose ~100 stars)
merged = join(berger, megan, keys='kepid')
berger_kepler = berger_kepler.loc[berger_kepler['kepid'].isin(merged['kepid'])]

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

# planet formation history model parameters
threshold = 10. # cosmic age in Gyr; 13.7 minus stellar age, then round
frac1 = 0.15 # frac1 must be < frac2 if comparing cosmic ages
frac2 = 0.55

# send da video
physical_planet_occurrences = []
physical_planet_occurrences_precut = []
detected_planet_occurrences_all = []
adjusted_planet_occurrences_all = []
transit_multiplicities_all = []
geom_transit_multiplicities_all = []
completeness_all = []
# for each model, draw around stellar age errors 10 times
for j in range(30): # 10

    # draw stellar radius, mass, and age using asymmetric errors 
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'iso_rad', 'iso_rad_err1', 'iso_rad_err2', 'stellar_radius')
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_age', 'iso_age_err1', 'iso_age_err2', 'age')
    berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'stellar_mass')

    # enrich berger_kepler with z_maxes using gala
    z_maxes = simulate_helpers.gala_galactic_heights(berger_kepler_temp, output=False)
    berger_kepler_temp['height'] = z_maxes * 1000 # pc

    ### create a Population object to hold information about the occurrence law governing that specific population
    # STEP
    pop = Population(berger_kepler_temp['age'], threshold, frac1, frac2)
    frac_hosts = pop.galactic_occurrence_step(threshold, frac1, frac2)

