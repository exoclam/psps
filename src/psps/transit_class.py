##################################################
### Functions for the physical pipeline ##########
##################################################

import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform
from math import lgamma
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import timeit 
from datetime import datetime
import json
from tqdm import tqdm 
import forecaster

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

import psps.simulate_helpers as simulate_helpers

pylab_params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(pylab_params)

path = '/Users/chrislam/Desktop/psps/' 
#berger_kepler = pd.read_csv(input_path+'berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell
berger_kepler = pd.read_csv(path+'data/berger_kepler_stellar_fgk.csv') # crossmatched with Gaia via Bedell

# make berger_kepler more wieldy
berger_kepler = berger_kepler[['kepid', 'iso_teff', 'iso_teff_err1', 'iso_teff_err2','feh_x','feh_err1','feh_err2',
						     'iso_age', 'iso_age_err1', 'iso_age_err2', 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'rrmscdpp06p0', 'iso_rad', 'iso_rad_err1', 'iso_rad_err2']]
G = 6.6743e-8 # gravitational constant in cgs

# how many params, how many dims, initialize cube
ndim = 3
nparams = 3
cube = [0, 0, 0]

# create JAX random seed
key = jax.random.key(42)

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

class Population: 
    """

    Functions that make populations of planetary systems

    Attributes:
    - ages [jnp.array]: input stellar ages [Gyr]

    Output:
    - Population object, which is itself populated by Stars objects

    """

    def __init__(
        self, ages, threshold=None, frac1=None, frac2=None, **kwargs 
    ):
        self.ages = jnp.array(ages)
        self.threshold = threshold
        self.frac1 = frac1
        self.frac2 = frac2
        self.children = []

    def add_child(self, child): 
        self.children.append(child)
    
    def sculpting(self, df, m, b, cutoff, bootstrap): # adapted from Ballard et al in prep, log version
        """ 
        Calculate the probability of system being intact vs disrupted, based on its age and the sculpting law.

        Input:
        - df: DataFrame of stars, with age column "iso_age"
        - m: sculpting law slope [dex]
        - b: sculpting law initial intact probability
        - cutoff: sculpting law turnoff time [yrs]
        - bootstrap: do we draw probability of intactness based on iso_age (no bootstrap) or age (bootstrap)

        Output:
        - df: same as input, but with new column called prob_intact

        """
        
        if bootstrap == False:
            df['prob_intact'] = np.where(
                    ((df['iso_age']*1e9 > 1e8) & (df['iso_age']*1e9 <= cutoff)), b+m*(np.log10(df['iso_age']*1e9)-8), np.where(
                        df['iso_age']*1e9 > cutoff, b+m*(np.log10(cutoff)-8), b))

        elif bootstrap == True:
            df['prob_intact'] = np.where(
                    ((df['age']*1e9 > 1e8) & (df['age']*1e9 <= cutoff)), b+m*(np.log10(df['age']*1e9)-8), np.where(
                        df['age']*1e9 > cutoff, b+m*(np.log10(cutoff)-8), b))

        # handle impossible probabilities
        df['prob_intact'] = np.where(
            df['prob_intact']<0, 0, np.where(
                df['prob_intact']>1, 1, df['prob_intact']))
                
        return df

    def galactic_occurrence_step(self, threshold, frac1, frac2):
        """
        Calculate the probability of system having planets, based on its age and three free parameters
        
        Input:
        - threshold: age beyond which probability of hosting a planet is frac2, versus frac1, in Gyr [float]
        - frac1: planet host fraction among systems younger than threshold [float]
        - frac2: planet host fraction among systems older than threshold [float]

        Output:
        - host_frac: jnp.array of fraction of planet hosts [float]

        """

        ages = self.ages

        # convert stellar ages to cosmic ages...but first make sure none are older than Universe
        ages = ages.at[ages > 13.7].set(13.7)
        cosmic_ages = 13.7 - ages

        host_frac = jnp.where(cosmic_ages <= threshold, frac1, frac2)

        """
        f, ((ax)) = plt.subplots(1, 1, figsize=(10, 5))
        x = np.linspace(0, 14, 1000)
        y = np.where(x <= threshold, frac1, frac2)
        plt.plot(x, y, color='powderblue')
        plt.xlabel('stellar age [Gyr]')
        plt.ylabel('planet host fraction')
        plt.title('f=%1.2f' % frac1 + ' if <=%i ' % threshold + 'Gyr; f=%1.2f' % frac2 + ' if >%i ' % threshold + 'Gyr') 
        plt.ylim([0, 1.05])
        plt.savefig(path+'galactic-occurrence/plots/step-model2.png')
        plt.show()
        quit()
        """

        return host_frac

    def galactic_occurrence_monotonic(self, y1, y2):
        """
        Read off the probability of system having planets, based on a monotonic rise in planet formation with cosmic time.
        a la m12z (the lowest mass galaxy in Garrison-Kimmel+ 2021). 

        Inputs:
        - y1: initial fraction of planet hosts, at cosmic time of 0 Gyrs
        - y2: present-day fraction of planet hosts, at cosmic time of 14 Gyrs

        Output:
        - host_frac: jnp.array of fraction of planet hosts [float]

        """

        ages = self.ages
        x = np.linspace(0, 14, 1000)

        # calculate slope in log space
        m = (y2-y1)/14.

        # model as a function of cosmic time, before applying to stellar sample
        """
        y = m * x + y1
        plt.plot(x, y, color='steelblue')
        plt.xlabel('cosmic age [Gyr]')
        plt.ylabel('planet host fraction')
        plt.ylim([0,1])
        #plt.savefig(path+'galactic-occurrence/plots/monotonic-model1.png')
        plt.show()
        quit()
        """

        # convert stellar ages to cosmic ages...but first make sure none are older than Universe
        ages = ages.at[ages > 14.].set(14.)
        cosmic_ages = 14. - ages

        # calculate host fraction as a function of cosmic age
        #host_frac = m * ages
        host_frac = m * cosmic_ages + y1
        print("mean f: ", np.mean(host_frac))

        # should we add burstiness? perhaps via a GP kernel with correlated noise turned up? 

        plot_df = pd.DataFrame({'cosmic_ages': cosmic_ages, 'host_frac': host_frac, 'stellar_ages': ages}).sort_values(by=['cosmic_ages']).reset_index()

        # flip back host_frac so that it corresponds to stellar age once again; need to turn into array, otherwise it doesn't stick
        plot_df['host_frac_reverse'] = np.array(plot_df['host_frac'][::-1])

        """
        f, ((ax)) = plt.subplots(1, 1, figsize=(10, 5))
        plt.plot(plot_df['cosmic_ages'][::100], plot_df['host_frac_reverse'][::100], color='powderblue')
        plt.xlabel('stellar age [Gyr]')
        plt.ylabel('planet host fraction')
        plt.ylim([0,1])
        plt.savefig(path+'galactic-occurrence/plots/monotonic-model1.png')
        plt.show()
        quit()
        """

        #return np.array(plot_df['host_frac_reverse'])
        return host_frac

    
    def galactic_occurrence_piecewise(self, y1, y2, threshold):
        """
        Planet formation starts constant, then a merger-like event causes planet formation to increase, potentially to a plateau. 

        Inputs:
        - y1: initial fraction of planet hosts, at cosmic time of 0 Gyrs
        - y2: present-day fraction of planet hosts
        - threshold: piecewise knee location in Gyrs, in cosmic time

        Output:
        - host_frac: jnp.array of fraction of planet hosts [float]

        """

        ages = self.ages
        x = np.linspace(0, 14, 1000)

        # calculate slope in log space
        m = (y2-y1)/(14 - threshold)

        # model as a function of cosmic time, before applying to stellar sample
        y = np.where(x < threshold, y1, y1 + m * (x-threshold))
        """
        plt.plot(x, y, color='steelblue')
        plt.xlabel('cosmic age [Gyr]')
        plt.ylabel('planet host fraction')
        plt.ylim([0,1])
        plt.savefig(path+'galactic-occurrence/plots/piecewise-model1.png')
        #plt.show()
        """

        # convert stellar ages to cosmic ages...but first make sure none are older than Universe
        ages = ages.at[ages > 14.].set(14.)
        cosmic_ages = 14. - ages

        # calculate host fraction as a function of cosmic age
        host_frac = np.where(cosmic_ages < threshold, y1, y1 + m * (cosmic_ages-threshold))
        print("mean f: ", np.mean(host_frac))

        # should we add burstiness? perhaps via a GP kernel with correlated noise turned up? 

        plot_df = pd.DataFrame({'cosmic_ages': cosmic_ages, 'host_frac': host_frac, 'stellar_ages': ages}).sort_values(by=['cosmic_ages']).reset_index()

        # flip back host_frac so that it corresponds to stellar age once again; need to turn into array, otherwise it doesn't stick
        plot_df['host_frac_reverse'] = np.array(plot_df['host_frac'][::-1])

        """
        f, ((ax)) = plt.subplots(1, 1, figsize=(10, 5))
        plt.plot(plot_df['cosmic_ages'][::100], plot_df['host_frac_reverse'][::100], color='powderblue')
        plt.xlabel('stellar age [Gyr]')
        plt.ylabel('planet host fraction')
        plt.ylim([0,1])
        plt.savefig(path+'galactic-occurrence/plots/monotonic-model1.png')
        plt.show()
        quit()
        """

        #return np.array(plot_df['host_frac_reverse'])
        return host_frac


    def galactic_occurrence_bumpy(self, xs, ys):
        """
        Read off the probability of system having planets, based on a PDF built from MW-like galaxy simulations, eg. FIRE
        We use m12i from Ma+ 2017 and Garrison-Kimmel+ 2021. 

        Input: 
        - xs: cosmic age, in Gyr [np.array of floats]
        - ys: star formation rate [np.array of floats]

        Output:
        - host_frac: jnp.array of fraction of planet hosts [float]

        """

        ages = self.ages

        # convert stellar ages to cosmic ages...but first make sure none are older than Universe
        ages = ages.at[ages > 14.].set(14.)
        ages = 14. - ages

        # snap age to nearest xs grid; also, np.searchsorted() needs indices to be in ascending order
        #x = np.searchsorted(xs[::-1], ages, side = "right")
        x = np.searchsorted(xs, ages)
        x[x >= 1000] = 999

        # get corresponding y value
        host_frac = ys[x]

        plot_df = pd.DataFrame({'ages': ages, 'host_frac': host_frac}).sort_values(by=['ages']).reset_index()

        # flip back host_frac so that it corresponds to stellar age once again; need to turn into array, otherwise it doesn't stick
        plot_df['host_frac_reverse'] = np.array(plot_df['host_frac'][::-1])
        print("mean f: ", np.mean(plot_df['host_frac_reverse']))

        """
        f, ((ax)) = plt.subplots(1, 1, figsize=(10, 5))
        plt.plot(plot_df['ages'], plot_df['host_frac_reverse'], color='powderblue')
        plt.xlabel('stellar age [Gyr]')
        plt.ylabel('planet host fraction')
        plt.savefig(path+'galactic-occurrence/plots/bumpy-model1.png')
        plt.show()
        quit()
        """
        return host_frac
        
    
class Star:
    """

    Functions that make planetary systems, at the per-star level

    Attributes, first four from Berger+ 2020 sample:
    - kepid: Kepler identifier
    - age: drawn stellar age, in Gyr
    - stellar_radius: drawn stellar radius, in Solar radii
    - stellar_mass: drawn stellar mass, in Solar masses
    - rrmscdpp06p0: 6-hr-window CDPP noise [ppm]
    - frac_host: calculated fraction of planet hosts 
    - height: galactic scale height [pc]
    - alpha_se: power law exponent for Super-Earth radii
    - alpha_sn: power law exponent for Sub-Neptune radii
    - kepid: default to None, unless user provides Kepler IDs

    Output:
    - Star object, which is populated by Planets

    """

    def __init__(
        self, age, stellar_radius, stellar_mass, rrmscdpp06p0, frac_host, height, alpha_se, alpha_sn, kepid=None, **kwargs 
    ):

        #self.kepid = kepid
        self.age = age
        self.stellar_radius = np.array(stellar_radius)
        self.stellar_mass = np.array(stellar_mass)
        self.rrmscdpp06p0 = np.array(rrmscdpp06p0)
        self.frac_host = frac_host
        self.height = np.array(height)
        self.alpha_se = alpha_se
        self.alpha_sn = alpha_sn
        self.kepid = np.array(kepid)
        #print(self.kepid, self.stellar_radius, self.stellar_mass, self.rrmscdpp06p0, self.height)

        #self.midplane = jax.random.uniform(key, minval=-np.pi/2, maxval=np.pi/2)
        self.midplane = np.random.uniform(low=-np.pi/2, high=np.pi/2) # JAX, but I need to figure out how to properly randomly draw

        # prescription for planet-making
        #prob_intact = 0.18 + 0.1 * jax.random.truncated_normal(key=subkey, lower=0, upper=1) # from Lam & Ballard 2024; out of planet hosts
        prob_intact = scipy.stats.truncnorm.rvs(0, 1, loc=0.18, scale=0.1)  # np vs JAX bc of random key issues
        self.prob_intact = prob_intact

        p = simulate_helpers.assign_status(self.frac_host, self.prob_intact)
        self.status = np.random.choice(['no-planet', 'intact', 'disrupted'], p=p)
        #self.intact_flag = assign_flag(key, self.prob_intact, self.frac_host)

        # assign system-level inclination spread based on intact flag
        self.sigma_incl = jnp.where(self.status=='intact', jnp.pi/90, jnp.pi/22.5) # no-planets will also have disrupted spread; need to figure out nested wheres in JAX

        # assign number of planets per system based on intact flag
        self.num_planets = simulate_helpers.assign_num_planets(self.status)

        """
        # populate the Planets here upon initialization
        for i in range(self.num_planets):
            planet = Planet(self.midplane, self.intact_flag, self.sigma_incl)
            self.planets.append(planet)
        """
        if self.num_planets!=None:

            ### draw planet radii and periods such that they satisfy a Hill radius check and satisfy population statistics of radius valley. also draw planet masses.
            self.periods, self.planet_radii, self.planet_masses = simulate_helpers.draw_planet_periods_and_radii(self.num_planets, self.alpha_se, self.alpha_sn, self.stellar_mass)

            # draw inclinations from Gaussian distribution centered on midplane (invariable plane)        
            mu = self.midplane
            sigma = self.sigma_incl
            #self.incls = mu + sigma * jax.random.normal(key, shape=(self.num_planets,)) # JAX, but I need to figure out how to properly randomly draw
            self.incls = np.random.normal(loc=mu, scale=sigma, size=self.num_planets)
            
            # obtain mutual inclinations for plotting to compare {e, i} distributions
            self.mutual_incls = self.midplane - self.incls

            # draw eccentricity
            if (model_flag=='limbach-hybrid') | (model_flag=='limbach'):
                # for drawing eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
                #limbach = pd.read_csv(input_path+'limbach_cdfs.txt', engine='python', header=0, sep='\s{2,20}') # space-agnostic separator
                limbach = pd.read_csv(path+'data/limbach_cdfs.txt', engine='python', header=0, sep='\s{2,20}') # space-agnostic separator
                self.eccs = simulate_helpers.draw_eccentricity_van_eylen_vectorized(model_flag, self.num_planets, limbach)
            else:
                self.eccs = simulate_helpers.draw_eccentricity_van_eylen_vectorized(model_flag, self.num_planets)

            # draw longitudes of periastron
            #self.omegas = jax.random.uniform(key, shape=(self.num_planets,), minval=0, maxval=2*jnp.pi) # JAX, but I need to figure out how to properly randomly draw
            self.omegas = np.random.uniform(low=0, high=2*np.pi, size=self.num_planets)

            # turn to comma-delimited lists for ease of reading in later
            self.incls = self.incls.tolist()
            self.periods = self.periods.tolist()
            self.planet_radii = self.planet_radii.tolist()
            self.mutual_incls = self.mutual_incls.tolist()
            self.eccs = self.eccs.tolist()
            self.omegas = self.omegas.tolist()
            self.planet_masses = self.planet_masses.tolist()
            
        else:
            self.periods = None
            self.planet_radii = None
            self.incls = None
            self.mutual_incls = None
            self.eccs = None
            self.omegas = None
            self.planet_masses = None

    def assign_num_planets(x):
        """
        Based on the status (no planet, dynamically cool, dynamically hot), assign the number of planets

        Input:
        - status: output of assign_status [str]
        
        Output:
        - num_planet: number of planets in the system [int]

        """

        if x=='intact':
            return jax.random.choice([5, 6])
        elif x=='disrupted':
            return jax.random.choice([1, 2])
        elif x=='no-planets':
            return 0

    def reprJSON(self):
        return dict(kepid=self.kepid, age=self.age, frac_host=self.frac_host, midplane=self.midplane, prob_intact=self.prob_intact,
        status=self.status, sigma_incl=self.sigma_incl, num_planets=self.num_planets, periods=self.periods, planet_radii=self.planet_radii, incls=self.incls, 
        mutual_incls=self.mutual_incls, eccs=self.eccs, omegas=self.omegas)  


class Planet:
    """
    Do I need Planets to be a class? For now, no.
    """

    def __init__(
        self, **kwargs 
    ):

        # draw period from loguniform distribution from 2 to 300 days
        self.periods = df.num_planets.apply(lambda x: np.array(loguniform.rvs(2, 300, size=x)))

        # calculate mutual inclination
        self.incl = jaxnp.random.normal(midplane, sigma, 1)
        self.mutual_incl = Star.midplane - self.incl

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

