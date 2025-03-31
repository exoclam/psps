##################################################
### Helper functions only ########################
##################################################

import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform
from scipy.interpolate import CubicSpline
from math import lgamma
import jax
import jax.numpy as jnp
from tqdm import tqdm
import forecaster
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import Table, join
from pyia import GaiaData
import gala.dynamics as gd
import gala.potential as gp

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#path = '/blue/sarahballard/c.lam/sculpting2/'
path = '/Users/chrislam/Desktop/mastrangelo/' # new computer has different username

"""
# Create sample bank of eccentricities to draw from so that you don't call np.searchsorted a bajillion times
# For drawing eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
limbach = pd.read_csv(path+'limbach_cdfs.txt', engine='python', header=0, sep='\s{2,20}') # space-agnostic separator
values = np.random.rand(10000)
value_bins1 = np.searchsorted(limbach['1'], values)
value_bins2 = np.searchsorted(limbach['2'], values)
value_bins5 = np.searchsorted(limbach['5'], values)
value_bins6 = np.searchsorted(limbach['6'], values)
"""

### helper conversion functions
def p_to_a(P, star_mass):
    """
    Newton's full version of Kepler's Third Law, assuming planet mass m --> 0
    Params: 
    - P: days
    - star_mass: Solar masses
    """

    P = P*86400 # days to seconds
    star_mass = star_mass*1.989e30 # solar mass to kg
    a_in_meters = (((P**2) * 6.67e-11 * star_mass)/(4*np.pi**2))**(1./3) # SI units(!)
    a = a_in_meters/(1.496e11) # meters to AU
    return a # in AU

def solar_radius_to_au(radius):
    return 0.00465047*radius

def earth_radius_to_au(radius):
    return 4.26352e-5*radius

def earth_mass_to_cgs(mass):
    return mass*5.974e27 # grams

def solar_mass_to_cgs(mass):
    return mass*1.989e33 # grams

def au_to_cgs(distance):
    return distance*1.496e13 # cm

def kpc_to_km(distance):
    return distance * 3.086e16

def myr_to_s(t):
    return t * 3.154e7 * 1e6

def kpc_per_myr_to_km_per_s(distance, t):
    return kpc_to_km(distance)/myr_to_s(t)

def degrees_to_radians(deg):
    return deg*np.pi/180

def radians_to_degrees(rad):
    return rad*180/np.pi

def m_to_M(m, dist):
    # convert apparent to absolute magnitude
    return m - 5 * np.log10(dist) + 5

### basic physics equations
def distance_modulus_to_distance(mu, perturb=False):
    """
    Input:
    - mu: distance modulus, or m-M0
    Output:
    - d: distance, in pc
    """

    if perturb==True:
        # from Luke Bouma (https://github.com/lgbouma/gyrojo/blob/main/gyrojo/trilegal.py#L92-L108)
        # m-M0 from TRILEGAL has a bin size of 0.05 dex, so we should introduce a perturbation of that size.
        eps = np.random.normal(loc=0, scale=0.05, size=len(mu))
        mu += eps
        d = 10**(1+(mu/5))
    
    else:
        d = 10**(1+(mu/5))

    return d

def dist_kepler_to_height(d):
    """
    Kepler field has a fixed inclination of 13.5 deg. Given a distance modulus, or better yet a distance, we can use middle school math to get the height

    Inputs:
    - d: distance, in pc

    Output:
    - h: height off the Galactic midplane, in pc
    """

    incl = degrees_to_radians(13.5)
    h = d * np.sin(incl)

    return h

def stellar_radius_stefan_boltzmann(T, L):
    """
    Use Stefan-Boltzmann law to calculate stellar radius. 

    Input:
    - T: Teff in K
    - L: luminosity, in LSun
    Output: 
    - r: stellar radius in Solar radii
    """

    sigma = 5.67e-8 # W m^-2 K^-4
    LSun = 3.846e26 # W
    L = L * LSun
    denominator = 4*np.pi*sigma*T**4
    r = np.sqrt(L/denominator) # m
    r = r / 6.957e8 # RSun

    return r

def bolometric_mag_luminosity(mbol):
    """
    Convert bolometric magnitude to luminosity, in W
    """
    L0 = 3.0128e28 # W
    L = L0 * 10**(-mbol/2.512)

    return L

def bolometric_mag_luminosity2(mbol):
    """
    Convert bolometric magnitude to luminosity, in W
    """
    
    #LSun = 3.9e33 # erg/s
    LSun = 3.9e33 * 1e-7 # erg/s to W
    L = 10**((4.72 - mbol)/2.5)

    return L

def stellar_mass_luminosity(L):
    """
    Relate stellar mass and luminosity, in Solar units
    """
    LSun = 3.8e26 # Solar luminosity in W
    L = L / LSun # convert luminosity to Solar units

    Mstar = L**(-3.5) # in Solar masses
    #Mstar = Mstar * 1.989e30 # kg (W is associated with kg, not g)

    return Mstar 

### helper main functions
def compute_prob(x, m, b, cutoff): # adapted from Ballard et al in prep, log version
    # calculate probability of intact vs disrupted
    
    x = x*1e9
    if x <= 1e8: # we don't care about (nor do we have) systems before 1e8 years
        y = b 

    elif (x > 1e8) & (x <= cutoff): # pre-cutoff regime
        #print(np.log10(x_elt), m, b)
        y = b + m*(np.log10(x)-8)

    elif x > cutoff: # if star is older than cutoff, use P(intact) at cutoff time
        y = b + m*(np.log10(cutoff)-8)

    if y < 0: # handle negative probabilities
        y = 0
    elif y > 1: # handle probabilities greater than 1
        y = 1
            
    return y

def compute_prob_vectorized(df, m, b, cutoff, bootstrap): # adapted from Ballard et al in prep, log version
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

def assign_intact_flag(x):
    # call np.random.choice() for df.apply for model_vectorized()
    # x: df.prod_intact

    return np.random.choice(['intact', 'disrupted'], p=[x, 1-x])

def assign_flag(key, prob_intact, frac_host):
    """
    Assign intact vs disrupted vs no planets

    Inputs:
    - prob_intact: fraction of dynamically cool systems (4-13% for Sun-like Kepler stars)
    - frac_host: fraction of systems with planets

    Output: 
    - status: intact vs disrupted vs no planets [string]
    """

    status = np.random.choice(a=['no-planets', 'intact', 'disrupted'], p=[1-frac_host, frac_host*prob_intact, frac_host*(1-prob_intact)])
    return status

def assign_status(frac_host, prob_intact):
        """
        Label system as having no planet, dynamically cool system, or dynamically hot system.

        Inputs: 
        - frac_host: planet host fraction [float]
        - prob_intact: fraction of dynamically cool systems (out of planet hosts, not all stars) [float]

        Output:
        - p: list of probabilities of a system being each of the three potential statuses, 'no-planet', 'intact', 'disrupted'

        """
        

        p = [1-frac_host, frac_host*prob_intact, frac_host*(1-prob_intact)]
        p = np.asarray(p).astype('float64')
        p = p / np.sum(p)

        return p  

def assign_num_planets(x):
    # # call np.random.choice() for df.apply for model_vectorized()
    # x: df.intact_flag
    if x=='intact':
        return np.random.choice([5, 6])
    elif x=='disrupted':
        return np.random.choice([1, 2])
    elif x=='no-planets':
        return 0

def draw_inclinations_vectorized(midplane, sigma, num_planets):
    # call np.random.normal() for df.apply for model_vectorized()
    # x: 
    return np.random.normal(midplane, sigma, num_planets)

def redundancy_check(m, b, cutoff):
    # skip simulations if cutoff occurs more than once after probability has reached zero (use the first one for all)
    # also don't vary cutoffs if m is flat   

    y = b + m*(np.log10(cutoff)-8)
    if y < 0:
        return False
    elif m==0:
        return False
    else:
        return True

### helper physical transit functions
def calculate_eccentricity_limbach(multiplicity):
    """
    Draw eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
    Params: multiplicity of system (int)
    Returns: np.array of eccentricity values with length==multiplicity
    """
    # for drawing eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
    limbach = pd.read_csv(path+'limbach_cdfs.txt', engine='python', header=0, sep='\s{2,20}') # space-agnostic separator

    values = np.random.rand(multiplicity) # draw an eccentricity per planet
    if multiplicity==1:
        value_bins = np.searchsorted(limbach['1'], values) # return positions in cdf vector where random values should go
    elif multiplicity==2:
        value_bins = np.searchsorted(limbach['2'], values) # return positions in cdf vector where random values should go
    elif multiplicity==5:
        value_bins = np.searchsorted(limbach['5'], values) # return positions in cdf vector where random values should go
    elif multiplicity==6:
        value_bins = np.searchsorted(limbach['6'], values) # return positions in cdf vector where random values should go
    random_from_cdf = np.logspace(-2,0,101)[value_bins] # select x_d positions based on these random positions
    
    return random_from_cdf

def calculate_eccentricity_limbach_vectorized(multiplicity, limbach):
    """
    Draw eccentricities using Limbach & Turner 2014 CDFs relating e to multiplicity
    Params: multiplicity of system (array of ints); limbach (DataFrame of Limbach & Turner 2014 CDFs)
    Returns: np.array of eccentricity values with length==multiplicity
    """
    if multiplicity > 0:
        values = np.random.rand(multiplicity) # draw an eccentricity per planet

        if multiplicity==1:
            value_bins = np.searchsorted(limbach['1'], values) # return positions in cdf vector where random values should go
        elif multiplicity==2:
            value_bins = np.searchsorted(limbach['2'], values) # return positions in cdf vector where random values should go
        elif multiplicity==5:
            value_bins = np.searchsorted(limbach['5'], values) # return positions in cdf vector where random values should go
        elif multiplicity==6:
            value_bins = np.searchsorted(limbach['6'], values) # return positions in cdf vector where random values should go

        random_from_cdf = np.logspace(-2,0,101)[value_bins] # select x_d positions based on these random positions
        #df['ecc'] = df.apply(lambda x: np.logspace(-2,0,101)[value_bins]) # select x_d positions based on these random positions

        return random_from_cdf
        
    elif multiplicity == 0:
        return np.array([])

def calculate_eccentricity_limbach_vectorized_bank(multiplicity):
    """
    Draw eccentricities using test bank of Limbach & Turner 2014 CDFs
    Params: multiplicity of system (array of ints)
    Returns: np.array of eccentricity values with length==multiplicity
    (under permanent construction)
    """

    if multiplicity==1:
        ecc_indices = np.random.choice(value_bins1, 1)
    elif multiplicity==2:
        ecc_indices = np.random.choice(value_bins2, 2)
    elif multiplicity==5:
        ecc_indices = np.random.choice(value_bins5, 5)
    elif multiplicity==6:
        ecc_indices = np.random.choice(value_bins6, 6)
    
    random_from_cdf = np.logspace(-2,0,101)[ecc_indices] # select x_d positions based on these random positions
    #df['ecc'] = df.apply(lambda x: np.logspace(-2,0,101)[value_bins]) # select x_d positions based on these random positions
    
    return random_from_cdf

def draw_eccentricity_van_eylen_vectorized(model_flag, num_planets, *args):
    # *args is optional parameter of the limbach DataFrame for certain model_flags

    if model_flag=='limbach-hybrid':
        limbach = args[0]
        sigma_rayleigh = 0.26
        #draws = np.where(num_planets > 1, calculate_eccentricity_limbach(num_planets), np.random.rayleigh(sigma_rayleigh, num_planets))
        draws = np.where(num_planets > 2, calculate_eccentricity_limbach_vectorized(num_planets, limbach), np.random.rayleigh(sigma_rayleigh, num_planets))
        #draws = np.where(num_planets > 1, calculate_eccentricity_limbach_vectorized_bank(num_planets), np.random.rayleigh(sigma_rayleigh, num_planets))

    elif model_flag=='rayleigh':
        sigma_single = 0.24
        sigma_multi = 0.061
        if num_planets==1:
            sigma = sigma_single
        elif num_planets>1:
            sigma = sigma_multi

        draws= np.random.rayleigh(sigma, num_planets)

    return draws

def draw_eccentricity_van_eylen(model_flag, num_planets):
    """
    Draw eccentricities per the four models of Van Eylen et al 2018 (https://arxiv.org/pdf/1807.00549.pdf)
    Params: flag (string) saying which of the four models; num_planets (int)
    Returns: list eccentricity per planet in the system
    """
    if model_flag=='rayleigh':
        sigma_single = 0.24
        sigma_multi = 0.061
        if num_planets==1:
            sigma = sigma_single
        elif num_planets>1:
            sigma = sigma_multi
            
        draw = np.random.rayleigh(sigma, num_planets)

    elif model_flag=='half-Gaussian':
        sigma_single = 0.32
        sigma_multi = 0.083
        if num_planets==1:
            sigma = sigma_single
        elif num_planets>1:
            sigma = sigma_multi
            
        draw = np.random.normal(0, sigma, num_planets)
        if any(d < 0 for d in draw): # put the 'half' in half-Gaussian by redrawing if any draw element is negative
            draw = draw_eccentricity_van_eylen('half-Gaussian', num_planets)
        
    elif model_flag=='beta':
        a_single = 1.58
        b_single = 4.4
        a_multi = 1.52
        b_multi = 29
        
        # errors for pymc3 implementation of eccentricity draws, should I wish to get fancy.
        # I currently do not wish to get fancy.
        a_single_err1 = 0.59
        a_single_err2 = 0.93
        b_single_err1 = 1.8
        b_single_err2 = 2.2
        a_multi_err1 = 0.5
        a_multi_err2 = 0.85
        b_multi_err1 = 9
        b_multi_err2 = 17
        
        if num_planets==1:
            a = a_single
            b = b_single
        elif num_planets>1:
            a = a_multi
            b = b_multi
        
        draw = np.random.beta(a, b, num_planets)
        
    elif model_flag=='mixed':
        sigma_half_gauss = 0.049
        sigma_rayleigh = 0.26
        f_single = 0.76
        f_multi = 0.08
        
        if num_planets==1:
            draw = np.random.rayleigh(sigma_rayleigh, num_planets)
        elif num_planets>1:
            draw = np.random.normal(0, sigma_half_gauss, num_planets)
            if any(d < 0 for d in draw): # redraw if any planet's eccentricity is negative
                draw = draw_eccentricity_van_eylen('mixed', num_planets)
                
    elif model_flag=='limbach-hybrid':
        """
        Testing something for Sarah: use Rayleigh for intact and Limbach for disrupted. 
        """
        sigma_rayleigh = 0.26
        #print(num_planets)
        if num_planets==1:
            draw = np.random.rayleigh(sigma_rayleigh, num_planets)
        elif num_planets>1:
            draw = calculate_eccentricity_limbach(num_planets)

    elif model_flag=='limbach': # OG way of drawing eccentricities, from Limbach & Turner 2014
        draw = calculate_eccentricity_limbach(num_planets)

    elif model_flag=='absurd': # sanity check
        if num_planets==1:
            draw = np.random.uniform(0.09, 0.11, num_planets)
        elif num_planets>1:
            draw = np.random.uniform(0.009, 0.011, num_planets)
            
    return draw

def calculate_impact_parameter(star_radius, a, e, incl, omega, angle_flag): # following Winn 2010 Eqn 7
    """
    angle_flag: True means indexed at 0; False means indexed at pi/2
    """
    star_radius = solar_radius_to_au(star_radius)
    if angle_flag==True:
        factor1 = (a * np.cos(np.pi/2 - incl))/star_radius  # again, we're indexed at 0 rather than pi/2
    elif angle_flag==False: # if indexed at pi/2
        factor1 = (a * np.cos(incl))/star_radius 
    factor2 = (1-e**2)/(1+e*np.sin(omega)) # leave this alone, right? Bc everyone always assumes omega=pi/2?
    
    return factor1 * factor2

def calculate_impact_parameter_vectorized(star_radius, a, e, incl, omega, angle_flag): # following Winn 2010 Eqn 7
    """
    angle_flag: True means indexed at 0; False means indexed at pi/2
    """
    star_radius = solar_radius_to_au(star_radius)
    if angle_flag==True:
        factor1_temp = np.pi/2 - incl
        factor1 = (a * factor1_temp.apply(lambda x: np.cos(x)))/star_radius  # again, we're indexed at 0 rather than pi/2
    elif angle_flag==False: # if indexed at pi/2
        factor1_temp = incl
        factor1 = (a * factor1_temp.apply(lambda x: np.cos(x)))/star_radius 
    factor2 = (1-e**2)/(1+e*omega.apply(lambda x: np.sin(x))) # leave this alone, right? Bc everyone always assumes omega=pi/2?
    
    return factor1 * factor2

def calculate_transit_duration_vectorized(P, r_star, r_planet, b, a, inc, e, omega, angle_flag): # Winn 2011s Eqn 14 & 16
    """
    Calculate tdur, but make it snappy.

    Input:
    - P: period, days
    - r_star: stellar radius, AU
    - r_planet: planet radius, AU
    - b: impact parameter
    - a: semi-major axis, AU
    - inc: inclination, radians
    - e: eccentricity
    - omega: longitude of periastron, radians
    - angle_flag: True = indexed at 0; False = indexed at pi/2

    Output:
    - tdur: transit duration, hours

    """
    #print("take 1: ", r_planet/r_star)
    #print("take 2: ", (1+(r_planet/r_star))**2 - b**2)
    
    arg1_temp = (1+(r_planet/r_star))**2 - b**2
    arg1 = arg1_temp.apply(lambda x: np.sqrt(x))
    if angle_flag==True:
        arg2_temp = np.pi/2 - inc
        arg2 = (r_star / a) * (arg1 / arg2_temp.apply(lambda x: np.sin(x))) # account for us being indexed at 0 rather than pi/2
    elif angle_flag==False:
        arg2_temp = inc
        arg2 = (r_star / a) * (arg1 / arg2_temp.apply(lambda x: np.sin(x))) # assuming indexed at pi/2
    arg3_temp = 1-e**2
    arg3 = arg3_temp.apply(lambda x: np.sqrt(x))/(1+e*omega.apply(lambda x: np.sin(x))) # eccentricity factor from Eqn 16 of Winn 2011
    #print("Winn args: ", arg1, arg2, arg3)
    
    return (P / np.pi) * arg2.apply(lambda x: np.arcsin(x)) * arg3

def calculate_transit_duration(P, r_star, r_planet, b, a, inc, e, omega, angle_flag): # Winn 2011s Eqn 14 & 16
    #print("take 1: ", r_planet/r_star)
    #print("take 2: ", (1+(r_planet/r_star))**2 - b**2)
    
    arg1 = np.sqrt((1+(r_planet/r_star))**2 - b**2)
    if angle_flag==True:
        arg2 = (r_star / a) * (arg1 / np.sin(np.pi/2 - inc)) # account for us being indexed at 0 rather than pi/2
    elif angle_flag==False:
        arg2 = (r_star / a) * (arg1 / np.sin(inc)) # assuming indexed at pi/2
    arg3 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16 of Winn 2011
    #print("Winn args: ", arg1, arg2, arg3)
    
    return (P / np.pi) * np.arcsin(arg2) * arg3

def calculate_transit_duration_paw(P, star_radius, planet_radius, b, a, incl, e, omega): # Paul Anthony Wilson website: https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/
    arg1 = np.sqrt((star_radius+planet_radius)**2 - (b*star_radius)**2) 
    arg2 = arg1/a
    arg3 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16
    #print("PAW args: ", arg1, arg2, arg3)
    return (P / np.pi) * np.arcsin(arg2) * arg3

def calculate_transit_duration_he(P, star_radius, planet_radius, a, e, omega): # from Matthias He: https://github.com/ExoJulia/SysSimExClusters/tree/master/src
    arg1 = (P/np.pi) * (star_radius/a)
    arg2 = np.sqrt(1-e**2)/(1+e*np.sin(omega)) # eccentricity factor from Eqn 16
    arg2 = 1
    return arg1 * arg2

def calculate_amd(m_pks, m_star, a_ks, e_ks, i_ks, multiplicity):
    ### 
    # calculate angular momentum deficit following Eqn 13 from Milholland et al 2021
    ###
    amd = []
    for i in range(multiplicity):
        lambda_k = m_pks[i] * np.sqrt(G*m_star*a_ks[i])
        second_term = 1 - (np.sqrt(1 - (e_ks[i])**2))*np.cos(i_ks[i])
        amd.append(lambda_k * second_term)
        
    return np.sum(amd)

### helper transit detection functions
def calculate_sn(P, rp, rs, cdpp, tdur, unit_test_flag=False): 
    """
    Calculate S/N per planet using Eqn 4 in Christiansen et al 2012: https://arxiv.org/pdf/1208.0595.pdf
    
    Params: P (days); rp (Earth radii); rs (Solar radii); cdpp (ppm); tdur (days)
    
    Returns: S/N
    """
    tobs = 365*3.5 # days; time spanned observing the target; set to 3.5 years, or the length of Kepler mission
    f0 = 0.92 # fraction of time spent actually observing and not doing spacecraft things
    tcdpp = 0.25 # days; using CDPP for 6 hour transit durations; could change to be more like Earth transiting Sun?
    rp = solar_radius_to_au(rp) # earth_radius_to_au when not using Matthias's test set
    rs = solar_radius_to_au(rs)
    #print(P, rp, rs, cdpp, tdur)
    
    factor1 = np.sqrt(tobs*f0/np.array(P)) # this is the number of transits
    delta = 1e6*(rp/rs)**2 # convert from parts per unit to ppm
    cdpp_eff = cdpp * np.sqrt(tcdpp/tdur)
    #print("CDPP ingredients: ", cdpp, tcdpp, tdur)
    factor2 = delta/cdpp_eff
    sn = factor1 * factor2
    
    if unit_test_flag==True:
        if np.isnan(sn)==True:
            sn = 0
        return sn
    else:
        sn = sn.fillna(0)
        return sn

def calculate_sn_vectorized(P, rp, rs, cdpp, tdur, unit_test_flag=False): 
    
    """
    Calculate S/N per planet using Eqn 4 in Christiansen et al 2012: https://arxiv.org/pdf/1208.0595.pdf
    
    Params: P (days); rp (Earth radii); rs (Solar radii); cdpp (ppm); tdur (days)
    
    Returns: S/N
    """

    tobs = 365*3.5 # days; time spanned observing the target; set to 3.5 years, or the length of Kepler mission
    f0 = 0.92 # fraction of time spent actually observing and not doing spacecraft things
    tcdpp = 0.25 # days; using CDPP for 6 hour transit durations; could change to be more like Earth transiting Sun?
    rp = earth_radius_to_au(rp) # earth_radius_to_au when not using Matthias's test set; solar_radius_to_au when using Matthias's test set
    rs = solar_radius_to_au(rs)

    #factor1_old = tobs*f0/P.apply(lambda x: np.sqrt(x)) # this is the number of transits
    factor1 = (tobs*f0/P).apply(lambda x: np.sqrt(x)) # this is the number of transits

    delta = 1e6*(rp/rs)**2 # convert from parts per unit to ppm

    #cdpp_eff_old = cdpp * tcdpp/tdur.apply(lambda x: np.sqrt(x))
    cdpp_eff = cdpp * (tcdpp/tdur).apply(lambda x: np.sqrt(x))

    factor2 = delta/cdpp_eff
    #factor2_old = delta/cdpp_eff_old
    sn = factor1 * factor2
    #sn_old = factor1_old * factor2_old

    """
    KEEP NANS ACTUALLY FOR FREE INFORMATION ON GEOMETRIC TRANSITS
    if unit_test_flag==True:
        if np.isnan(sn)==True:
            sn = 0
        return sn
    else:
        sn = sn.fillna(0)
        return sn
    """

    return sn

def draw_cdpp(star_radius, df):
    df = df.loc[(df.st_radius<star_radius+0.15)&(df.st_radius>star_radius-0.15)]
    #print("df length: ", len(df))
    cdpp = np.random.choice(df.rrmscdpp06p0)
    return cdpp

def draw_cdpp_array(star_radius, df):
    # calculate CDPP by drawing from Kepler dataset relation with star radius
    cdpp = [draw_cdpp(sr, berger_kepler) for sr in star_radius]
    return cdpp

def kepmag_to_noise_floor(target_mag):
    """
    Based on Natalia Guerrero's tutorial on photometric precision (actually, the noise floor). 
    This isn't quite CDPP, but it's close enough for the purposes of calculating planet detections from TRILEGAL. 

    Input: 
    - target_mag: Kepler filter magnitude of target star (np.array)

    Output:
    - CDPP: [ppm]
    """

    ### reference star
    m1 = np.array([15.962]) # Kepler magnitude of KOI-225; 14.954418 vs 15.962?
    ref_precision = 473 / 1e6 # photometric precision [ppm] of KOI-225, per Table 2, Colon+ 2012 (https://arxiv.org/pdf/1207.2481)
    #m1 = np.array([13.046]) # Kepler magnitude of KOI-526; third row of Table 2, Colon+ 2012
    #ref_precision = 1223/ 1e6 # ditto

    ratio = 100**((m1 - target_mag)/5)  # f2 / f1

    target_precision = (np.sqrt(ratio) / ratio ) * ref_precision * 1e6

    return target_precision

def kepmag_to_cdpp(df, new_kepmag):

    """
    Use Kepler mag vs CDPP (6 hr) relation from Berger Kepler-Gaia crossmatch to infer CDPP given simulated kepmag from TRILEGAL, etc
    Inputs:
    - df: reference DataFrame, with Kepler mag and CDPP
    - new_kepmag: new Series of Kepler magnitudes, for which I want to sample new CDPPs, eg. from TRILEGAL

    Output:
    - xbins[new_kepmag_snaps]: snapped version of new_kepmag
    - new_cdpp_snaps: new array of CDPPs
    """

    new_kepmag = np.array(new_kepmag)
    df = df.dropna(subset=['kepmag', 'rrmscdpp06p0'])

    kepmag = np.array(df['kepmag'])
    cdpp = np.array(df['rrmscdpp06p0'])

    # Create 2D histogram
    xbins = np.linspace(8, 16, 100)
    ybins = np.linspace(0, 1000, 100) 
    hist, yedges, xedges = np.histogram2d(cdpp, kepmag, bins=[ybins, xbins])

    # for each element in new_kepmag, get closest index along Kep mag (x axis)
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    new_kepmag_snaps = []
    new_cdpp_snaps = []
    for i in range(len(new_kepmag)):
        new_kepmag_snap = find_nearest(xedges[:-1], value=new_kepmag[i])
        new_kepmag_snaps.append(new_kepmag_snap)

        # for that column, sample from the 1d histogram 
        one_d_hist = hist[:, new_kepmag_snap]
        new_cdpp_snap = np.random.choice(ybins[:-1], size=1, p=one_d_hist/np.sum(one_d_hist))[0]
        new_cdpp_snaps.append(new_cdpp_snap)

    ### test by visualizing
    #plt.scatter(xbins[new_kepmag_snaps], new_cdpp_snaps, s=10)
    #plt.xlabel('Kepler mag')
    #plt.ylabel('CDPP rms (6 hr) [ppm]')
    #plt.show()

    return xbins[new_kepmag_snaps], new_cdpp_snaps

def kepmag_to_cdpp_vectorized(data1, data2):

    """
    Use Kepler mag vs CDPP (6 hr) relation from Berger Kepler-Gaia crossmatch to infer CDPP given simulated kepmag from TRILEGAL, etc
    Fold in Teff and galactic height, since these bear indirectly on noise properties and directly on experimental dependent variables
    Inputs:
    - data1: reference DataFrame, with Kepler mag and CDPP
    - data2: target DataFrame from which to sample

    Output:
    #- cdpps: new array of CDPPs
    - new_data2: this operation culls records that have NaN heights, etc. So let's use this opportunity to cull the entire DF, rather than output a column of different length from the original
    """

    #cols_to_match = ['mag_bins', 'teff_bins', 'age_bins']
    cols_to_match = ['mag_bins', 'teff_bins', 'logg_bins']
    logg_bins = np.linspace(3.0, 4.6, 5) 
    teff_bins = np.linspace(5300, 7500, 50) # (5300, 7500, 50) or (4000, 6500, 50)
    mag_bins = np.linspace(8, 16, 5)
    age_bins = np.linspace(0, 14, 10) # (0, 14, 10) or (0, 8, 10)
    stellar_radius_bins = np.linspace(1, 3.5, 5)
    cdpp_bins = np.linspace(0, 1000, 10) # (0, 100, 10)
    height_bins = np.logspace(2,3,6)

    # reference DataFrame
    data1['mag_bins'] = pd.cut(data1['kepmag'], bins=mag_bins, include_lowest=True)
    data1['logg_bins'] = pd.cut(data1['iso_logg'], bins=logg_bins, include_lowest=True)
    data1['teff_bins'] = pd.cut(data1['iso_teff'], bins=teff_bins, include_lowest=True)
    #data1['stellar_radius_bins'] = pd.cut(data1['iso_rad'], bins=stellar_radius_bins, include_lowest=True)
    #data1['cdpp_bins'] = pd.cut(data1['rrmscdpp06p0'], bins=cdpp_bins, include_lowest=True)
    #data1['height_bins'] = pd.cut(data1['height'], bins=height_bins, include_lowest=True)
    #data1['age_bins'] = pd.cut(data1['age'], bins=age_bins, include_lowest=True)

    data1_cdpp_mean = data1.groupby(cols_to_match)['rrmscdpp06p0'].mean().reset_index()
    data1_cdpp_mean = data1_cdpp_mean.pivot(index='mag_bins', columns=cols_to_match[1:])

    data1_cdpp_std = data1.groupby(cols_to_match)['rrmscdpp06p0'].std().reset_index()
    data1_cdpp_std = data1_cdpp_std.pivot(index='mag_bins', columns=cols_to_match[1:])

    # unstack pivot tables
    data1_unstacked_mean = data1_cdpp_mean.unstack().reset_index()
    data1_unstacked_mean = data1_unstacked_mean[['mag_bins', 'teff_bins', 'logg_bins', 0]]
    data1_unstacked_mean.columns = ['mag_bins', 'teff_bins', 'logg_bins','cdpp_mean']
    data1_unstacked_mean.dropna(subset=['cdpp_mean'], inplace=True)
    #print(data1_unstacked_mean)

    data1_unstacked_std = data1_cdpp_std.unstack().reset_index()
    data1_unstacked_std = data1_unstacked_std[['mag_bins', 'teff_bins', 'logg_bins', 0]]
    data1_unstacked_std.columns = ['mag_bins', 'teff_bins', 'logg_bins','cdpp_std']
    data1_unstacked_std.dropna(subset=['cdpp_std'], inplace=True)
    #print(data1_unstacked_std)

    # target DataFrame from which to sample
    data2['mag_bins'] = pd.cut(data2['Kepler'], bins=mag_bins, include_lowest=True)
    data2['logg_bins'] = pd.cut(data2['logg'], bins=logg_bins, include_lowest=True)
    data2['teff_bins'] = pd.cut(data2['Teff'], bins=teff_bins, include_lowest=True)
    #data2['stellar_radius_bins'] = pd.cut(data2['stellar_radius'], bins=stellar_radius_bins, include_lowest=True)
    #data2['cdpp_bins'] = pd.cut(data2['cdpp'], bins=cdpp_bins, include_lowest=True)
    #data2['height_bins'] = pd.cut(data2['height'], bins=height_bins, include_lowest=True)
    #data2['age_bins'] = pd.cut(data2['age'], bins=age_bins, include_lowest=True)
    #print(data2)

    # merge the DataFrames on the multiple columns
    new_data2 = pd.merge(data2, data1_unstacked_mean, on=cols_to_match, how='inner')
    new_data2.dropna(subset=['cdpp_mean'], inplace=True)

    new_data2 = pd.merge(new_data2, data1_unstacked_std, on=cols_to_match, how='inner')
    new_data2.dropna(subset=['cdpp_std'], inplace=True)
    #print(new_data2)

    # sample from target
    new_data2['cdpp'] = np.random.normal(new_data2['cdpp_mean'], new_data2['cdpp_std'])

    return new_data2

def kepmag_to_cdpp_3d(df, new_kepmag, new_teff):

    """
    Use Kepler mag vs CDPP (6 hr) relation from Berger Kepler-Gaia crossmatch to infer CDPP given simulated kepmag from TRILEGAL, etc
    But wait! This relation changes across spectral type. So we need to sample cdpp jointly per kepmag and teff. 
    Even this doesn't fully correct things. 
    Inputs:
    - df: reference DataFrame, with Kepler mag, CDPP, and Teff, eg. Berger20
    - new_kepmag: new Series of Kepler magnitudes, for which I want to sample new CDPPs, eg. from TRILEGAL
    - new_teff: Series of Teff from TRILEGAL
    - new_radius? 

    Output:    
    - new_cdpp_snaps: new array of CDPPs
    """

    new_kepmag = np.array(new_kepmag)
    new_teff = np.array(new_teff)
    new_df = pd.DataFrame({'kepmag': new_kepmag, 'teff': new_teff})

    df = df.dropna(subset=['kepmag', 'rrmscdpp06p0', 'iso_teff'])
    ### cull Berger20 reference data to Kepler mag and Teff limits of TRILEGAL
    #df = df.loc[(df['kepmag'] >= 8) & (df['kepmag'] <= 16)]
    #df = df.loc[(df['iso_teff'] >= 3500) & (df['iso_teff'] <= 7500)]
    #plt.hist(df['iso_teff'], bins=np.linspace(3500, 7500, 100))
    #plt.show()
    #quit()

    mag_bins = np.linspace(8, 16, 100)
    cdpp_bins = np.linspace(0, 200, 100) 
    teff_bins = np.linspace(4800, 7500, 100)

    df['mag_bins'] = pd.cut(df['kepmag'], bins=mag_bins, include_lowest=True)
    df['teff_bins'] = pd.cut(df['iso_teff'], bins=teff_bins, include_lowest=True)
    #df['cdpp_bins'] = pd.cut(df['rrmscdpp06p0'], bins=cdpp_bins, include_lowest=True)

    new_df['mag_bins'] = pd.cut(new_df['kepmag'], bins=mag_bins, include_lowest=True)
    new_df['teff_bins'] = pd.cut(new_df['teff'], bins=teff_bins, include_lowest=True)

    def find_nearest_pd(df, new_kepmag_bin, new_teff_bin):
        sub_df = df.loc[(df['mag_bins'] == new_kepmag_bin) & (df['teff_bins'] == new_teff_bin)]
        cdpp_mean = np.nanmean(sub_df['rrmscdpp06p0'])
        cdpp_std = np.nanstd(sub_df['rrmscdpp06p0'])
        cdpp_draw = np.random.normal(cdpp_mean, cdpp_std)
        return cdpp_draw

    cdpps = []
    for i in range(len(new_df)):
        row = new_df.iloc[i]

        cdpp_draw = find_nearest_pd(df, row['mag_bins'], row['teff_bins'])
        cdpps.append(cdpp_draw)

    new_df['cdpp'] = cdpps

    return np.array(new_df['cdpp'])
    

def kepmag_to_cdpp_height(df, new_kepmag, new_teff, new_height):

    """
    Use Kepler mag vs CDPP (6 hr) relation from Berger Kepler-Gaia crossmatch to infer CDPP given simulated kepmag from TRILEGAL, etc
    Matched-sample by radius and height as well, since without it, cdpp and radius per height bin drop off way more in trilegal than b20.

    Inputs:
    - df: reference DataFrame, with Kepler mag, CDPP, and Teff, eg. Berger20
    - new_kepmag: new Series of Kepler magnitudes, for which I want to sample new CDPPs, eg. from TRILEGAL
    - new_teff: Series of Teff from TRILEGAL
    - new_height: Series of calculated heights from TRILEGAL 

    Output:    
    - new_cdpp_snaps: new array of CDPPs
    """

    new_kepmag = np.array(new_kepmag)
    new_teff = np.array(new_teff)
    new_height = np.array(new_height)
    new_df = pd.DataFrame({'kepmag': new_kepmag, 'teff': new_teff, 'height': new_height})

    df = df.dropna(subset=['kepmag', 'rrmscdpp06p0', 'iso_teff', 'height'])
    ### cull Berger20 reference data to Kepler mag and Teff limits of TRILEGAL
    #df = df.loc[(df['kepmag'] >= 8) & (df['kepmag'] <= 16)]
    #df = df.loc[(df['iso_teff'] >= 3500) & (df['iso_teff'] <= 7500)]
    #plt.hist(df['iso_teff'], bins=np.linspace(3500, 7500, 100))
    #plt.show()
    #quit()

    mag_bins = np.linspace(8, 16, 100)
    cdpp_bins = np.linspace(0, 200, 100) 
    teff_bins = np.linspace(4800, 7500, 100)
    height_bins = np.logspace(2, 3, 6)

    df['mag_bins'] = pd.cut(df['kepmag'], bins=mag_bins, include_lowest=True)
    df['teff_bins'] = pd.cut(df['iso_teff'], bins=teff_bins, include_lowest=True)
    df['height_bins'] = pd.cut(df['height'], bins=height_bins, include_lowest=True)
    #df['cdpp_bins'] = pd.cut(df['rrmscdpp06p0'], bins=cdpp_bins, include_lowest=True)

    new_df['mag_bins'] = pd.cut(new_df['kepmag'], bins=mag_bins, include_lowest=True)
    new_df['teff_bins'] = pd.cut(new_df['teff'], bins=teff_bins, include_lowest=True)
    new_df['height_bins'] = pd.cut(df['height'], bins=height_bins, include_lowest=True)

    def find_nearest_pd(df, new_kepmag_bin, new_teff_bin):
        sub_df = df.loc[(df['mag_bins'] == new_kepmag_bin) & (df['teff_bins'] == new_teff_bin)]
        cdpp_mean = np.nanmean(sub_df['rrmscdpp06p0'])
        cdpp_std = np.nanstd(sub_df['rrmscdpp06p0'])
        cdpp_draw = np.random.normal(cdpp_mean, cdpp_std)
        return cdpp_draw

    cdpps = []
    for i in range(len(new_df)):
        row = new_df.iloc[i]

        cdpp_draw = find_nearest_pd(df, row['mag_bins'], row['teff_bins'])
        cdpps.append(cdpp_draw)

    new_df['cdpp'] = cdpps

    return np.array(new_df['cdpp'])

def make_pdf_rows(x, mode, err1, err2):
    """
    Row-wise likelihood of asymmetric uncertainty, using Eqn 6 from https://iopscience.iop.org/article/10.3847/1538-3881/abd93f
    Less efficient, but row-wise in order to troubleshoot. 
    Takes in scalar values for mode, err1, and err2, not arrays.

    Input:
    - x: np.linspace(0.5, 10, 100); just something to undergird the PDF and represent stellar ages
    - mode: mean age; peak of asymmetric PDF
    - err1: + uncertainty
    - err2: - uncertainty (note: must be positive)

    Output:
    - pdf: asymmetric PDF

    """
    
    factor1 = 1/np.sqrt(2*np.pi*err1**2) 
    beta = err1/err2
    gamma = (err1 - np.abs(err2))/(err1 * np.abs(err2))
    
    factor2_arg_a = np.log(1+gamma*(x - mode))
    factor2_arg_b = np.log(beta)
    factor2_arg = factor2_arg_a/factor2_arg_b
    factor2 = np.exp(-0.5*(factor2_arg)**2)

    if np.isnan(factor1*factor2).all():
        print(mode, err1, err2)

    out = factor1 * factor2
    out[~np.isfinite(out)] = 0.0
    
    return out

def draw_asymmetrically(df, mode_name, err1_name, err2_name, drawn):
    """
    Draw stellar properties with asymmetric errors. 
    This is the generalized version of draw_star_ages(), below
    
    Inputs:
    - df: berger_kepler [Pandas DataFrame]
    - mode_name: name of mode column [string]
    - err1_name: name of err1 column [string]
    - err2_name: name of err2 column [string]
    - drawn: name of new column [string]

    Output:
    - df: berger_kepler_df, now with new column with drawn parameter, "drawn" [Pandas DataFrame]
    """

    # in case df is broken up by planet and not star
    uniques = df.drop_duplicates(subset=['kepid'])
    
    if drawn=='age':
        x = np.linspace(0.5, 13.5, 100)
    elif drawn=='gyro_age':
        x = np.linspace(0.5, 4., 100)
    elif drawn=='stellar_radius':
        x = np.linspace(0.5, 5., 100)
    elif drawn=='stellar_mass':
        x = np.linspace(0.5, 2.5, 100)
    elif drawn=='Teff':
        x = np.linspace(3700, 7500, 1000)
    elif drawn=='stellar_feh':
        x = np.linspace(-0.5, 0.5, 100)
    elif drawn=='distance':
        x = np.linspace(0, 5000, 1000)
    elif drawn=='planet_radius':
        x = np.linspace(0.5, 10, 100)

    else: 
        print("Please create a column that is either age, gyro_age, stellar_radius, stellar_mass, distance, planet_radius, stellar_feh, or stellar_teff!")

    """
    TESTING
    print("x: ", x)
    pdf = make_pdf_rows(x, mode, err1, err2)
    pdf = pdf/np.sum(pdf)
    draw = np.around(np.random.choice(x, p=pdf), 2)
    print("draw: ", draw)
    """

    if drawn=='planet_radius':
        df_or_uniques = df
    else:
        df_or_uniques = uniques

    draws = np.ones(len(df_or_uniques))
    for i in range(len(df_or_uniques)):
        mode = df_or_uniques.iloc[i][mode_name]
        err1 = df_or_uniques.iloc[i][err1_name]
        err2 = np.abs(df_or_uniques.iloc[i][err2_name])
        
        # symmetric uncertainties
        if err1==err2:
            draw = 0
            while draw <= 0: # make sure the draw is positive
                draw = np.around(np.random.normal(mode, err1), 2)

        # asymmetric uncertainties
        elif err1!=err2:
            pdf = make_pdf_rows(x, mode, err1, err2)
            pdf = pdf/np.sum(pdf)

            try:
                draw = 0
                while draw <= 0: # make sure the draw is positive
                    draw = np.around(np.random.choice(x, p=pdf), 2)
            except Exception as e:
                print("EXCEPTION: ", i, pdf, mode, err1, err2)
                print(e)
                break
        #print(mode, err1, err2, draw)
        draws[i] = draw

    #print(len(df), len(draws), len(uniques))
    df_or_uniques[drawn] = draws

    # break back out into planet rows and forward fill across systems
    df = uniques.merge(df, how='right')
    df[drawn] = df[drawn].fillna(method='ffill')
    
    return df

def draw_star_ages(df):
    """
    Draw star's age, taking into account asymmetric age errors. Enriches input DataFrame.
    """

    # in case df is broken up by planet and not star
    uniques = df.drop_duplicates(subset=['kepid'])

    x = np.linspace(0.5, 13.5, 100)
    ages = np.ones(len(uniques))
    for i in range(len(uniques)):
        mode = uniques.iloc[i].iso_age
        err1 = uniques.iloc[i].iso_age_err1
        err2 = np.abs(uniques.iloc[i].iso_age_err2)

        # symmetric age uncertainties
        if err1==err2:
            age = 0
            while age <= 0: # make sure the age is positive
                age = np.around(np.random.normal(mode, err1), 2)

        # asymmetric uncertainties
        elif err1!=err2:
            pdf = make_pdf_rows(x, mode, err1, err2)
            pdf = pdf/np.sum(pdf)

            try:
                age = 0
                while age <= 0: # make sure the age is positive
                    age = np.around(np.random.choice(x, p=pdf), 2)
            except:
                print(i, pdf, mode, err1, err2)
                break
        
        ages[i] = age

    df['age'] = ages

    # break back out into planet rows and forward fill across systems
    df = uniques.merge(df, how='right')
    df['age'] = df.age.fillna(method='ffill')

    return df 

def draw_star(df):
    """
    Draw star's age, metallicity, and effective temperature based on given errors. Enrich input DataFrame.
    """

    # in case df is broken up by planet and not star
    uniques = df.drop_duplicates(subset=['kepid'])

    # draw age
    uniques['iso_age_err'] = 0.5 * (uniques.iso_age_err1 + np.abs(uniques.iso_age_err2)) # old way

    uniques['age'] =  np.random.normal(uniques.iso_age, uniques.iso_age_err)

    # draw metallicity...if I do feh instead of iso_feh, do I get a lot of NaNs??
    uniques['feh_err'] = 0.5 * (uniques.feh_err1 + np.abs(uniques.feh_err2))
    uniques['feh'] = np.random.normal(uniques.feh_x, uniques.feh_err)

    # draw Teff
    uniques['iso_teff_err'] = 0.5 * (uniques.iso_teff_err1 + np.abs(uniques.iso_teff_err2))
    uniques['teff'] = np.random.normal(uniques.iso_teff, uniques.iso_teff_err)

    # break back out into planet rows and forward fill across systems
    df = uniques.merge(df, how='right')
    df['age'] = df.age.fillna(method='ffill')
    df['feh'] = df.feh.fillna(method='ffill')

    return df

def galactic_occurrence_step(ages, threshold, frac1, frac2):
        """
        Calculate the probability of system having planets, based on its age and three free parameters
        
        Input:
        - ages: stellar ages, in Gyr [float]
        - threshold: age beyond which probability of hosting a planet is frac2, versus frac1, in Gyr [float]
        - frac1: planet host fraction among systems younger than threshold [float]
        - frac2: planet host fraction among systems older than threshold [float]

        Output:
        - host_frac: jnp.array of fraction of planet hosts [float]

        """

        host_frac = jnp.where(ages <= threshold, frac1, frac2)

        return host_frac

def draw_planet_periods_and_radii(num_planets, alpha_se, alpha_sn, m_star):

    """
    It says it on the tin. 
    This function draws period and radius "together" instead of separately bc masses (by way of radii) are required to calculate Hill radii when drawing periods.
    So, periods are drawn from a loguniform distribution, then radii and masses are drawn, which enables a check on Hill radii. 
    If a system fails the check, we sample everything again.

    Inputs:
    - num_planets: system multiplicity
    - alpha_se: power law exponent for drawing Super-Earth radii, from Zink+ 2023
    - alpha_sn: power law exponent for drawing Sub-Neptune radii, from Zink+ 2023
    - m_star: stellar host mass, in Solar masses

    Outputs:
    - periods: list of planet periods, in days
    - planet_radii: list of planet radii, in Earth radii
    - planet_masses: list of planet masses, in Earth masses
    """

    hill_radius_check = True

    # initially assign planet planet periods from loguniform distribution from 2 to 300 days
    periods = jnp.sort(jnp.array(loguniform.rvs(2, 300, size=num_planets)))

    # draw planet radii
    planet_radii = draw_planet_radii(periods, alpha_se, alpha_sn)

    # draw planet masses using Forecaster (Chen & Kipping 2016: https://iopscience.iop.org/article/10.3847/1538-4357/834/1/17)
    # code from Ben Cassese: https://github.com/ben-cassese/astro-forecaster 
    planet_masses = forecaster.Rpost2M(planet_radii,
                                unit='Earth',
                                classify=False)

    # convert periods to semi-major axes
    a = p_to_a(periods, m_star)

    # check if stable by inspecting mutual Hill radii between all adjacent planet pairs
    if num_planets > 1:
        for i in range(num_planets-1):
            j = i+1 # j is outer; i is inner

            # calculate mutual Hill radius
            r_hm = 0.5*(a[i]+a[j]) * ((earth_mass_to_cgs(planet_masses[i])+earth_mass_to_cgs(planet_masses[j]))/(3*solar_mass_to_cgs(m_star)))**(1./3)

            k = (a[j]-a[i])/r_hm
            if k <= 12: # https://academic.oup.com/mnras/article/527/1/79/7285825#429734139 and references therein; should it be lower? 
                check = False

    if hill_radius_check == False:
        periods = draw_planet_periods_and_radii(num_planets, alpha_se, alpha_sn, m_star) # risky recursion of the day

    return periods, planet_radii, planet_masses

def draw_planet_periods(m, m_star):
    """
    Draw periods for all planets in a planetary system

    Input: 
    - m: list of planets' masses [Earth masses]
    - m_star: mass of stellar host [Solar masses]

    Output:
    - periods: list of planets' periods [days]
    """

    periods = jnp.array(loguniform.rvs(2, 300, size=self.num_planets))

    semi_major_axes = p_to_a(periods, m_star)

    check = True # all is well
    if len(semi_major_axes) > 1:
        for i in range(len(semi_major_axes)-1):
            j = i+1 # j is outer; i is inner

            r_hm = 0.5*(a[i]+a[j]) * ((m[i]+m[j])/(3*m_star))**(1./3)

            k = (a[j]-a[i])/r_hm

            if k <= 12: # https://academic.oup.com/mnras/article/527/1/79/7285825#429734139 and references therein; should it be lower? 
                check = False

    if check == False:
        periods = draw_planet_periods(m, m_star) # risky recursion of the day

    return periods

def draw_planet_radii(periods, alpha_se, alpha_sn):
    """
    Draw planet radii following Zink+ 2023: https://iopscience.iop.org/article/10.3847/1538-3881/acd24c#ajacd24cs4. 
    This starts with a simple power law.
    Then we partition using the Van Eylen+ 2018 planet radius-period relation: https://academic.oup.com/mnras/article/479/4/4786/5050069
    We do not yet offer size ordering within systems. 

    Input: 
    - periods: planet periods [np.array of floats]
    - alpha_se: power relation between radius and the PDF of the radius distribution, for Super-Earths [float]
    - alpha_sn: power relation between radius and the PDF of the radius distribution, for Sub-Neptunes [float]

    Output: 
    - radii: planet radii [np.array of floats]
    """

    # mise en place
    # are these going to be Super-Earths (se) or Sub-Neptunes (sn)? Zink+ 2023 says their occurrence rates overlap, so in a world with only small planets, let's say 50-50
    # should I assume Peas in a Pod? Should I allow for that? 
    se_or_sn = np.random.choice(['se','sn'], size=len(periods), p=[0.5, 0.5])
    se_grid = np.linspace(1.2, 2., 100)
    sn_grid = np.linspace(2., 4., 100)
    period_grid = np.logspace(np.log10(3), np.log10(200), 100)
    m_grid = np.linspace(-0.5, 0.5, 100) # for radius gap
    a_grid = np.linspace(np.log10(1), np.log10(4), 100) # for radius gap

    # draw alpha, which is the power relation between radius and the PDF of the radius distribution

    # generate PDF for radius, and then normalize to sum to 1
    q_se = se_grid**alpha_se
    q_se = q_se/np.sum(q_se)

    q_sn = sn_grid**alpha_sn
    q_sn = q_sn/np.sum(q_sn)
    
    # draw initial radii
    radii = [] # this will be flexible if I want to break Peas in a Pod later
    radii_se = np.random.choice(se_grid, size=len(se_or_sn[se_or_sn == 'se']), p=q_se)
    radii_sn = np.random.choice(sn_grid, size=len(se_or_sn[se_or_sn == 'sn']), p=q_sn)
    radii.append(radii_se)
    radii.append(radii_sn)
    radii = np.concatenate([x.ravel() for x in radii])

    """
    Cull radius valley planets. Maybe 5%? 
    """
    # draw m, which is the power law slope between radius and period
    radius_valley_m_pdf = make_pdf_rows(m_grid, -0.09, 0.02, 0.04)
    radius_valley_m_pdf = radius_valley_m_pdf/np.sum(radius_valley_m_pdf)
    m = np.around(np.random.choice(m_grid, p=radius_valley_m_pdf), 2)

    # draw upper and lower envelope y-intercepts 
    radius_valley_a_upper_pdf = make_pdf_rows(a_grid, 0.44, 0.04, 0.03)
    radius_valley_a_upper_pdf = radius_valley_a_upper_pdf/np.sum(radius_valley_a_upper_pdf)
    a_upper = np.around(np.random.choice(a_grid, p=radius_valley_a_upper_pdf), 2)

    radius_valley_a_lower_pdf = make_pdf_rows(a_grid, 0.29, 0.04, 0.03)
    radius_valley_a_lower_pdf = radius_valley_a_lower_pdf/np.sum(radius_valley_a_lower_pdf)
    a_lower = np.around(np.random.choice(a_grid, p=radius_valley_a_lower_pdf), 2)

    # is it in the radius valley?
    upper = 10**(m * np.log10(periods) + a_upper)
    lower = 10**(m * np.log10(periods) + a_lower)

    # resample 95% of the time if a planet lands in here
    for index, radius in enumerate(radii):

        # only for gap planets
        while (radius >= lower[index]) & (radius <= upper[index]):

            # 5% chance of grace 
            grace = np.random.choice(['accept gap', 'reject gap'], p=[0.05, 0.95])
            if grace=='accept gap':
                break
            elif grace=='reject gap':
                pass

            # resample until planet is out of gap
            if radius <= 2.:
                radii[index] = np.random.choice(se_grid, size=1, p=q_se)[0]
            elif (radius > 2.) & (radius <= 4.):
                radii[index] = np.random.choice(sn_grid, size=1, p=q_sn)[0]
            #print("try: ", radii[index], lower[index], upper[index])

    """
    # plot to check if radius sampling and valley are working as expected; feed in: stats.loguniform.rvs(2, 300, size=1000)
    df = pd.DataFrame({'p': periods, 'r': radii, 'upper': upper, 'lower': lower})
    df = df.sort_values(by='p')
    plt.scatter(df.p, df.r, s=10)
    plt.xscale('log')
    plt.plot(df.p, df.upper, label='upper', color='k')
    plt.plot(df.p, df.lower, label='lower', color='r')
    plt.xlabel('period [days]')
    plt.ylabel('radius [$R_{\oplus}$]')
    plt.legend(bbox_to_anchor=(1., 1.05))
    plt.tight_layout()
    plt.savefig(path+'radius-v-period.png')
    plt.show()
    """

    return radii

def collect_galactic(df):
    """
    Compute geometric and detected transit multiplicities, as well as other population-wide statistics, like fraction of planet hosts and fraction of intact systems
    
    This is different from collect() in the dynamical sculpting case because the models there affected intact_frac, not frac_host. 
    So transit multiplicity doesn't assume a constant f. In fact, it already contains it. 

    Also, the scope of this function is per Star, not per Population. 

    Input:
    - df: Pandas DataFrame of planet-hosting systems, with rows broken down by planet, not star

    Outputs:
    - transit_multiplicity_among_planet_hosts: 
    """

    # isolate transiting planets
    transiters_berger_kepler = df.loc[df['transit_status']==1]

    # compute transit multiplicity 
    transit_multiplicity_among_planet_hosts = transiters_berger_kepler.groupby('kepid').count()['transit_status'].reset_index().groupby('transit_status').count().reset_index().kepid
    transit_multiplicity_among_planet_hosts = transit_multiplicity_among_planet_hosts.to_list()
    transit_multiplicity_among_planet_hosts += [0.] * (6 - len(transit_multiplicity_among_planet_hosts)) # pad with zeros to match length of k

    # also calculate the geometric transit multiplicity
    geom_transiters_berger_kepler = df.loc[df['geom_transit_status']==1]
    geom_transit_multiplicity = geom_transiters_berger_kepler.groupby('kepid').count()['geom_transit_status'].reset_index().groupby('geom_transit_status').count().reset_index().kepid
    geom_transit_multiplicity = geom_transit_multiplicity.to_list()
    geom_transit_multiplicity += [0.] * (6 - len(geom_transit_multiplicity)) # pad with zeros to match length of k

    # calculate logLs 
    logL = better_loglike(transit_multiplicity, k)
    logL_score = better_loglike(transit_multiplicity, k_score)
    logL_fpp = better_loglike(transit_multiplicity, k_fpp)

    # get intact and disrupted fractions (combine them later to get fraction of systems w/o planets)
    intact = df.loc[df.intact_flag=='intact']
    disrupted = df.loc[df.intact_flag=='disrupted']
    intact_frac = len(intact.kepid.unique())/len(df.kepid.unique())
    disrupted_frac = len(disrupted.kepid.unique())/len(df.kepid.unique())

    return transit_multiplicity_among_planet_hosts, geom_transit_multiplicity_among_planet_hosts, intact_fracs, disrupted_fracs, logLs, logLs_score, logLs_fpp

def normalize(array):
    array = 10**array # exponentiate first!!
    sum = np.sum(array)
    
    return array/sum

def draw_galactic_heights(df):

    # in case df is broken up by planet and not star
    uniques = df.drop_duplicates(subset=['kepid'])

    # Read in stellar density vs galactic height curves from Ma+ 2017 Fig 2: https://academic.oup.com/mnras/article/467/2/2430/2966031. I converted the image to data using PlotDigitizer.
    data1 = pd.read_csv(path+'galactic-occurrence/data/Ma17-fig2-0-2Gyr.csv', header=None, 
                    names=['height','density'])
    data2 = pd.read_csv(path+'galactic-occurrence/data/Ma17-fig2-2-4Gyr.csv', header=None, # Ma_midplaneheight_2Gyr_4Gyr.txt
                    names=['height','density'])
    data3 = pd.read_csv(path+'galactic-occurrence/data/Ma17-fig2-4-6Gyr.csv', header=None, 
                    names=['height','density'])
    data4 = pd.read_csv(path+'galactic-occurrence/data/Ma17-fig2-6-8Gyr.csv', header=None, 
                    names=['height','density'])
    data5 = pd.read_csv(path+'galactic-occurrence/data/Ma17-fig2-8-moreGyr.csv', header=None, 
                    names=['height','density'])

    data1_height = np.linspace(0, np.array(data1['height'])[-1], 100)
    data2_height = np.linspace(0, np.array(data2['height'])[-1], 100)
    data3_height = np.linspace(0, np.array(data3['height'])[-1], 100)
    data4_height = np.linspace(0, np.array(data4['height'])[-1], 100)
    data5_height = np.linspace(0, np.array(data5['height'])[-1], 100)

    # interpolate for finer galactic height draws
    cs1 = CubicSpline(data1['height'], data1['density'])
    cs2 = CubicSpline(data2['height'], data2['density'])
    cs3 = CubicSpline(data3['height'], data3['density'])
    cs4 = CubicSpline(data4['height'], data4['density'])
    cs5 = CubicSpline(data5['height'], data5['density'])

    data1_density = cs1(data1_height)
    data2_density = cs2(data2_height)
    data3_density = cs3(data3_height)
    data4_density = cs4(data4_height)
    data5_density = cs5(data5_height)

    data1_density = normalize(data1_density) 
    data2_density = normalize(data2_density) 
    data3_density = normalize(data3_density) 
    data4_density = normalize(data4_density) 
    data5_density = normalize(data5_density) 

    # draw heights based on stellar ages
    heights = []
    for i in range(len(df)):
        if uniques['age'][i] <= 2.:
            height = np.random.choice(data1_height, p=data1_density)
        elif (uniques['age'][i] > 2.) & (uniques['age'][i] <= 4.):
            height = np.random.choice(data2_height, p=data2_density)
        elif (uniques['age'][i] > 4.) & (uniques['age'][i] <= 6.):
            height = np.random.choice(data3_height, p=data3_density)
        elif (uniques['age'][i] > 6.) & (uniques['age'][i] <= 8.):
            height = np.random.choice(data4_height, p=data4_density)
        elif uniques['age'][i] > 8.:
            height = np.random.choice(data5_height, p=data5_density)
        heights.append(height)

    df['height'] = np.array(heights) * 1000

     # break back out into planet rows and forward fill across systems
    df = uniques.merge(df, how='right')
    df['height'] = df.height.fillna(method='ffill')

    return df 

def gala_galactic_heights(df, output=True):

    """
    Use Gala (Price-Whelan+) to simulate orbits of Gaia stars and get their Z_maxes
    """

    """
    # merge sample with Megan Bedell's Gaia-Kepler cross-match because those save info on RV, proper motion, parallax, etc required for Gala
    berger = Table.read(path+'data/berger_kepler_stellar_fgk.csv')
    megan = Table.read(path+'data/kepler_dr3_good.fits')
    merged = join(berger, megan, keys='kepid')
    merged.rename_column('parallax_2', 'parallax')
    #print(merged[['parallax', 'parallax_error', 'radial_velocity', 'radial_velocity_error']])
    """
    df['radial_velocity'] = np.random.normal(df['radial_velocity'], df['radial_velocity_error'])
    df['pmra'] = np.random.normal(df['pmra'], df['pmra_error'])
    df['pmdec'] = np.random.normal(df['pmdec'], df['pmdec_error'])
    df['parallax'] = np.random.normal(df['parallax'], df['parallax_error'])

    # mise en place
    with coord.galactocentric_frame_defaults.set("v4.0"):
        galcen_frame = coord.Galactocentric()

    sun_xyz = u.Quantity(
        [-galcen_frame.galcen_distance, 0 * u.kpc, galcen_frame.z_sun]  # x  # y  # z
    )

    sun_w0 = gd.PhaseSpacePosition(pos=sun_xyz, vel=galcen_frame.galcen_v_sun)

    mw_potential = gp.MilkyWayPotential()

    sun_orbit = mw_potential.integrate_orbit(sun_w0, dt=0.5 * u.Myr, t1=0, t2=4 * u.Gyr)

    star_gaia = GaiaData(df)

    star_gaia_c = star_gaia.get_skycoord()
    star_galcen = star_gaia_c.transform_to(galcen_frame)
    star_w0 = gd.PhaseSpacePosition(star_galcen.data)

    # calculate orbits and retrieve Z_maxes
    zmaxes = []
    for i in tqdm(range(len(star_gaia))):
    #for i in range(1000):
        star_orbit = mw_potential.integrate_orbit(star_w0[i], t=sun_orbit.t) 
        zmax = star_orbit.zmax().value
        zmaxes.append(zmax)

    zmaxes_df = pd.DataFrame({'height': zmaxes})
    #zmaxes_df.to_csv(path+'data/zmaxes.csv', index=False)
    
    return zmaxes

def completeness(physical, detected):
    """
    Calculate completeness (geometric transit + sensitivity = detected) across period and radius bins

    Inputs:
    - physical: all simulated 
    - detected: 

    Output:
    - piv: Pandas pivot table of completeness fractions across period and radius bins
    """

    geom = physical.loc[physical['geom_transit_status']==1]
    #print(geom.loc[((geom['planet_radii'] > 3.667) & (geom['planet_radii'] <= 4) & (geom['periods'] > 3.49) & (geom['periods'] <= 6.09))])

    period_grid = np.logspace(np.log10(2), np.log10(40), 10)
    radius_grid = np.linspace(1, 4, 10)

    # physical occurrence map
    df_physical = physical[['periods', 'planet_radii']]
    df_physical['radius_bins'] = pd.cut(df_physical['planet_radii'], bins=radius_grid, include_lowest=True)
    df_physical['period_bins'] = pd.cut(df_physical['periods'], bins=period_grid, include_lowest=True)
    #print(physical.loc[((physical['planet_radii'] > 3.667) & (physical['planet_radii'] <= 4) & (physical['periods'] > 3.49) & (physical['periods'] <= 6.09))])
    
    piv_physical = df_physical.groupby(['period_bins', 'radius_bins']).count().reset_index()
    piv_physical = piv_physical.pivot(index='radius_bins', columns='period_bins', values='periods')

    # detected occurrence map
    df_detected = detected[['periods', 'planet_radii']]
    df_detected['radius_bins'] = pd.cut(df_detected['planet_radii'], bins=radius_grid, include_lowest=True)
    df_detected['period_bins'] = pd.cut(df_detected['periods'], bins=period_grid, include_lowest=True)
    #print(detected.loc[((detected['planet_radii'] > 3.667) & (detected['planet_radii'] <= 4) & (detected['periods'] > 3.49) & (detected['periods'] <= 6.09))])

    piv_detected = df_detected.groupby(['period_bins', 'radius_bins']).count().reset_index()
    piv_detected = piv_detected.pivot(index='radius_bins', columns='period_bins', values='periods')
    piv = piv_detected/piv_physical

    return piv, piv_physical, piv_detected

def completeness_height(detected1, physical):
    """
    Take period-radius completeness map-adjusted yield, which did not factor height-related stellar parameters (Kmag --> CDPP)
    Args:
        detected1 (np.array): period-radius completeness map-adjusted yield
        physical (np.array): actual, physical synthetic yield

    Returns:
        detected2 (Pandas DataFrame): height-bin-adjusted yield
    """
    #print(detected1)
    #print(physical)
    efficiency = detected1 / physical
    #print(efficiency)
    detected2 = detected1 / efficiency
    #print(detected2)
    #quit()
    return detected2

def completeness_3d(physical, detected):
    """
    Calculate completeness (geometric transit + sensitivity = detected) across period and radius bins

    Inputs:
    - physical: all simulated 
    - detected: 

    Output:
    - piv: Pandas pivot table of completeness fractions across period and radius bins
    """

    geom = physical.loc[physical['geom_transit_status']==1]
    #print(geom.loc[((geom['planet_radii'] > 3.667) & (geom['planet_radii'] <= 4) & (geom['periods'] > 3.49) & (geom['periods'] <= 6.09))])

    period_grid = np.logspace(np.log10(2), np.log10(40), 5)
    radius_grid = np.linspace(1, 4, 5)
    height_grid = np.logspace(2, 3, 6)

    # physical occurrence map
    df_physical = physical[['periods', 'planet_radii', 'height']]
    df_physical['radius_bins'] = pd.cut(df_physical['planet_radii'], bins=radius_grid, include_lowest=True)
    df_physical['period_bins'] = pd.cut(df_physical['periods'], bins=period_grid, include_lowest=True)
    df_physical['height_bins'] = pd.cut(df_physical['height'], bins=height_grid, include_lowest=True)
    #print(physical.loc[((physical['planet_radii'] > 3.667) & (physical['planet_radii'] <= 4) & (physical['periods'] > 3.49) & (physical['periods'] <= 6.09))])
    
    piv_physical = df_physical.groupby(['period_bins', 'radius_bins', 'height_bins']).count().reset_index()

    piv_physical = piv_physical.pivot(index='radius_bins', columns=['period_bins', 'height_bins'], values=['periods'])
    #print(piv_physical)
    
    # detected occurrence map
    df_detected = detected[['periods', 'planet_radii', 'height']]
    df_detected['radius_bins'] = pd.cut(df_detected['planet_radii'], bins=radius_grid, include_lowest=True)
    df_detected['period_bins'] = pd.cut(df_detected['periods'], bins=period_grid, include_lowest=True)
    df_detected['height_bins'] = pd.cut(df_detected['height'], bins=height_grid, include_lowest=True)
    #print(detected.loc[((detected['planet_radii'] > 3.667) & (detected['planet_radii'] <= 4) & (detected['periods'] > 3.49) & (detected['periods'] <= 6.09))])

    piv_detected = df_detected.groupby(['period_bins', 'radius_bins', 'height_bins']).count().reset_index()
    piv_detected = piv_detected.pivot(index='radius_bins', columns=['period_bins', 'height_bins'], values=['periods'])
    piv = piv_detected/piv_physical

    return piv, piv_physical, piv_detected

def adjust_for_completeness(df, completeness_map, radius_grid, period_grid, flag="detected"):

    """
    For a given DataFrame of planets, group by radius and period bins, then take completeness map and adjust transit_status counts per cell

    Inputs:
    - df: DataFrame of planets
    - completeness map: completeness map of detected vs generated planets, grouped by radius and period bins
    - radius_grid: list of radius bins
    - period_grid: list of period bins
    - flag: DataFrame of either raw or detected planets, labeled "physical" or "detected"

    Output:
    - adjusted_count: completeness-adjusted count of planets
    - df_piv: radius-and-period divided occurrence
    """

    period_grid = np.logspace(np.log10(2), np.log10(40), 10)
    radius_grid = np.linspace(1, 4, 10)
    
    # For detected planets, use completeness map to get back an inferred physical occurrence
    df['radius_bins'] = pd.cut(df['planet_radii'], bins=radius_grid, include_lowest=True)
    df['period_bins'] = pd.cut(df['periods'], bins=period_grid, include_lowest=True)

    """
    berger_kepler_transiters = pd.merge(berger_kepler_transiters, completeness_map.stack().reset_index(), on=['radius_bins', 'period_bins'])
    berger_kepler_transiters.columns = berger_kepler_transiters.columns.astype(str)
    berger_kepler_transiters = berger_kepler_transiters.rename(columns={"0": "completeness"})
    #berger_kepler_planets_temp['adjusted_transit'] = berger_kepler_planets_temp['']
    print(berger_kepler_transiters)
    """

    df_small = df[['radius_bins', 'period_bins', 'transit_status']]
    df_small = pd.merge(df_small, completeness_map.stack(dropna=False).reset_index(), on=['radius_bins', 'period_bins'])
    df_small.columns = df_small.columns.astype(str)

    if flag=="detected":
        df_small = df_small.rename(columns={"0": "completeness"})
        df_small = df_small.groupby(['radius_bins','period_bins']).sum(['transit_status']).reset_index()
        df_piv = df_small.pivot(index='radius_bins', columns='period_bins', values='transit_status')

        # actually adjust transit_status counts
        ###df_small['adjusted_transit_status'] = df_small['transit_status']/df_small['completeness']
        df_piv = df_piv/completeness_map        
        
        ###adjusted_count = int(np.round(np.nansum(df_small['adjusted_transit_status'])))
        adjusted_count = np.nansum(df_piv)

    elif flag=="physical":
        df_small = df_small.rename(columns={"0": "completeness"})
        df_small['placeholder'] = np.where(df_small["completeness"].notna(), 1, np.nan) # count only cells with valid completeness
        df_small = df_small.groupby(['radius_bins','period_bins']).sum(['placeholder']).reset_index()
        df_piv = df_small.pivot(index='radius_bins', columns='period_bins', values='placeholder')

        adjusted_count = np.nansum(df_piv)

    else:
        print("flag must be either 'physical' or 'detected'")

    return adjusted_count, df_piv

def adjust_for_completeness2(df, completeness_map, radius_grid, period_grid):
    
    """
    For a given DataFrame of planets, group by radius and period bins, then take completeness map and adjust transit_status counts per cell

    Inputs:
    - df: DataFrame of planets
    - completeness map: completeness map of detected vs generated planets, grouped by radius and period bins
    - radius_grid: list of radius bins
    - period_grid: list of period bins
    - flag: DataFrame of either raw or detected planets, labeled "physical" or "detected"

    Output:
    - adjusted_count: completeness-adjusted count of planets
    - piv: radius-and-period divided occurrence

    """
    
    period_grid = np.logspace(np.log10(2), np.log10(40), 10)
    radius_grid = np.linspace(1, 4, 10)
    # new, experimental way of applying completeness map
    df['radius_bins'] = pd.cut(df['planet_radii'], bins=radius_grid, include_lowest=True)
    df['period_bins'] = pd.cut(df['periods'], bins=period_grid, include_lowest=True)
    df_small = df[['radius_bins', 'period_bins', 'transit_status']]
    
    # make sure both maps are oriented the same way
    df_small = df_small.groupby(['radius_bins','period_bins']).sum(['transit_status']).reset_index()
    df_piv = df_small.pivot(index='radius_bins', columns='period_bins', values='transit_status')
    piv = df_piv.to_numpy()
    piv = np.flip(piv, axis=0)/completeness_map
    adjusted_count = np.nansum(piv)

    return adjusted_count, piv

def adjust_for_completeness_3d(df, completeness_map, radius_grid, period_grid):
    
    """
    For a given DataFrame of planets, group by radius and period bins, then take completeness map and adjust transit_status counts per cell

    Inputs:
    - df: DataFrame of planets
    - completeness map: completeness map of detected vs generated planets, grouped by radius and period bins
    - radius_grid: list of radius bins
    - period_grid: list of period bins
    - flag: DataFrame of either raw or detected planets, labeled "physical" or "detected"

    Output:
    - adjusted_count: completeness-adjusted count of planets
    - piv: radius-and-period divided occurrence

    """

    period_grid = np.logspace(np.log10(2), np.log10(40), 5)
    radius_grid = np.linspace(1, 4, 5)
    height_grid = np.logspace(2, 3, 6)

    # new, experimental way of applying completeness map
    df['radius_bins'] = pd.cut(df['planet_radii'], bins=radius_grid, include_lowest=True)
    df['period_bins'] = pd.cut(df['periods'], bins=period_grid, include_lowest=True)
    df['height_bins'] = pd.cut(df['height'], bins=height_grid, include_lowest=True)
    df_small = df[['radius_bins', 'period_bins', 'height_bins', 'transit_status']]
    
    # make sure both maps are oriented the same way
    df_small = df_small.groupby(['radius_bins','period_bins','height_bins']).sum(['transit_status']).reset_index()
    df_piv = df_small.pivot(index='radius_bins', columns=['period_bins', 'height_bins'], values='transit_status')
    piv = df_piv.to_numpy()
    #print(completeness_map)
    #print(piv)
    #print(piv.shape)
    piv = np.flip(piv, axis=0)/completeness_map
    adjusted_count = np.nansum(piv)

    return adjusted_count, piv

def collect_age_histograms(df, f=1.):

    """
    Collect age spread across different transit multiplicity bins, given a model. 
    Transit multiplicities are capped at 3+

    Inputs: 
    - df: read-in DataFrames of the simulated planetary system products of injection_recovery_main.py
    - f: fraction of planet-hosting stars (default is 1)

    Outputs:
    - total: Pandas DataFrame with yield counts for each planet row

    """

    # if f < 1, randomly steal some non-nontransits for the nontransits gang
    samples_indices = df.sample(frac=1-f, replace=True, random_state=42).index 
    df.loc[samples_indices, 'transit_status'] = 0
    
    # isolate transiting systems
    transit = df.loc[df['transit_status']==1]

    # create column counting number of planets per system
    transit['yield'] = 1
    #transit = transit.groupby(['kepid', 'iso_age'])[['yield']].count().reset_index()
    transit['yield'] = transit.groupby(["kepid", "age"])["yield"].transform("count")
    
    # isolate non-transiting systems
    #nontransit = df.loc[df['transit_status']==0] # that double counts systems with planets that transit/don't 
    nontransit = df.loc[~df.kepid.isin(transit.kepid.unique())]
    nontransit['yield'] = 0
    
    # re-combine, with yields
    #nontransit['yield'] = 0
    total = pd.concat([nontransit, transit])
    #print(len(df.kepid.unique()), len(total.kepid.unique()))

    #print(transit.groupby('kepid').count().reset_index().groupby('iso_teff').count())
    
    return total, transit, nontransit

def zink_model(kappa, z_max, radius_type='se'):

    """
    Equation 4 from Zink+ 2023, describing occurrence rate over metalllicity, stellar type, galactic height
    
    Inputs:
    - kappa: re-normalization factor 
    - z_max: maximum galactic oscillation amplitude, which is more accurate than simply position in the sky for assessing thin/thick disk membership.
    We are using this interchangeably with scale height [pc]
    - radius_type: boolean flag noting whether we are modeling Super-Earths (se) or Sub-Neptunes (sn)

    Output:
    - predicted planet occurrence across scale heights, per 100 stars

    Let's first simplify by assuming Z_max is independent of Teff and Fe/H.
    """

    lam = 0
    teff = np.linspace(4000, 6500, 100)
    feh = np.linspace(-0.5, 0.5, 100)
    z_max = np.logspace(2, 3, 100)

    lambda_se = 0.
    lambda_se_err = 0.02
    lambda_sn = 0.26
    lambda_sn_err = 0.09

    gamma_se = -0.14
    gamma_se_err = 0.04
    gamma_sn = -0.25
    gamma_sn_err = 0.03

    tau_se = -0.3
    tau_se_err = 0.06
    tau_sn = -0.37
    tau_sn_err = 0.07

    if radius_type == 'se':
        tau = tau_se
        tau_err = tau_se_err
        gamma = gamma_se
        gamma_err = gamma_se_err
        lam = lambda_se
        lambda_err = lambda_se_err
    elif radius_type == 'sn':
        tau = tau_sn
        tau_err = tau_sn_err
        gamma = gamma_sn
        gamma_err = gamma_sn_err
        lam = lambda_sn 
        lambda_err = lambda_sn_err
        
    #gamma = 0
    gamma_err = 0
    lam = 0
    lambda_err = 0
    #feh = 0

    teff = 5772 # Teff of the Sun in K

    power = feh * lam + (teff/1000) * gamma
    power_upper = feh * (lam+lambda_err) + (teff/1000) * (gamma+gamma_err)
    power_lower = feh * (lam-lambda_err) + (teff/1000) * (gamma-gamma_err)

    n = 100 * kappa * (10**power) * z_max**tau
    n_upper = 100 * kappa * (10**power) * z_max**(tau+tau_err) # power_upper
    n_lower = 100 * kappa * (10**power) * z_max**(tau-tau_err) # power_lower (lower envelope isn't necessarily taking the minus err for every param)

    return n, n_upper, n_lower

def zink_model_simple(z_max, tau):

    teff_sun = 5772
    gamma = -0.14
    kappa = 1

    power = (teff_sun/1000) * gamma # feh and lam set to 0
    n = 100 * kappa * (10**power) * z_max**tau

    return n

def generate_models(df):
    """
    Generate galactic sculpting/enhancement models for galactic occurrence project. 
    Models are valid if they produce a planet-host fraction of 0.3-0.5.
    We assume an event that occurs at a threshold, changing the fraction of hosts from f1 to f2.

    Input: 
    - df: DataFrame of Berger+ 2020 Kepler-Gaia cross-match with iso_age column

    Output:
    - valid_models: DataFrame of valid models 
    """

    f1s = np.linspace(0, 1, 101) # planet host fraction before threshold age
    f2s = np.linspace(0, 1, 101) # planet host fraction after threshold age
    thresholds = np.linspace(1, 10, 19) # threshold age determining fraction of planet hosts in sub-population

    threshold_col = []
    f1_col = []
    f2_col = []
    f_col = []
    for f1 in f1s:
        for f2 in f2s:
            for threshold in thresholds:
                pop1 = len(df.loc[df['iso_age'] < threshold]) * f1
                pop2 = len(df.loc[df['iso_age'] >= threshold]) * f2
                f = (pop1+pop2)/len(df)
                if (f <= 0.5) and (f >= 0.3):
                    threshold_col.append(threshold)
                    f1_col.append(f1)
                    f2_col.append(f2)
                    f_col.append(f)

    valid_models = pd.DataFrame({'threshold': threshold_col, 'f1': f1_col, 'f2': f2_col, 'f': f_col})
    
    return valid_models

def plot_galactic_occurrence_models(df):
    """
    Plot valid models

    Input:
    - df: DataFrame of valid models (valid_models)
    """

    x = np.linspace(0, 12, 100)

    df_small_sculpting = df.loc[df['f1'] > df['f2']]

    """
    rows = np.linspace(0, len(df_small_sculpting), len(df_small_sculpting)+1)
    indices = rows[25::161].astype(int) # try to time the cycle so that I get a qualitatively different model each time

    df_smaller = df_small_sculpting.iloc[indices]
    print(len(df_smaller))
    """

    df_smaller = subsample_models(df_small_sculpting)

    #for row in range(len(df)):
    #for row in range(20):
    #for row in indices:
    for row in range(len(df_smaller)):
        df_row = df_smaller.iloc[row]
        f1 = df_row['f1']
        f2 = df_row['f2']
        thresh = df_row['threshold']

        y = np.where(x < thresh, f1, f2)

        plt.plot(x, y, alpha=0.5, color='powderblue')
    
    plt.xlabel('age [Gyr]')
    plt.ylabel('planet host fraction')
    plt.ylim([0, 1])
    plt.savefig(path+'galactic-occurrence/plots/valid_models.png')

    return

def subsample_models(df):
    """
    There are many valid galactic sculpting models. Let's just choose a representative sample of them. 
    It is faster to grab indices first, append to a list, and then recreate one (1) DataFrame, instead of concat-ing 10x.

    Input:
    - df: DataFrame of valid models (valid_models)
    """

    keys = []

    # retrieve only models that prescribe galactic sculpting
    df_sculpting = df.loc[df['f1'] > df['f2']]

    # randomly choose a model for each 0.5 Gyr-increment threshold
    thresholds = np.linspace(1, 10, 19) # threshold age determining fraction of planet hosts in sub-population
    for thresh in thresholds:
        df_temp = df_sculpting.loc[df_sculpting['threshold'] == thresh]
        #key = np.random.choice(np.arange(len(df_temp)))

        key = df_temp.sample(n=1).index
        keys.append(key[0])

    model_sample = df_sculpting[df_sculpting.index.isin(keys)].sort_values(by=['threshold'])

    return model_sample
