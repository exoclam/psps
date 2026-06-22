########################################################################################################################
# Build completeness using a systematic injection of planets at log uniform interval of P and uniform interval of R. 
########################################################################################################################

import numpy as np
from numpy import log, exp, pi
import pandas as pd
import scipy
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, loguniform
from math import lgamma
import matplotlib.pyplot as plt
import timeit 
from datetime import datetime
from tqdm import tqdm

import psps.simulate_helpers as simulate_helpers
import psps.simulate_transit as simulate_transit
from psps.transit_class import Population, Star, GeneralStar, K2Star # Star is for Kepler stars
import psps.utils as utils

# mise en place
G = 6.6743e-8 # gravitational constant in cgs
path = '/Users/chrislam/Desktop/psps/' 
hu25_b20_kepler_b25_k2_kepmag = pd.read_csv(path+'data/joint/hu25_b20_kepler_b25_k2.csv')
hu25_b20_kepler_b25_k2_kepmag['height'] = hu25_b20_kepler_b25_k2_kepmag['height'] * 1000 # convert from kpc to pc, since Zink+23's height bins are in pc

# define P, R grids
period_grid = np.logspace(np.log10(3), np.log10(40), 10)
radius_grid = np.linspace(1.2, 4, 10)
period_grid_k2 = np.array([0.5, 5, 10, 20, 30, 40])
radius_grid_k2 = np.array([0,2,4])
period_grid_k2 = np.logspace(np.log10(3), np.log10(40), 10)
radius_grid_k2 = np.linspace(1.2, 4, 10)
# also height grid
height_bins = np.logspace(2, 3, 6) 
age_bins = np.linspace(1, 8, 7)
#height_bins = np.logspace(2, 3, 11) 

# draw eccentricities using Van Eylen+ 2019
model_flag = 'rayleigh'

# planet formation history model parameters for Population object
threshold = 11 # doesn't matter what this is, since frac1 and frac2 are the same
frac1 = 1. # you get a planet, you get a planet, everybody gets a planet
frac2 = 1.

# K2 campaign pointings, from https://archive.stsci.edu/missions-and-data/k2/campaign-fields. We need all this to do detection efficiency for K2
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

# make planets around Kepler-like stars. this is identical to how I used to calculate completeness around real stars, with two differences:
# 1) periods and radii are drawn systematically using the grids above
# 2) 100 single-planet draws are made for each bin, for statistical robustness, ie., this is how we marginalize over Kepler-like stars' Teff, CDPP, etc.

# how do I reconcile the fact that in real life, different intact vs disrupted status will lead to different transit geometry probabilities?? 
temp_df = hu25_b20_kepler_b25_k2_kepmag.copy() # copy the original DataFrame to avoid modifying it in place during each iteration

# draw stellar radius, mass, and age using asymmetric errors 
temp_df['stellar_radius'] = np.random.normal(temp_df['Rad'], temp_df['e_Rad'])
temp_df['stellar_mass'] = np.random.normal(temp_df['Mass'], temp_df['e_Mass'])
temp_df['teff_drawn'] = np.random.normal(temp_df['Teff'], temp_df['e_Teff'])
temp_df = simulate_helpers.draw_asymmetrically(temp_df, 'age', 'age_err1', 'age_err2', 'age')

# split between Kepler and K2
#temp_kepler = temp_df.loc[temp_df['Kepler_ID']>0]
#temp_k2 = temp_df.loc[temp_df['EPIC_ID']>0]
#temp_k2 = pd.merge(temp_k2, k2_pointings[['Campaign','baseline','logistic_a','logistic_k','logistic_l']], on='Campaign', how='left')

# plt.hist(temp_k2['height'], bins=height_bins, density=True, alpha=0.5, label='K2')
# plt.hist(temp_kepler['height'], bins=height_bins, density=True, alpha=0.5, label='Kepler')
# plt.xlabel(r'$Z_{max} [pc]$')
# plt.legend()
# plt.show()

# use the same 100 stars for each {p, r} draw, per mission
#temp_kepler = temp_kepler.sample(n=100).reset_index()
#temp_k2 = temp_k2.sample(n=100).reset_index()
#plt.hist(temp_kepler['teff_drawn'])
#plt.hist(temp_k2['teff_drawn'])
#plt.show()

### create a Population object to hold information about the occurrence law governing that specific population 
# here, we assume every star has a planet. 
# we also assume each planet geometrically transits, with zero eccentricity, so this is actually a *sensitivity* map. We put this together with transit geometry later to do completeness.
e = pd.Series(np.zeros(100))
incl = pd.Series(np.zeros(100))
omega = pd.Series(np.zeros(100))
angle_flag = True
#k2_sensitivity = np.zeros((len(period_grid), len(radius_grid)))
kepler_sensitivity = np.zeros((len(period_grid[1:]), len(radius_grid[1:])))
"""
for i, p in enumerate(period_grid[1:]):
	for j, r in enumerate(radius_grid[1:]):
		### guess I don't have to actually make planet and can go straight to planet detection, since all planet parameters are on rails for this exercise

		### Kepler
		# calculate semi-major axis
		a = simulate_helpers.p_to_a(p, temp_kepler['stellar_mass'])

		# calculate impact parameters; distance units in solar radii
		b = simulate_helpers.calculate_impact_parameter_vectorized(temp_kepler['stellar_radius'], r, a, e, incl, omega, angle_flag)
		
		# transit duration
		tdur = simulate_helpers.calculate_transit_duration_vectorized(p, simulate_helpers.solar_radius_to_au(temp_kepler['stellar_radius']), simulate_helpers.earth_radius_to_au(r), b, a, incl, e, omega, angle_flag)

		# calculate SN based on Eqn 4 in Christiansen et al 2012
		sn = np.array(simulate_helpers.calculate_sn_vectorized(pd.Series(p*np.ones(100)), pd.Series(r*np.ones(100)), temp_kepler['stellar_radius'], temp_kepler['CDPP6'], tdur, unit_test_flag=False))
		sn = sn.astype(float)

		# NOW I can fill in NaNs with zeros
		sn = np.nan_to_num(sn, nan=0.) #sn.fillna(0)

		# calculate Fressin detection probability based on S/N (Fressin+13 Fig 2: https://iopscience.iop.org/article/10.1088/0004-637X/766/2/81#apj460761f1)
		prob_detection = 0.1*(sn-6) # vectorize
		prob_detection = np.where(prob_detection < 0., 0., prob_detection) # replace negative probs with zeros
		prob_detection = np.where(prob_detection > 1, 1, prob_detection) # replace probs > 1 with just 1

		# sample transit status and multiplicity based on Fressin detection probability
		#transit_status = [ts1_elt * ts2_elt for ts1_elt, ts2_elt in zip(ts1, ts2)]
		transit_status = np.array([np.random.choice([1, 0], p=[pd, 1-pd]) for pd in prob_detection])
		kepler_sensitivity[i, j] = len(transit_status[transit_status==1])/100.
		#print(p, r, len(transit_status[transit_status==1])/100.) # verify which corner to start with when I read in maps

k2_sensitivity = np.zeros((len(period_grid_k2[1:]), len(radius_grid_k2[1:])))
for i, p in enumerate(period_grid_k2[1:]):
	for j, r in enumerate(radius_grid_k2[1:]):
		### K2
		# calculate SN based on MES, Eqn 5 in Zink+22 https://iopscience.iop.org/article/10.3847/1538-3881/ac2309#ajac2309t3
		mes =  simulate_transit.calculate_mes(r, temp_k2['stellar_radius'], p, temp_k2['CDPP6'], temp_k2['baseline']).astype(float)

		# calculate recovery function f, Eqn 6 in Zink+22 https://iopscience.iop.org/article/10.3847/1538-3881/ac2309#ajac2309t3
		recovery_fraction = simulate_transit.calculate_recovery_fraction(mes, temp_k2['logistic_a'], temp_k2['logistic_k'], temp_k2['logistic_l'])
		
		recovery_status = np.array([np.random.choice([1, 0], p=[rf, 1-rf]) for rf in recovery_fraction])
		k2_sensitivity[i, j] = len(recovery_status[recovery_status==1])/100.
		#print(len(recovery_status[recovery_status==1])/100.)

np.save(path+'data/joint/kepler_sensitivity.npy', kepler_sensitivity)
np.save(path+'data/joint/k2_sensitivity.npy', k2_sensitivity)
quit()
"""

### Cool, but that's for stars agnostic of Zmax. If I want to do occurrence rates as a function of Zmax, I need to get different completeness per Zmax bin. 

# how do I reconcile the fact that in real life, different intact vs disrupted status will lead to different transit geometry probabilities?? 
temp_df = hu25_b20_kepler_b25_k2_kepmag.copy() # copy the original DataFrame to avoid modifying it in place during each iteration

# draw stellar radius, mass, and age using asymmetric errors 
temp_df['stellar_radius'] = np.random.normal(temp_df['Rad'], temp_df['e_Rad'])
temp_df['stellar_mass'] = np.random.normal(temp_df['Mass'], temp_df['e_Mass'])
temp_df['teff_drawn'] = np.random.normal(temp_df['Teff'], temp_df['e_Teff'])
temp_df = simulate_helpers.draw_asymmetrically(temp_df, 'age', 'age_err1', 'age_err2', 'age')
temp_df['height_bin'] = pd.cut(temp_df['height'], bins=height_bins, labels=False)
temp_df = temp_df.dropna(subset=['height_bin'])
temp_df['height_bin'] = temp_df['height_bin'].astype(int)

kepler_sensitivity_list = []
k2_sensitivity_list = []
kepler_sensitivity_height_list = []
k2_sensitivity_height_list = []
for i, h in enumerate(height_bins[:-1]):	
	# split between Kepler and K2. 
	temp_kepler = temp_df.loc[temp_df['Kepler_ID']>0]
	temp_k2 = temp_df.loc[temp_df['EPIC_ID']>0]

	# isolate height bin
	temp_kepler_bin = temp_kepler.loc[temp_kepler['height_bin']==i]
	temp_k2_bin = temp_k2.loc[temp_k2['height_bin']==i]
	#print(np.nanmean(temp_kepler_bin['CDPP6']))
	#print(np.nanmean(temp_k2_bin['CDPP6']))

	# keep at most 100 (or 1000) stars (but this won't be possible for most height bins for K2)
	try:
		temp_kepler_bin = temp_kepler_bin.sample(n=1000).reset_index()
	except ValueError:
		temp_kepler_bin = temp_kepler_bin.reset_index()
	try:
		temp_k2_bin = temp_k2_bin.sample(n=1000).reset_index()
	except ValueError:
		temp_k2_bin = temp_k2_bin.reset_index() 
	temp_k2_bin = pd.merge(temp_k2_bin, k2_pointings[['Campaign','baseline','logistic_a','logistic_k','logistic_l']], on='Campaign', how='left')

	angle_flag = True
	k2_sensitivity = np.zeros((len(period_grid_k2[1:]), len(radius_grid_k2[1:])))
	kepler_sensitivity = np.zeros((len(period_grid[1:]), len(radius_grid[1:])))
	k2_sensitivity_height = np.zeros(len(height_bins)-1)
	kepler_sensitivity_height = np.zeros(len(height_bins)-1)

	print("height: ", height_bins[i], "Kepler: ", len(temp_kepler_bin), "K2: ", len(temp_k2_bin))
	for j, p in enumerate(period_grid[:-1]):
		for k, r in enumerate(radius_grid[:-1]):
			### Kepler sensitivity
			# calculate semi-major axis
			a = simulate_helpers.p_to_a(p, temp_kepler_bin['stellar_mass'])

			# calculate impact parameters; distance units in solar radii
			b = simulate_helpers.calculate_impact_parameter_vectorized(temp_kepler_bin['stellar_radius'], r, a, pd.Series(np.zeros(len(temp_kepler_bin))), pd.Series(np.zeros(len(temp_kepler_bin))), pd.Series(np.zeros(len(temp_kepler_bin))), angle_flag)
			
			# transit duration
			tdur = simulate_helpers.calculate_transit_duration_vectorized(p, simulate_helpers.solar_radius_to_au(temp_kepler_bin['stellar_radius']), simulate_helpers.earth_radius_to_au(r), b, a, pd.Series(np.zeros(len(temp_kepler_bin))), pd.Series(np.zeros(len(temp_kepler_bin))), pd.Series(np.zeros(len(temp_kepler_bin))), angle_flag)

			# calculate SN based on Eqn 4 in Christiansen et al 2012
			sn = np.array(simulate_helpers.calculate_sn_vectorized(pd.Series(p*np.ones(len(temp_kepler_bin))), pd.Series(r*np.ones(len(temp_kepler_bin))), temp_kepler_bin['stellar_radius'], temp_kepler_bin['CDPP6'], tdur, unit_test_flag=False))
			sn = sn.astype(float)

			# fill in NaNs with zeros
			sn = np.nan_to_num(sn, nan=0.) #sn.fillna(0)

			# calculate Fressin detection probability based on S/N (Fressin+13 Fig 2: https://iopscience.iop.org/article/10.1088/0004-637X/766/2/81#apj460761f1)
			prob_detection = 0.1*(sn-6) # vectorize
			prob_detection = np.where(prob_detection < 0., 0., prob_detection) # replace negative probs with zeros
			prob_detection = np.where(prob_detection > 1, 1, prob_detection) # replace probs > 1 with just 1

			# sample transit status and multiplicity based on Fressin detection probability
			transit_status = np.array([np.random.choice([1, 0], p=[pd, 1-pd]) for pd in prob_detection])
			kepler_sensitivity[j, k] = len(transit_status[transit_status==1])/len(temp_kepler_bin)

	for j, p in enumerate(period_grid_k2[:-1]):
		for k, r in enumerate(radius_grid_k2[:-1]):
			### K2
			# calculate SN based on MES, Eqn 5 in Zink+22 https://iopscience.iop.org/article/10.3847/1538-3881/ac2309#ajac2309t3
			mes =  simulate_transit.calculate_mes(r, temp_k2_bin['stellar_radius'], p, temp_k2_bin['CDPP6'], temp_k2_bin['baseline']).astype(float)

			# calculate recovery function f, Eqn 6 in Zink+22 https://iopscience.iop.org/article/10.3847/1538-3881/ac2309#ajac2309t3
			recovery_fraction = simulate_transit.calculate_recovery_fraction(mes, temp_k2_bin['logistic_a'], temp_k2_bin['logistic_k'], temp_k2_bin['logistic_l'])
			
			recovery_status = np.array([np.random.choice([1, 0], p=[rf, 1-rf]) for rf in recovery_fraction])
			k2_sensitivity[j, k] = len(recovery_status[recovery_status==1])/len(temp_k2_bin)

	print("Kepler sensitivity for height bin ", i, ": ", kepler_sensitivity)
	print("K2 sensitivity for height bin ", i, ": ", k2_sensitivity)
	kepler_sensitivity_list.append(kepler_sensitivity)
	k2_sensitivity_list.append(k2_sensitivity)

	### height-wise sensitivity only
	kepler_sensitivity_height = np.mean(kepler_sensitivity)
	k2_sensitivity_height = np.mean(k2_sensitivity)
	kepler_sensitivity_height_list.append(kepler_sensitivity_height)
	k2_sensitivity_height_list.append(k2_sensitivity_height)

np.save(path+'data/joint/kepler_sensitivity_list.npy', kepler_sensitivity_list) # kepler_sensitivity_list_ten.npy
np.save(path+'data/joint/k2_sensitivity_list.npy', k2_sensitivity_list)
print("Kepler sensitivity height-wise: ", kepler_sensitivity_height_list) # np.array([0.932, 0.928, 0.935, 0.933, 0.934])
print("K2 sensitivity height-wise: ", k2_sensitivity_height_list) # np.array([0.306, 0.296, 0.293, 0.284, 0.277])