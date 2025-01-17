##################################################
####### For plotting, etc ########################
##################################################

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

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns

from psps.transit_class import Population, Star
import psps.simulate_helpers as simulate_helpers
import psps.simulate_transit as simulate_transit

path = '/Users/chrislam/Desktop/psps/' 

def plot_properties(teffs, ages):
    """
    Make 2-subplot figure showing distributions of Teff and age. Tentatively Figs 1 & 2, in Paper III

    Input: 
    - teffs: np array of effective temps [K]
    - ages: np array of stellar ages [Gyr]

    """

    ### VISUALIZE TRILEGAL SAMPLE PROPERTIES, FOR PAPER FIGURE
    teff_hist, teff_bin_edges = np.histogram(teffs, bins=50)
    print("Teff peak: ", teff_bin_edges[np.argmax(teff_hist)])
    age_hist, age_bin_edges = np.histogram(ages, bins=50)
    print("age peak: ", age_bin_edges[np.argmax(age_hist)])

    #fig, axes = plt.subplots(figsize=(7,5))
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7, 5))

    #ax1 = plt.subplot2grid((2,1), (0,0))
    ax1.hist(teffs, bins=50, alpha=0.7)
    ax1.set_ylabel("count")
    ax1.set_xlabel(r"$T_{eff}$ [K]")
    # plot vertical red line through median Teff
    ax1.plot([np.median(teffs), np.median(teffs)], 
            [0,4000], color='r', alpha=0.3, linestyle='--', label=r'median $T_{eff}$')
    #ax1.set_xlim([4800, 7550])
    ax1.legend()

    #ax2 = plt.subplot2grid((2,1), (1,0))
    ax2.hist(ages, bins=50, alpha=0.7)
    # plot vertical red line through median age 
    ax2.plot([np.median(ages), np.median(ages)], 
            [0,3600], color='r', alpha=0.3, linestyle='--', label='median age')
    ax2.set_ylabel("count")
    ax2.set_xlabel("age [Gyr]")
    #ax2.set_xlim([0, 18])
    ax2.legend()
    fig.tight_layout()

    print("median Teff: ", np.median(teffs))
    print("median age: ", np.median(ages))

    plt.savefig(path+'plots/sample_properties_trilegal_heights_only.pdf', format='pdf')
    plt.show()

    return


def plot_models(thresholds, frac1s, frac2s, ax=None):
    """
    Make Fig 3 in Paper III, ie. a sample of the step function models for which we later show results 
    
    Inputs:
    - thresholds: list of time at which f1 goes to f2 (cosmic age) [Gyr]
    - frac1s: list of initial planet host fraction, before threshold
    - frac2s: list of planet host fraction after threshold
    - ax: matplotlib ax object, for modular plotting

    """
    
    x = np.linspace(0, 14, 1000)
    if ax is None:
        # step model
        for i in range(len(frac1s)):
            threshold = thresholds[i]
            frac1 = frac1s[i]
            frac2 = frac2s[i]

            y = np.where(x <= threshold, frac1, frac2)

            plt.plot(x, y, color='powderblue')
            plt.xlabel('cosmic age [Gyr]')
            plt.ylabel('planet host fraction')
            plt.ylim([0,1])

        plt.savefig(path+'plots/models.png', format='png', bbox_inches='tight')
        plt.show()

    else:
        # general models
        for i in range(len(frac1s)):
            frac1 = frac1s[i]
            frac2 = frac2s[i]

            b = frac1
            m = (frac2 - frac1)/(x[-1] - x[0])
            y = b + m * x

        ax.plot(x, y, color='powderblue')
        ax.set_xlabel('cosmic age [Gyr]')
        ax.set_ylabel('planet host fraction')
        ax.set_ylim([0,1])

        return ax            

    return

def completeness(berger_kepler):
    """"
    Build completeness map to characterize psps detection pipeline

    - For each {period, radius} bin, simulate 100 planetary systems
    - Calculate how many are geometric transits
    - Calculate how many are detections
    - Output completeness map that can be used to back out "true" occurrence, a la IDEM method, eg. Dressing & Charbonneau 2015
    (https://iopscience.iop.org/article/10.1088/0004-637X/807/1/45/meta#apj515339s7)

    The result should resemble similar maps for FGK dwarfs.

    """

    # mise en place
    period_grid = np.logspace(np.log10(2), np.log10(300), 10)
    radius_grid = np.linspace(1, 4, 10)
    completeness_map = np.ndarray((9, 9))
    
    frac_hosts = np.ones(len(berger_kepler))

    for p_elt, p in tqdm(enumerate(period_grid[:-1])):
        for r_elt, r in enumerate(radius_grid[:-1]):
            star_data = []
            alpha_se = np.random.normal(-1., 0.2)
            alpha_sn = np.random.normal(-1.5, 0.1)

            # draw stellar radius, mass, and age using asymmetric errors 
            berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler, 'iso_rad', 'iso_rad_err1', 'iso_rad_err2', 'stellar_radius')
            berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_age', 'iso_age_err1', 'iso_age_err2', 'age')
            berger_kepler_temp = simulate_helpers.draw_asymmetrically(berger_kepler_temp, 'iso_mass', 'iso_mass_err1', 'iso_mass_err2', 'stellar_mass')

            for i in range(len(berger_kepler)):
                # create one planet with {p, r} in that system
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
                
                # re-assign planet period and radius to the appropriate grid element
                period = np.random.uniform(p, period_grid[p_elt+1])
                radius = np.random.uniform(r, radius_grid[r_elt+1])
                star_update['planet_radii'] = radius
                star_update['periods'] = period
                
                star_update['incls'] = star_update['incls'][0]
                star_update['mutual_incls'] = star_update['mutual_incls'][0]
                star_update['eccs'] = star_update['eccs'][0]
                star_update['omegas'] = star_update['omegas'][0]

                star_data.append(star_update)

            # convert back to DataFrame
            berger_kepler_all = pd.DataFrame.from_records(star_data)

            # calculate geometric transits and detections
            prob_detections, transit_statuses, sn, geom_transit_statuses = simulate_transit.calculate_transit_vectorized(berger_kepler_all.periods, 
                                            berger_kepler_all.stellar_radius, berger_kepler_all.planet_radii,
                                            berger_kepler_all.eccs, 
                                            berger_kepler_all.incls, 
                                            berger_kepler_all.omegas, berger_kepler_all.stellar_mass,
                                            berger_kepler_all.rrmscdpp06p0, angle_flag=True) 

            berger_kepler_all['transit_status'] = transit_statuses[0]
            berger_kepler_all['prob_detections'] = prob_detections[0]
            berger_kepler_all['sn'] = sn
            berger_kepler_all['geom_transit_status'] = geom_transit_statuses

                # need kepid to be str or tuple, else unhashable type when groupby.count()
            berger_kepler_all['kepid'] = berger_kepler_all['kepid'].apply(str) 
            #print(berger_kepler_all[['planet_radii', 'periods', 'transit_status']])
            #print(berger_kepler_all.loc[berger_kepler_all['transit_status']==1][['planet_radii', 'periods', 'transit_status']])
            #quit()

            # isolate detected transiting planets
            berger_kepler_transiters = berger_kepler_all.loc[berger_kepler_all['transit_status']==1]

            completeness = len(berger_kepler_transiters)/len(berger_kepler)
            #print(p, r, completeness)
            completeness_map[p_elt][r_elt] = completeness
    
    #print(completeness_map)

    return completeness_map

def plot_completeness(mean_completeness_map, std_completeness_map, radius_grid, period_grid):

    """
    Plot cell by cell completeness over radius and period space, with errorbars

    Input: 
    - mean_completeness_map: completeness map, averaged over 30 detection pipeline realizations
    - std_completeness_map: std of completeness map
    - radius_grid: np.linspace(1, 4, 10)
    - period_grid: np.logspace(2, 300, 10)
    """

    # several cells have uncertainties of 0% because there is only one surviving non-NaN realization; get rid of those, too 
    # some of them still round to 0%, though
    #mean_completeness_map[std_completeness_map == 0] = np.nan
    #std_completeness_map[std_completeness_map == 0] = np.nan
    #print("mean map: ", mean_completeness_map)
    #print("std map: ", std_completeness_map)

    # mask NaNs
    #mean_completeness_ma = np.ma.masked_invalid(mean_completeness_map)
    #std_completeness_map = np.ma.masked_invalid(std_completeness_map)

    # plot
    f, ((ax1)) = plt.subplots(1, 1, figsize=(8, 8))
    formatted_text = (np.asarray(["{0}±{1}%".format( 
        np.round(100*mean, 1), np.round(100*std, 1)) for mean, std in zip(mean_completeness_map.flatten(), std_completeness_map.flatten())])).reshape(9, 9) 
    sns.heatmap(mean_completeness_map, yticklabels=np.around(radius_grid, 1), xticklabels=np.around(period_grid, 0), vmin=0., vmax=1., cmap='Blues', cbar_kws={'label': 'completeness'}, annot=formatted_text, fmt="", annot_kws={"size": 7})
    ax1.set_xticks(ax1.get_xticks()[::2]) # sample every other tick, for cleanness
    ax1.set_yticks(ax1.get_yticks()[::2]) # sample every other tick, for cleanness
    ax1.invert_yaxis()
    plt.xlabel('period [days]')
    plt.ylabel('radius [$R_{\oplus}$]')
    #plt.xticks(ticks=period_grid)
    #plt.yticks(ticks=radius_grid)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path+'plots/sensitivity.png')
    plt.show()

    return