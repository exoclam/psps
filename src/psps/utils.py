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
from scipy import stats

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
            [0,3000], color='r', alpha=0.3, linestyle='--', label=r'median $T_{eff}$')
    #ax1.set_xlim([4800, 7550])
    ax1.legend()

    #ax2 = plt.subplot2grid((2,1), (1,0))
    ax2.hist(ages, bins=50, alpha=0.7)
    # plot vertical red line through median age 
    ax2.plot([np.median(ages), np.median(ages)], 
            [0,1800], color='r', alpha=0.3, linestyle='--', label='median age')
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

def plot_host_vs_height(df_all, df_planets):
    """
    Figure of scatter points, one per system, color-coded by host/not-host
    Y axis is Zmax. X axis is arbitrary. 

    Args:
        df_all (pandas DataFrame): of all systems, eg. trilegal_kepler_all or berger_kepler_all
        df_planets (pd DataFrame): of only planet-hosting systems
    """

    df_none = df_all.loc[df_all['num_planets'].isnull()]
    df_none = df_none.reset_index()
    
    df_planets = df_planets.reset_index()

    # randomize indices so hosts and non-hosts aren't bunched up together 
    len_total = len(df_none)+len(df_planets)
    pre_shuffled_index = np.linspace(0, len_total-1, len_total)
    np.random.shuffle(pre_shuffled_index) 

    df_none['shuffled_index'] = pre_shuffled_index[:len(df_none)]
    df_planets['shuffled_index'] = pre_shuffled_index[len(df_none):]

    # bin DF by height
    all1 = df_all.loc[(df_all['height'] > 100) & (df_all['height'] <= np.logspace(2,3,6)[1])]
    all2 = df_all.loc[(df_all['height'] > np.logspace(2,3,6)[1]) & (df_all['height'] <= np.logspace(2,3,6)[2])]
    all3 = df_all.loc[(df_all['height'] > np.logspace(2,3,6)[2]) & (df_all['height'] <= np.logspace(2,3,6)[3])]
    all4 = df_all.loc[(df_all['height'] > np.logspace(2,3,6)[3]) & (df_all['height'] <= np.logspace(2,3,6)[4])]
    all5 = df_all.loc[(df_all['height'] > np.logspace(2,3,6)[4]) & (df_all['height'] <= 1000)]

    # make custom RHS labels
    custom_locs = np.logspace(2, 3, 6)[1:]
    custom_labels = np.round(np.array([np.nanmean(all1['frac_host']), np.nanmean(all2['frac_host']), np.nanmean(all3['frac_host']), np.nanmean(all4['frac_host']), np.nanmean(all5['frac_host'])]), 2)

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.scatter(df_none['shuffled_index'], df_none['height'], s=10, alpha=0.6, color='#CB4335', label='non-host', zorder=1)
    ax.scatter(df_planets['shuffled_index'], df_planets['height'], s=10, alpha=0.6, color='steelblue', label='host', zorder=2)
    ax.set_xlabel('star index')
    ax.set_ylabel(r'$Z_{max}$ [pc]')

    ax1 = ax.secondary_yaxis('right')
    ax1.set_yticks(custom_locs)
    ax1.set_yticklabels(custom_labels)
    ax1.set_ylabel("< host fraction > at this height")

    fig.tight_layout()
    plt.legend()
    plt.savefig(path+'plots/color-code-frac-host.png')
    #plt.show()

    return

def plot_age_vs_height_scatter(df_all):
    """
    Figure of scatter points, one per system, color-coded by age
    Y axis is Zmax. X axis is arbitrary. 

    Args:
        df_all (pandas DataFrame): of all systems, eg. trilegal_kepler_all or berger_kepler_all
    """

    df1 = df_all.loc[(df_all['age'] > 0) & (df_all['age'] <= 2)] 
    df2 = df_all.loc[(df_all['age'] > 2) & (df_all['age'] <= 4)] 
    df3 = df_all.loc[(df_all['age'] > 4) & (df_all['age'] <= 6)] 
    df4 = df_all.loc[(df_all['age'] > 6) & (df_all['age'] <= 8)] 
    df5 = df_all.loc[(df_all['age'] > 8)] 

    # bin DF by height
    all1 = df_all.loc[(df_all['height'] > 100) & (df_all['height'] <= np.logspace(2,3,6)[1])]
    all2 = df_all.loc[(df_all['height'] > np.logspace(2,3,6)[1]) & (df_all['height'] <= np.logspace(2,3,6)[2])]
    all3 = df_all.loc[(df_all['height'] > np.logspace(2,3,6)[2]) & (df_all['height'] <= np.logspace(2,3,6)[3])]
    all4 = df_all.loc[(df_all['height'] > np.logspace(2,3,6)[3]) & (df_all['height'] <= np.logspace(2,3,6)[4])]
    all5 = df_all.loc[(df_all['height'] > np.logspace(2,3,6)[4]) & (df_all['height'] <= 1000)]

    # make custom RHS labels
    custom_locs = np.logspace(2, 3, 6)[1:]
    custom_labels = np.round(np.array([np.nanmean(all1['age']), np.nanmean(all2['age']), np.nanmean(all3['age']), np.nanmean(all4['age']), np.nanmean(all5['age'])]), 2)

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.scatter(df1.index, df1['height'], s=5, alpha=0.5, color='#7D3C98', label='0-2 Gyr', zorder=5)
    ax.scatter(df2.index, df2['height'], s=5, alpha=0.5, color='#2E86C1', label='2-4 Gyr', zorder=4)
    ax.scatter(df3.index, df3['height'], s=5, alpha=0.5, color='#138D75', label='4-6 Gyr', zorder=3)
    ax.scatter(df4.index, df4['height'], s=5, alpha=0.5, color='#D4AC0D', label='6-8 Gyr', zorder=2)
    ax.scatter(df5.index, df5['height'], s=5, alpha=0.5, color='#CB4335', label='>8 Gyr', zorder=1)
    ax.set_xlabel('star index')
    ax.set_ylabel(r'$Z_{max}$ [pc]')

    ax1 = ax.secondary_yaxis('right')
    ax1.set_yticks(custom_locs)
    ax1.set_yticklabels(custom_labels)
    ax1.set_ylabel("< age > at this height [Gyr]")

    fig.tight_layout()
    plt.legend()
    plt.savefig(path+'plots/color-code-age.png')
    #plt.show()

    return

def plot_kepmag_vs_cdpp(new_df):

    """
    Plot Kepler magnitude vs CDPP (6 hr)

    Input:
    - new_df: pd.DataFrame with Kep mag and CDPP columns
    """

    ### test by visualizing
    new_df_f = new_df.loc[(new_df['Teff'] <= 7500) & (new_df['Teff'] >= 6000)]
    new_df_g = new_df.loc[(new_df['Teff'] <= 6000) & (new_df['Teff'] >= 5300)]
    new_df_k = new_df.loc[(new_df['Teff'] <= 5300) & (new_df['Teff'] >= 3500)]
    plt.scatter(new_df['Kepler'], new_df['cdpp'], s=10, alpha=0.2)
    plt.scatter(new_df_f['Kepler'], new_df_f['cdpp'], s=10, c='#e1bb3e', alpha=0.5, label='F dwarfs')
    plt.scatter(new_df_g['Kepler'], new_df_g['cdpp'], s=10, c='#e35436', alpha=0.5, label='G dwarfs')
    plt.scatter(new_df_k['Kepler'], new_df_k['cdpp'], s=10, c='#891d1a', alpha=0.5, label='K dwarfs')
    plt.xlabel(r'$m_{Kepler}$')
    plt.ylabel('CDPP rms (6 hr) [ppm]')
    plt.ylim([0, 1000])
    plt.legend(bbox_to_anchor=(1., 1.05))
    plt.tight_layout()
    plt.savefig(path+'plots/trilegal/kepmag_vs_cdpp.png')
    plt.show()

    return

def plot_kepmag_vs_cdpp_heatmap(new_df):

    """
    Plot Kepler magnitude vs CDPP (6 hr)

    Input:
    - new_df: pd.DataFrame with Kep mag and CDPP columns
    """

    bins2d = [np.linspace(8, 16, 20), np.linspace(0, 500, 20)]

    def actual_plotting(df):

        kepmags = df['Kepler']
        cdpps = df['cdpp']
        norm = 10
        hist, xedges, yedges = np.histogram2d(kepmags, cdpps, bins=bins2d)
        hist = hist.T
        #with np.errstate(divide='ignore', invalid='ignore'):  # suppress division by zero warnings
            #hist *= norm / hist.sum(axis=0, keepdims=True)
            #hist *= norm / hist.sum(axis=1, keepdims=True)
        plt.pcolormesh(xedges, yedges, hist, cmap='Blues')

        plt.xlabel(r'$m_{Kepler}$')
        plt.ylabel('CDPP rms (6 hr) [ppm]')
        plt.xlim([8, 16])
        plt.ylim([0, 200])
        plt.legend(bbox_to_anchor=(1., 1.05))
        plt.tight_layout()
        #plt.savefig(path+'plots/trilegal/kepmag_vs_cdpp.png', format='png', bbox_inches='tight')
        plt.show()

    ### test by visualizing
    new_df_f = new_df.loc[(new_df['Teff'] <= 7500) & (new_df['Teff'] >= 6000)]
    new_df_g = new_df.loc[(new_df['Teff'] <= 6000) & (new_df['Teff'] >= 5300)]
    new_df_k = new_df.loc[(new_df['Teff'] <= 5300) & (new_df['Teff'] >= 3500)]

    actual_plotting(new_df)
    actual_plotting(new_df_f)
    actual_plotting(new_df_g)
    #actual_plotting(new_df_k)

    return

def plot_kepmag_vs_cdpp_heatmap_berger(old_df):

    """
    Plot Kepler magnitude vs CDPP (6 hr)

    Input:
    - old_df: pd.DataFrame with kepmag and rrmscdpp06p0 columns
    """

    bins2d = [np.linspace(8, 16, 20), np.linspace(0, 500, 20)]
    old_df = old_df.dropna(subset=['kepmag','rrmscdpp06p0'])

    def actual_plotting(df):

        kepmags = df['kepmag']
        cdpps = df['rrmscdpp06p0']
        norm = 10
        hist, xedges, yedges = np.histogram2d(kepmags, cdpps, bins=bins2d)
        hist = hist.T
        plt.pcolormesh(xedges, yedges, hist, cmap='Greens')

        plt.xlabel(r'$m_{Kepler}$')
        plt.ylabel('CDPP rms (6 hr) [ppm]')
        plt.xlim([8, 16])
        plt.ylim([0, 500])
        plt.legend(bbox_to_anchor=(1., 1.05))
        plt.tight_layout()
        #plt.savefig(path+'plots/trilegal/kepmag_vs_cdpp.png', format='png', bbox_inches='tight')
        plt.show()

    ### test by visualizing
    old_df_f = old_df.loc[(old_df['iso_teff'] <= 7500) & (old_df['iso_teff'] >= 6000)]
    old_df_g = old_df.loc[(old_df['iso_teff'] <= 6000) & (old_df['iso_teff'] >= 5300)]
    old_df_k = old_df.loc[(old_df['iso_teff'] <= 5300) & (old_df['iso_teff'] >= 3500)]

    actual_plotting(old_df)
    actual_plotting(old_df_f)
    actual_plotting(old_df_g)

    return

def plot_age_vs_height(new_df, label='TRI'):

    """
    Plot age vs height

    Input:
    - new_df: pd.DataFrame with age and height columns
    - label: 'TRI' or 'B20'
    """

    bins2d = [np.linspace(0, 14, 20), np.logspace(2, 3, 20)]

    def actual_plotting(df):

        ages = df['age']
        heights = df['height']
        norm = 10
        hist, xedges, yedges = np.histogram2d(ages, heights, bins=bins2d)
        hist = hist.T
        #with np.errstate(divide='ignore', invalid='ignore'):  # suppress division by zero warnings
            #hist *= norm / hist.sum(axis=0, keepdims=True)
            #hist *= norm / hist.sum(axis=1, keepdims=True)
        if label=='TRI':
            plt.pcolormesh(xedges, yedges, hist, cmap='Blues')
            plt.xlabel('TRILEGAL age [Gyr]')
            plt.ylabel('TRILEGAL height [pc]')
        elif label=='B20':
            plt.pcolormesh(xedges, yedges, hist, cmap='Greens')
            plt.xlabel('B20 age [Gyr]')
            plt.ylabel('B20 height [pc]')
        plt.legend(bbox_to_anchor=(1., 1.05))
        plt.tight_layout()
        if label=='TRI':
            plt.savefig(path+'plots/trilegal/age_vs_height.png', format='png', bbox_inches='tight')
        elif label=='B20':
            plt.savefig(path+'plots/age_vs_height.png', format='png', bbox_inches='tight')
        plt.show()

    ### test by visualizing
    #new_df_f = new_df.loc[(new_df['Teff'] <= 7500) & (new_df['Teff'] >= 6000)]
    #new_df_g = new_df.loc[(new_df['Teff'] <= 6000) & (new_df['Teff'] >= 5300)]
    #new_df_k = new_df.loc[(new_df['Teff'] <= 5300) & (new_df['Teff'] >= 3500)]

    actual_plotting(new_df)
    #actual_plotting(new_df_f)
    #actual_plotting(new_df_g)
    #actual_plotting(new_df_k)

    return

""" 
STATISTICS
"""

def rejection_sampling(data1, data2):
    """
    Args:
        data1 (np.array): sub-sample
        data2 (np.array): bigger population

    Returns:
        new_data1: sub-sample
        new_data2: rejection-sampled population
    """

    if drawn=='age':
        x = np.linspace(0.5, 13.5, 100)
    elif drawn=='stellar_radius':
        x = np.linspace(0.5, 5., 100)
    elif drawn=='stellar_mass':
        x = np.linspace(0.5, 2.5, 100)
    elif drawn=='stellar_teff':
        x = np.linspace(2000, 7500, 1000)
    elif drawn=='stellar_feh':
        x = np.linspace(-0.5, 0.5, 100)

    # set constant at max height of sample distribution 
    c_age = np.max(data1['age'])
    c_radius = np.max(data1['stellar_radius'])
    c_mass = np.max(data1['stellar_mass'])
    c_teff = np.max(data1['stellar_teff'])
    c_feh = np.max(data1['stellar_feh'])

    drawn = 'age'
    kernel = stats.gaussian_kde(data1[drawn])
    y = np.reshape(kernel(x).T, x.shape)
    print(y)
    
    # how to construct a second kde that minimally matches the shape of the first one? 

    return new_data1, new_data2

def matched_sampling(data1, data2):
    """
    Args:
        data1 (np.array): template population, eg. B20
        data2 (np.array): copycat population, eg. trilegal

    Returns:
        new_data2: re-sampled copycat population
    """

    #print(len(data1))
    #print(len(data2))
    # columns to match/group by on
    cols_to_match = ['mag_bins', 'stellar_radius_bins', 'teff_bins', 'logg_bins', 'age_bins', 'height_bins']
    #cols_to_match = ['stellar_radius_bins', 'cdpp_bins', 'age_bins']

    ### create binned map across Teff, logg, and kepmag, per van Saders+19, https://iopscience.iop.org/article/10.3847/1538-4357/aafafe
    #mass_bins = np.linspace(0, 3, 30)
    logg_bins = np.linspace(3.0, 4.6, 5) 
    teff_bins = np.linspace(5300, 7500, 50)
    mag_bins = np.linspace(8, 16, 5)
    age_bins = np.linspace(0, 8, 10)
    stellar_radius_bins = np.linspace(1, 3.5, 5)
    cdpp_bins = np.linspace(0, 100, 10)
    height_bins = np.logspace(2,3,6)

    data1['mag_bins'] = pd.cut(data1['kepmag'], bins=mag_bins, include_lowest=True)
    data1['logg_bins'] = pd.cut(data1['iso_logg'], bins=logg_bins, include_lowest=True)
    data1['teff_bins'] = pd.cut(data1['iso_teff'], bins=teff_bins, include_lowest=True)
    data1['stellar_radius_bins'] = pd.cut(data1['iso_rad'], bins=stellar_radius_bins, include_lowest=True)
    #data1['cdpp_bins'] = pd.cut(data1['rrmscdpp06p0'], bins=cdpp_bins, include_lowest=True)
    data1['height_bins'] = pd.cut(data1['height'], bins=height_bins, include_lowest=True)
    data1['age_bins'] = pd.cut(data1['age'], bins=age_bins, include_lowest=True)

    data1_counts = data1.groupby(cols_to_match).count().reset_index()
    data1_counts = data1_counts.pivot(index='mag_bins', columns=['stellar_radius_bins', 'teff_bins', 'age_bins', 'logg_bins','height_bins'], values='kepmag')
    #print(data1_counts)

    ### create binned map for TRILEGAL and begin drawing stars. 
    # allocate each drawn star into a bin and divide map by B20 map. 
    # as long as all bins <1, keep drawing stars with replacement.
    data2['mag_bins'] = pd.cut(data2['Kepler'], bins=mag_bins, include_lowest=True)
    data2['logg_bins'] = pd.cut(data2['logg'], bins=logg_bins, include_lowest=True)
    data2['teff_bins'] = pd.cut(data2['Teff'], bins=teff_bins, include_lowest=True)
    data2['stellar_radius_bins'] = pd.cut(data2['stellar_radius'], bins=stellar_radius_bins, include_lowest=True)
    #data2['cdpp_bins'] = pd.cut(data2['cdpp'], bins=cdpp_bins, include_lowest=True)
    data2['height_bins'] = pd.cut(data2['height'], bins=height_bins, include_lowest=True)
    data2['age_bins'] = pd.cut(data2['age'], bins=age_bins, include_lowest=True)
    #print(data2[['mag_bins','teff_bins','stellar_radius_bins']])
    #print(data2)

    """
    # start a template of draws, with an initial set using sample() to make it go faster
    sample = data2.sample(frac=100, replace=True)#.groupby(['mag_bins','logg_bins','teff_bins']).count()
    sample_counts = sample.groupby(cols_to_match).count().reset_index()
    sample_counts = sample_counts.pivot(index='logg_bins', columns=['mag_bins','teff_bins'], values='Teff')
    print(sample_counts)

    # divide maps and show only those less than 1
    ratio = sample_counts/data1_counts
    print(np.nansum(ratio[ratio < 1]))
    """

    data1_unstacked = data1_counts.unstack()
    data1_unstacked = data1_unstacked.loc[data1_unstacked > 0].reset_index()
    #print(data1_unstacked)
    data1_unstacked.columns = ['stellar_radius_bins','teff_bins','age_bins','logg_bins','height_bins','mag_bins','thresh']
    #print(data1_unstacked)

    # Merge the DataFrames on the specified columns, using an inner join
    valid_pool = pd.merge(data2, data1_unstacked, on=cols_to_match, how='inner')
    #print(valid_pool)

    # Group by and then sample
    def sample_by_size(group):
        n = group['thresh'].iloc[0]
        return group.sample(n=n, replace = True)

    new_data2 = valid_pool.groupby(cols_to_match, group_keys=False).apply(sample_by_size)

    return new_data2


def infer_stellar_radius(reference, mass, lum, teff):
    """
    Avoid Stefan-Boltzmann because it under-predicts radius.

    Args:
        reference (pd.DataFrame): reference DF
        mass (pd.Series): stellar mass [solar masses]
        lum (pd.Series): stellar luminosity [solar lum]
        teff (pd.Series): effective temp [K]
    """

    mass_bins = np.linspace(0, 3, 100)
    lum_bins = np.linspace(0, 1000, 100) 
    teff_bins = np.linspace(4800, 7500, 100)

    return radius