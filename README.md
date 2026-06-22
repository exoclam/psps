# psps
pspspspspsps.

psps stands for Planetary System Population Synthesizer. psps makes stars and planets, with an emphasis on intuitive data products. psps is powered in part by JAX and is in active development. 

See [here](https://github.com/exoclam/mastrangelo/) for an example of psps in use.

See [here](https://arxiv.org/abs/2507.21250) as well for another example of psps in use. 

You can now install it from PyPI!

```
pip install psps
```

The create/ directory contains two archetypes of scripts. 
- berger_batch.py: create synthetic systems, where each record is a system, which may have zero or one or multiple planets.
- collect_.py: run physical systems through the Kepler sensitivity function (and account for geometric transits) to get a completeness-corrected "detected" yield.

The paper_dir/ directory is messier and was used to generate plots and stats for Lam+25, in review. 

The bulk of psps's runtime comes from integrating orbits with Gala in order to calculate the maximum oscillation amplitude (Zmax) for each star in the sample. This takes 4-5 minutes on a M2 Macbook Air for a sample of 70K stars. You can choose not to run gala_galactic_heights(); the runtime for psps without this is about 30s, with the same specs and sample. 

## Update for referee report
For the purposes of keeping myself organized while addressing the referee report, here are the following steps for my workflow from the beginning:
- kepler_k2.ipynb: a single notebook where I read in HU25, B20, and B25, cross-match them, make appropriate cuts, and show diagnostic plots (e.g., Kiel and CMDs, K-S tests). 
- trilegal_kepler_k2.ipynb: a single notebook where I read in TRILEGAL queried data for Kepler and K2, make appropriate cuts, put them together, and show diagnostic plots (e.g., comparative histograms for relevant parameters). 
- make_stars.py: industrial-scale planetary system population synthesis. I make separate populations for Kepler (using the original Star class from transit_class.py) and K2 (using the K2Star class from transit_class.py), then pd.concat() them. Exports to /data/joint/stellar_samples/ folder, organized by model. 
- build_completeness.py: build sensitivity maps for Kepler and K2 by injecting planets at systematic period/radius grids and recovering them, while assuming they geometrically transit. Produces np arrays called kepler_sensitivity.npy, kepler_sensitivity_height.npy (broken down by Zmax bin), and K2 versions of those. 
- detect_planets.py: industrial-scale detection, completeness calculations, and adjusted "recovered" occurrence rates. Start from /data/joint/stellar_samples/modelXYZ/ stellar and planetary populations, and split into Kepler and K2. Run survey-specific completeness codes to get detected yields. Divide from reliability maps from Thompson+18 (Kepler) and Zink+21 (K2), to get occurrence rates. 
- make_trilegal_stars.py, build_trilegal_completeness.py, detect_trilegal_planets.py: the same workflow, but for TRILEGAL planets

Currently, I've decided to combine detect_planets.py and detect_trilegal_planets.py into detect_planets.py, merely showing physical TRILEGAL planets in addition to Z23, physical real planets, and completeness-and-reliability-adjusted real planets. 