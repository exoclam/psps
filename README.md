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
