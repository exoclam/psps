# psps
pspspspspsps.

psps stands for Planetary System Population Synthesizer. Basically, we make stars and planets, and we do it agnostically. psps is powered in part by JAX and is in active development. 

See [here](https://github.com/exoclam/mastrangelo/) for an example of psps in use.

You can now install it from PyPI! 

The bulk of psps's runtime comes from integrating orbits with Gala in order to calculate the maximum oscillation amplitude (Zmax) for each star in the sample. This takes 4-5 minutes on a M2 Macbook Air for a sample of 70K stars. You can choose not to run gala_galactic_heights(); the runtime for psps without this is about 30s, with the same specs and sample. 
