## meas_extensions_gaap

This is a measurement plugin for the LSST stack that implements the Gaussian-Aperture and PSF photometry (Kuijken 2006, Kuijken et al. 2015).

The PSF for each source is convolved with an appropriate kernel to make it a
circular Gaussian slightly larger than the original PSF by a user-defined factor.
The flux is then measured with another (circular) Gaussian aperture, so that the
sum of covariance matrix of the aperture and the Gaussianized PSF is same for
all the sources, across all the bands.

Also see:
    - meas_extensions_convolved
