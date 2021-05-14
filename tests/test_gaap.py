# This file is part of meas_extensions_gaap
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org/).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.

import math
import unittest
import galsim
import itertools
import lsst.afw.display as afwDisplay
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.geom as geom
from lsst.pex.exceptions import InvalidParameterError
import lsst.meas.base as measBase
import lsst.meas.base.tests
import lsst.meas.extensions.gaap
import lsst.utils.tests
import numpy as np
import scipy


try:
    type(display)
except NameError:
    display = False
    frame = 1


def makeGalaxyExposure(scale, psfSigma=0.9, flux=1000., galSigma=3.7, variance=1.0):
    """Make an ideal exposure of circular Gaussian

    For the purpose of testing Gaussian Aperture and PSF algorithm (GAaP), this
    generates a noiseless image of circular Gaussian galaxy of a desired total
    flux convolved by a circular Gaussian PSF. The Gaussianity of the galaxy
    and the PSF allows comparison with analytical results, modulo pixelization.

    Parameters
    ----------
    scale : `float`
        Pixel scale in the exposure.
    psfSigma : `float`
        Sigma of the circular Gaussian PSF.
    flux : `float`
        The total flux of the galaxy.
    galSigma : `float`
        Sigma of the pre-seeing circular Gaussian galaxy.

    Returns
    -------
    exposure, center
        A tuple containing an lsst.afw.image.Exposure and lsst.geom.Point2D
        objects, corresponding to the galaxy image and its centroid.
    """
    psfWidth = 2*int(4.0*psfSigma) + 1
    galWidth = 2*int(40.*math.hypot(galSigma, psfSigma)) + 1
    gal = galsim.Gaussian(sigma=galSigma, flux=flux)

    galIm = galsim.Image(galWidth, galWidth)
    galIm = galsim.Convolve([gal, galsim.Gaussian(sigma=psfSigma, flux=1.)]).drawImage(image=galIm,
                                                                                       scale=1.0,
                                                                                       method='no_pixel')
    exposure = afwImage.makeExposure(afwImage.makeMaskedImageFromArrays(galIm.array))
    exposure.setPsf(afwDetection.GaussianPsf(psfWidth, psfWidth, psfSigma))

    exposure.variance.set(variance)
    exposure.mask.set(0)
    center = exposure.getBBox().getCenter()

    cdMatrix = afwGeom.makeCdMatrix(scale=scale)
    exposure.setWcs(afwGeom.makeSkyWcs(crpix=center,
                                       crval=geom.SpherePoint(0.0, 0.0, geom.degrees),
                                       cdMatrix=cdMatrix))
    return exposure, center


class GaapFluxTestCase(lsst.utils.tests.TestCase):
    """Main test case for the GAaP plugin.
    """
    def setUp(self):
        self.center = lsst.geom.Point2D(100.0, 770.0)
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(-20, -30),
                                    lsst.geom.Extent2I(240, 1600))
        self.dataset = lsst.meas.base.tests.TestDataset(self.bbox)

        # We will consider three sources in our test case
        # recordId = 0: A bright point source
        # recordId = 1: An elliptical (Gaussian) galaxy
        # recordId = 2: A source near a corner
        self.dataset.addSource(1000., self.center - lsst.geom.Extent2I(0, 100))
        self.dataset.addSource(1000., self.center + lsst.geom.Extent2I(0, 100),
                               afwGeom.Quadrupole(9., 9., 4.))
        self.dataset.addSource(600., lsst.geom.Point2D(self.bbox.getMin()) + lsst.geom.Extent2I(10, 10))

    def tearDown(self):
        del self.center
        del self.bbox
        del self.dataset

    def makeAlgorithm(self, gaapConfig=None):
        schema = lsst.meas.base.tests.TestDataset.makeMinimalSchema()
        if gaapConfig is None:
            gaapConfig = lsst.meas.extensions.gaap.GaapFluxConfig()
        gaapPlugin = lsst.meas.extensions.gaap.GaapFluxPlugin(gaapConfig, 'ext_gaap_GaapFlux', schema, None)
        return gaapPlugin, schema

    def check(self, psfSigma=0.5, flux=1000., scalingFactors=[1.15], forced=False):
        """Check for non-negative values for GAaP instFlux and instFluxErr.
        """
        scale = 0.1*geom.arcseconds

        TaskClass = measBase.ForcedMeasurementTask if forced else measBase.SingleFrameMeasurementTask

        # Create an image of a tiny source
        exposure, center = makeGalaxyExposure(scale, psfSigma, flux, galSigma=0.001, variance=0.)

        measConfig = TaskClass.ConfigClass()
        algName = "ext_gaap_GaapFlux"

        measConfig.plugins.names.add(algName)

        if forced:
            measConfig.copyColumns = {"id": "objectId", "parent": "parentObjectId"}

        algConfig = measConfig.plugins[algName]
        algConfig.scalingFactors = scalingFactors
        algConfig.scaleByFwhm = True

        if forced:
            offset = geom.Extent2D(-12.3, 45.6)
            refWcs = exposure.getWcs().copyAtShiftedPixelOrigin(offset)
            refSchema = afwTable.SourceTable.makeMinimalSchema()
            centroidKey = afwTable.Point2DKey.addFields(refSchema, "my_centroid", doc="centroid",
                                                        unit="pixel")
            shapeKey = afwTable.QuadrupoleKey.addFields(refSchema, "my_shape", "shape")
            refSchema.getAliasMap().set("slot_Centroid", "my_centroid")
            refSchema.getAliasMap().set("slot_Shape", "my_shape")
            refSchema.addField("my_centroid_flag", type="Flag", doc="centroid flag")
            refSchema.addField("my_shape_flag", type="Flag", doc="shape flag")
            refCat = afwTable.SourceCatalog(refSchema)
            refSource = refCat.addNew()
            refSource.set(centroidKey, center + offset)
            refSource.set(shapeKey, afwGeom.Quadrupole(1.0, 1.0, 0.0))

            refSource.setCoord(refWcs.pixelToSky(refSource.get(centroidKey)))
            taskInitArgs = (refSchema,)
            taskRunArgs = (refCat, refWcs)
        else:
            taskInitArgs = (afwTable.SourceTable.makeMinimalSchema(),)
            taskRunArgs = ()

        # Activate undeblended measurement with the same configuration
        measConfig.undeblended.names.add(algName)
        measConfig.undeblended[algName] = measConfig.plugins[algName]

        # We are no longer going to change the configs.
        # So validate and freeze as they would happen when run from a CLI
        measConfig.validate()
        measConfig.freeze()

        algMetadata = dafBase.PropertyList()
        task = TaskClass(*taskInitArgs, config=measConfig, algMetadata=algMetadata)

        schema = task.schema
        measCat = afwTable.SourceCatalog(schema)
        source = measCat.addNew()
        source.getTable().setMetadata(algMetadata)
        ss = afwDetection.FootprintSet(exposure.getMaskedImage(), afwDetection.Threshold(10.0))
        fp = ss.getFootprints()[0]
        source.setFootprint(fp)

        task.run(measCat, exposure, *taskRunArgs)

        if display:
            disp = afwDisplay.Display(frame)
            disp.mtv(exposure)
            disp.dot("x", *center, origin=afwImage.PARENT, title="psfSigma=%f" % (psfSigma,))

        self.assertFalse(source.get(algName + "_flag"))  # algorithm succeeded

        # We first check if it produces a positive number (non-nan)
        for baseName in algConfig.getAllGaapResultNames(algName):
            self.assertTrue((source.get(baseName + "_instFlux") >= 0))
            self.assertTrue((source.get(baseName + "_instFluxErr") >= 0))

        # For sF > 1, check that the measured value is close to the true value
        for baseName in algConfig.getAllGaapResultNames(algName):
            if "_1_0x_" not in baseName:
                self.assertFloatsAlmostEqual(source.get(baseName + "_instFlux"), flux, rtol=0.1)

    def runGaap(self, forced, psfSigma, scalingFactors=(1.0, 1.05, 1.1, 1.15, 1.2, 1.5, 2.0)):
        self.check(psfSigma=psfSigma, forced=forced, scalingFactors=scalingFactors)

    @lsst.utils.tests.methodParameters(psfSigma=(1.7, 0.95, 1.3,))
    def testGaapPluginUnforced(self, psfSigma):
        """Run GAaP as Single-frame measurement plugin.
        """
        self.runGaap(False, psfSigma)

    @lsst.utils.tests.methodParameters(psfSigma=(1.7, 0.95, 1.3,))
    def testGaapPluginForced(self, psfSigma):
        """Run GAaP as forced measurement plugin.
        """
        self.runGaap(True, psfSigma)

    def testFlags(self, sigmas=[2.5, 3.0, 4.0], scalingFactors=[1.15, 1.25, 1.4]):
        """Test that GAaP flags are set properly.

        Specifically, we test that

        1. for invalid combinations of config parameters, only the
        appropriate flags are set and not that the entire measurement itself is
        flagged.
        2. for sources close to the edge, the edge flags are set.

        Parameters
        ----------
        sigmas : `list` [`float`], optional
            The list of sigmas (in pixels) to construct the `GaapFluxConfig`.
        scalingFactors : `list` [`float`], optional
            The list of scaling factors to construct the `GaapFluxConfig`.

        Raises
        -----
        InvalidParameterError
            Raised if none of the config parameters will fail a measurement.

        Notes
        -----
        Since the seeing in the test dataset is 2 pixels, at least one of the
        ``sigmas`` should be smaller than at least twice of one of the
        ``scalingFactors`` to avoid the InvalidParameterError exception being
        raised.
        """
        gaapConfig = lsst.meas.extensions.gaap.GaapFluxConfig(sigmas=sigmas, scalingFactors=scalingFactors)
        gaapConfig.scaleByFwhm = True

        # Make an instance of GAaP algorithm from a config
        algName = "ext_gaap_GaapFlux"
        algorithm, schema = self.makeAlgorithm(gaapConfig)
        # Make a noiseless exposure and measurements for reference
        exposure, catalog = self.dataset.realize(0.0, schema)
        record = catalog[0]
        algorithm.measure(record, exposure)
        seeing = exposure.getPsf().getSigma()
        # Measurement must fail (i.e., flag_bigpsf must be set) if
        # sigma < scalingFactor * seeing
        # Ensure that there is at least one combination of parameters that fail
        if not(min(gaapConfig.sigmas) < seeing*max(gaapConfig.scalingFactors)):
            raise InvalidParameterError("The config parameters do not trigger a measurement failure. "
                                        "Consider including lower values in ``sigmas`` and/or larger values "
                                        "for ``scalingFactors``")
        # Ensure that the measurement is not a complete failure
        self.assertFalse(record[algName + "_flag"])
        self.assertFalse(record[algName + "_flag_edge"])
        # Ensure that flag_bigpsf is set if sigma < scalingFactor * seeing
        for sF, sigma in itertools.product(gaapConfig.scalingFactors, gaapConfig.sigmas):
            targetSigma = sF*seeing
            baseName = gaapConfig._getGaapResultName(sF, sigma, algName)
            if targetSigma >= sigma:
                self.assertTrue(record[baseName+"_flag_bigpsf"])
            else:
                self.assertFalse(record[baseName+"_flag_bigpsf"])

        # Ensure that the edge flag is set for the source at the corner.
        record = catalog[2]
        algorithm.measure(record, exposure)
        self.assertTrue(record[algName + "_flag_edge"])
        self.assertFalse(record[algName + "_flag"])

    def getFluxErrScaling(self, kernel, aperShape):
        """Returns the value by which the standard error has to be scaled due
        to noise correlations.

        This is an alternative implementation to the `_getFluxErrScaling`
        method of `BaseGaapFluxPlugin`, but is less efficient.

        Parameters
        ----------
        `kernel` : `~lsst.afw.math.Kernel`
            The PSF-Gaussianization kernel.

        Returns
        -------
        fluxErrScaling : `float`
            The factor by which the standard error on GAaP flux must be scaled.
        """
        kim = afwImage.ImageD(kernel.getDimensions())
        kernel.computeImage(kim, False)
        weight = galsim.Image(np.zeros_like(kim.array))
        aperSigma = aperShape.getDeterminantRadius()
        gauss = galsim.Gaussian(sigma=aperSigma, flux=2*np.pi*aperSigma**2)
        weight = gauss.drawImage(image=weight, scale=1.0, method='no_pixel')
        kwarr = scipy.signal.convolve2d(weight.array, kim.array, boundary='fill')
        fluxErrScaling = np.sqrt(np.sum(kwarr*kwarr))
        fluxErrScaling /= np.sqrt(np.pi*aperSigma**2)
        return fluxErrScaling

    def testCorrelatedNoiseError(self, sigmas=[3.0, 4.0], scalingFactors=[1.15, 1.2, 1.25, 1.3, 1.4]):
        """Test the scaling to standard error due to correlated noise.

        The uncertainty estimate on GAaP fluxes is scaled by an amount
        determined by the auto-correlation function of the PSF-matching kernel;
        see Eqs. A11 & A17 of Kuijken et al. (2015). This test ensures that the
        calculation of the scaling factors matches the analytical expression
        when the PSF-matching kernel is a Gaussian.

        Parameters
        ----------
        sigmas : `list` [`float`], optional
            A list of effective Gaussian aperture sizes.
        scalingFactors : `list` [`float`], optional
            A list of factors by which the PSF size must be scaled.
        """
        # Create an image of an extended source
        gaapConfig = lsst.meas.extensions.gaap.GaapFluxConfig(sigmas=sigmas, scalingFactors=scalingFactors)
        gaapConfig.scaleByFwhm = True

        algorithm, schema = self.makeAlgorithm(gaapConfig)
        exposure, catalog = self.dataset.realize(0.0, schema)
        record = catalog[0]
        center = self.center
        seeing = exposure.getPsf().computeShape(center).getDeterminantRadius()
        for sF in gaapConfig.scalingFactors:
            targetSigma = sF*seeing
            modelPsf = afwDetection.GaussianPsf(algorithm.config.modelPsfDimension,
                                                algorithm.config.modelPsfDimension,
                                                targetSigma)
            result, _ = algorithm._generic._convolve(exposure, modelPsf, record)
            kernel = result.psfMatchingKernel
            kernelAcf = algorithm._generic._computeKernelAcf(kernel)
            for sigma in gaapConfig.sigmas:
                aperSigma2 = sigma**2 - targetSigma**2
                aperShape = afwGeom.Quadrupole(aperSigma2, aperSigma2, 0.0)
                fluxErrScaling1 = algorithm._generic._getFluxErrScaling(kernelAcf, aperShape)
                fluxErrScaling2 = self.getFluxErrScaling(kernel, aperShape)

                # The PSF matching kernel is a Gaussian of sigma^2 = (f^2-1)s^2
                # where f is the scalingFactor and s is the original seeing.
                # The integral of ACF of the kernel times the elliptical
                # Gaussian described by aperShape is given below.
                analyticalValue = ((sigma**2 - (targetSigma)**2)/(sigma**2-seeing**2))**0.5
                self.assertFloatsAlmostEqual(fluxErrScaling1, analyticalValue, rtol=1e-4)
                self.assertFloatsAlmostEqual(fluxErrScaling1, fluxErrScaling2, rtol=1e-4)

    @lsst.utils.tests.methodParameters(noise=(0.001, 0.01, 0.1))
    def testMonteCarlo(self, noise, recordId=1, sigmas=[3.0, 4.0], scalingFactors=[1.1, 1.15, 1.2, 1.3, 1.4]):
        """Test GAaP flux uncertainties.

        This test should demonstate that the estimated flux uncertainties agree
        with those from Monte Carlo simulations.

        Parameters
        ----------
        noise : `float`
            The RMS value of the Gaussian noise field divided by the total flux
            of the source.
        recordId : `int`, optional
            The source Id in the test dataset to measure.
        sigmas : `list` [`float`], optional
            The list of sigmas (in pixels) to construct the `GaapFluxConfig`.
        scalingFactors : `list` [`float`], optional
            The list of scaling factors to construct the `GaapFluxConfig`.
        """
        gaapConfig = lsst.meas.extensions.gaap.GaapFluxConfig(sigmas=sigmas, scalingFactors=scalingFactors)
        gaapConfig.scaleByFwhm = True

        algorithm, schema = self.makeAlgorithm(gaapConfig)
        # Make a noiseless exposure and keep measurement record for reference
        exposure, catalog = self.dataset.realize(0.0, schema)
        recordNoiseless = catalog[recordId]
        totalFlux = recordNoiseless["truth_instFlux"]
        algorithm.measure(recordNoiseless, exposure)

        nSamples = 1024
        catalog = afwTable.SourceCatalog(schema)
        for repeat in range(nSamples):
            exposure, cat = self.dataset.realize(noise*totalFlux, schema, randomSeed=repeat)
            record = cat[recordId]
            algorithm.measure(record, exposure)
            catalog.append(record)

        catalog = catalog.copy(deep=True)
        for baseName in gaapConfig.getAllGaapResultNames(name="ext_gaap_GaapFlux"):
            instFluxKey = schema.join(baseName, "instFlux")
            instFluxErrKey = schema.join(baseName, "instFluxErr")
            instFluxMean = catalog[instFluxKey].mean()
            instFluxErrMean = catalog[instFluxErrKey].mean()
            instFluxStdDev = catalog[instFluxKey].std()

            # GAaP fluxes are not meant to be total fluxes.
            # We compare the mean of the noisy measurements to its
            # corresponding noiseless measurement instead of the true value
            instFlux = recordNoiseless[instFluxKey]
            self.assertFloatsAlmostEqual(instFluxErrMean, instFluxStdDev, rtol=0.02)
            self.assertLess(abs(instFluxMean - instFlux), 2.0*instFluxErrMean/nSamples**0.5)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module, backend="virtualDevice"):
    lsst.utils.tests.init()
    try:
        afwDisplay.setDefaultBackend(backend)
    except Exception:
        print("Unable to configure display backend: %s" % backend)


if __name__ == "__main__":
    import sys

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--backend', type=str, default="virtualDevice",
                        help="The backend to use, e.g. 'ds9'. Be sure to 'setup display_<backend>'")
    args = parser.parse_args()

    setup_module(sys.modules[__name__], backend=args.backend)
    unittest.main()
