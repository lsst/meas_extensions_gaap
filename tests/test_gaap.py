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
import lsst.afw.detection as afwDetection
import lsst.afw.display as afwDisplay
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


class GaapFluxTestCase(lsst.meas.base.tests.AlgorithmTestCase, lsst.utils.tests.TestCase):
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
            gaapConfig = lsst.meas.extensions.gaap.SingleFrameGaapFluxConfig()
        gaapPlugin = lsst.meas.extensions.gaap.SingleFrameGaapFluxPlugin(gaapConfig,
                                                                         "ext_gaap_GaapFlux",
                                                                         schema, None)
        if gaapConfig.doOptimalPhotometry:
            afwTable.QuadrupoleKey.addFields(schema, "psfShape", "PSF shape")
            schema.getAliasMap().set("slot_PsfShape", "psfShape")
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

        # Remove sky coordinate plugin because we don't have the columns
        # in the tests.
        if "base_SkyCoord" in measConfig.plugins.names:
            measConfig.plugins.names.remove("base_SkyCoord")

        measConfig.plugins.names.add(algName)

        if forced:
            measConfig.copyColumns = {"id": "objectId", "parent": "parentObjectId"}

        algConfig = measConfig.plugins[algName]
        algConfig.scalingFactors = scalingFactors
        algConfig.scaleByFwhm = True
        algConfig.doPsfPhotometry = True
        # Do not turn on optimal photometry; not robust for a point-source.
        algConfig.doOptimalPhotometry = False

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

        # For scalingFactor > 1, check if the measured value is close to truth.
        for baseName in algConfig.getAllGaapResultNames(algName):
            if "_1_0x_" not in baseName:
                rtol = 0.1 if "PsfFlux" not in baseName else 0.2
                self.assertFloatsAlmostEqual(source.get(baseName + "_instFlux"), flux, rtol=rtol)

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

    def testFail(self, scalingFactors=[100.], sigmas=[500.]):
        """Test that the fail method sets the flags correctly.

        Set config parameters that are guaranteed to raise exceptions,
        and check that they are handled properly by the `fail` method and that
        expected log messages are generated.
        For failure modes not handled by the `fail` method, we test them
        in the ``testFlags`` method.
        """
        algName = "ext_gaap_GaapFlux"
        dependencies = ("base_SdssShape",)
        config = self.makeSingleFrameMeasurementConfig(algName, dependencies=dependencies)
        gaapConfig = config.plugins[algName]
        gaapConfig.scalingFactors = scalingFactors
        gaapConfig.sigmas = sigmas
        gaapConfig.doPsfPhotometry = True
        gaapConfig.doOptimalPhotometry = True

        gaapConfig.scaleByFwhm = True
        self.assertTrue(gaapConfig.scaleByFwhm)  # Test the getter method.

        algMetadata = lsst.daf.base.PropertyList()
        sfmTask = self.makeSingleFrameMeasurementTask(algName, dependencies=dependencies, config=config,
                                                      algMetadata=algMetadata)
        exposure, catalog = self.dataset.realize(0.0, sfmTask.schema)
        self.recordPsfShape(catalog)

        # Expected debug messages in the logs when running `sfmTask`.
        errorMessage = [("Failed to solve for PSF matching kernel in GAaP for (100.000000, 670.000000): "
                         "Problematic scaling factors = 100.0 "
                         "Errors: RuntimeError('Unable to determine kernel sum; 0 candidates')"),
                        ("MeasurementError in ext_gaap_GaapFlux.measure on record 1: "
                         "Failed to solve for PSF matching kernel"),
                        ("Failed to solve for PSF matching kernel in GAaP for (100.000000, 870.000000): "
                         "Problematic scaling factors = 100.0 "
                         "Errors: RuntimeError('Unable to determine kernel sum; 0 candidates')"),
                        ("MeasurementError in ext_gaap_GaapFlux.measure on record 2: "
                         "Failed to solve for PSF matching kernel"),
                        ("Failed to solve for PSF matching kernel in GAaP for (-10.000000, -20.000000): "
                         "Problematic scaling factors = 100.0 "
                         "Errors: RuntimeError('Unable to determine kernel sum; 0 candidates')"),
                        ("MeasurementError in ext_gaap_GaapFlux.measure on record 3: "
                         "Failed to solve for PSF matching kernel")]

        testCatalog = catalog.copy(deep=True)
        plugin_logger_name = sfmTask.log.getChild(algName).name
        self.assertEqual(plugin_logger_name, "lsst.measurement.ext_gaap_GaapFlux")
        with self.assertLogs(plugin_logger_name, "DEBUG") as cm:
            sfmTask.run(testCatalog, exposure)
        self.assertEqual([record.message for record in cm.records], errorMessage)

        self._checkAllFlags(
            testCatalog,
            algName,
            scalingFactors,
            sigmas,
            gaapConfig,
            specificFlag="flag_gaussianization",
        )

        # Trigger a "not (psfSigma > 0) error":
        exposureJunkPsf = exposure.clone()
        testCatalog = catalog.copy(deep=True)
        junkPsf = afwDetection.GaussianPsf(1, 1, 0)
        exposureJunkPsf.setPsf(junkPsf)
        sfmTask.run(testCatalog, exposureJunkPsf)

        self._checkAllFlags(
            testCatalog,
            algName,
            scalingFactors,
            sigmas,
            gaapConfig,
            specificFlag="flag_gaussianization",
        )

        # Trigger a NoPixelError.
        testCatalog = catalog.copy(deep=True)
        testCatalog[0].setFootprint(afwDetection.Footprint())
        with self.assertLogs(plugin_logger_name, "DEBUG") as cm:
            sfmTask.run(testCatalog, exposure)

        self.assertEqual(
            cm.records[0].message,
            "MeasurementError in ext_gaap_GaapFlux.measure on record 1: No good pixels in footprint",
        )
        self.assertEqual(testCatalog[f"{algName}_flag_no_pixel"][0], True)
        self.assertEqual(testCatalog[f"{algName}_flag"][0], True)

        self._checkAllFlags(testCatalog[0: 1], algName, scalingFactors, sigmas, gaapConfig, allFailFlag=True)

        # Try and "fail" with no PSF.
        # Since fatal exceptions are not caught by the measurement framework,
        # use a context manager and catch it here.
        exposure.setPsf(None)
        with self.assertRaises(lsst.meas.base.FatalAlgorithmError):
            sfmTask.run(catalog, exposure)

    def _checkAllFlags(
        self,
        catalog,
        algName,
        scalingFactors,
        sigmas,
        gaapConfig,
        specificFlag=None,
        allFailFlag=False
    ):
        for record in catalog:
            self.assertEqual(record[algName + "_flag"], allFailFlag)
            for scalingFactor in scalingFactors:
                if specificFlag is not None:
                    flagName = gaapConfig._getGaapResultName(scalingFactor, specificFlag, algName)
                    self.assertTrue(record[flagName])
                for sigma in sigmas + ["Optimal"]:
                    baseName = gaapConfig._getGaapResultName(scalingFactor, sigma, algName)
                    self.assertTrue(record[baseName + "_flag"])
                    self.assertFalse(record[baseName + "_flag_bigPsf"])
                baseName = gaapConfig._getGaapResultName(scalingFactor, "PsfFlux", algName)
                self.assertTrue(record[baseName + "_flag"])

    def testFlags(self, sigmas=[0.4, 0.5, 0.7], scalingFactors=[1.15, 1.25, 1.4, 100.]):
        """Test that GAaP flags are set properly.

        Specifically, we test that

        1. for invalid combinations of config parameters, only the
        appropriate flags are set and not that the entire measurement itself is
        flagged.
        2. for sources close to the edge, the edge flags are set.

        Parameters
        ----------
        sigmas : `list` [`float`], optional
            The list of sigmas (in arcseconds) to construct the
            `SingleFrameGaapFluxConfig`.
        scalingFactors : `list` [`float`], optional
            The list of scaling factors to construct the
            `SingleFrameGaapFluxConfig`.

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
        gaapConfig = lsst.meas.extensions.gaap.SingleFrameGaapFluxConfig(sigmas=sigmas,
                                                                         scalingFactors=scalingFactors)
        gaapConfig.scaleByFwhm = True
        gaapConfig.doOptimalPhotometry = True

        # Make an instance of GAaP algorithm from a config
        algName = "ext_gaap_GaapFlux"
        algorithm, schema = self.makeAlgorithm(gaapConfig)
        # Make a noiseless exposure and measurements for reference
        exposure, catalog = self.dataset.realize(0.0, schema)
        # Record the PSF shapes if optimal photometry is performed.
        if gaapConfig.doOptimalPhotometry:
            self.recordPsfShape(catalog)

        record = catalog[0]
        algorithm.measure(record, exposure)
        seeing = exposure.getPsf().getSigma()
        pixelScale = exposure.getWcs().getPixelScale(exposure.getBBox().getCenter()).asArcseconds()
        # Measurement must fail (i.e., flag_bigPsf and flag must be set) if
        # sigma < scalingFactor * seeing
        # Ensure that there is at least one combination of parameters that fail
        if not (min(gaapConfig.sigmas)/pixelScale < seeing*max(gaapConfig.scalingFactors)):
            raise InvalidParameterError("The config parameters do not trigger a measurement failure. "
                                        "Consider including lower values in ``sigmas`` and/or larger values "
                                        "for ``scalingFactors``")
        # Ensure that the measurement is not a complete failure
        self.assertFalse(record[algName + "_flag"])
        self.assertFalse(record[algName + "_flag_edge"])
        # Ensure that flag_bigPsf is set if sigma < scalingFactor * seeing
        for scalingFactor, sigma in itertools.product(gaapConfig.scalingFactors, gaapConfig.sigmas):
            targetSigma = scalingFactor*seeing
            baseName = gaapConfig._getGaapResultName(scalingFactor, sigma, algName)
            # Give some leeway for the edge case and compare against a small
            # negative number instead of zero.
            if targetSigma*pixelScale - sigma >= -2e-7:
                self.assertTrue(record[baseName+"_flag_bigPsf"],
                                msg=f"bigPsf flag not set for {scalingFactor=} and {sigma=}",
                                )
                self.assertTrue(record[baseName+"_flag"],
                                msg=f"Flag not set for {scalingFactor=} and {sigma=}",
                                )
            else:
                self.assertFalse(record[baseName+"_flag_bigPsf"],
                                 msg=f"bigPsf flag set for {scalingFactor=} and {sigma=}",
                                 )
                self.assertFalse(record[baseName+"_flag"],
                                 msg=f"Flag set for {scalingFactor=} and {sigma=}",
                                 )

        # Ensure that flag_bigPsf is set if OptimalShape is not large enough.
        if gaapConfig.doOptimalPhotometry:
            aperShape = afwTable.QuadrupoleKey(schema[schema.join(algName, "OptimalShape")]).get(record)
            for scalingFactor in gaapConfig.scalingFactors:
                targetSigma = scalingFactor*seeing
                baseName = gaapConfig._getGaapResultName(scalingFactor, "Optimal", algName)
                try:
                    afwGeom.Quadrupole(aperShape.getParameterVector()-[targetSigma**2, targetSigma**2, 0.0],
                                       normalize=True)
                    self.assertFalse(record[baseName + "_flag_bigPsf"])
                except InvalidParameterError:
                    self.assertTrue(record[baseName + "_flag_bigPsf"])

        # Set an empty footprint and check that no_pixels flag is set.
        record = catalog[1]
        record.setFootprint(afwDetection.Footprint())
        with self.assertRaises(lsst.meas.extensions.gaap._gaap.NoPixelError):
            algorithm.measure(record, exposure)
        self.assertTrue(record[algName + "_flag"])
        self.assertTrue(record[algName + "_flag_no_pixel"])

        # Ensure that the edge flag is set for the source at the corner.
        record = catalog[2]
        algorithm.measure(record, exposure)
        self.assertTrue(record[algName + "_flag_edge"])
        self.assertFalse(record[algName + "_flag"])

    def recordPsfShape(self, catalog) -> None:
        """Record PSF shapes under the appropriate fields in ``catalog``.

        This method must be called after the dataset is realized and a catalog
        is returned by the `realize` method. It assumes that the schema is
        non-minimal and has "psfShape_xx", "psfShape_yy" and "psfShape_xy"
        fields setup

        Parameters
        ----------
        catalog : `~lsst.afw.table.SourceCatalog`
            A source catalog containing records of the simulated sources.
        """
        psfShapeKey = afwTable.QuadrupoleKey(catalog.schema["slot_PsfShape"])
        for record in catalog:
            record.set(psfShapeKey, self.dataset.psfShape)

    @staticmethod
    def invertQuadrupole(shape: afwGeom.Quadrupole) -> afwGeom.Quadrupole:
        """Compute the Quadrupole object corresponding to the inverse matrix.

        If M = [[Q.getIxx(), Q.getIxy()],
                [Q.getIxy(), Q.getIyy()]]

        for the input quadrupole Q, the returned quadrupole R corresponds to

        M^{-1} = [[R.getIxx(), R.getIxy()],
                  [R.getIxy(), R.getIyy()]].
        """
        invShape = afwGeom.Quadrupole(shape.getIyy(), shape.getIxx(), -shape.getIxy())
        invShape.scale(1./shape.getDeterminantRadius()**2)
        return invShape

    @lsst.utils.tests.methodParameters(gaussianizationMethod=("auto", "overlap-add", "direct", "fft"))
    def testGalaxyPhotometry(self, gaussianizationMethod):
        """Test GAaP fluxes for extended sources.

        Create and run a SingleFrameMeasurementTask with GAaP plugin and reuse
        its outputs as reference for ForcedGaapFluxPlugin. In both cases,
        the measured flux is compared with the analytical expectation.

        For a Gaussian source with intrinsic shape S and intrinsic aperture W,
        the GAaP flux is defined as (Eq. A16 of Kuijken et al. 2015)
        :math:`\\frac{F}{2\\pi\\det(S)}\\int\\mathrm{d}x\\exp(-x^T(S^{-1}+W^{-1})x/2)`
        :math:`F\\frac{\\det(S^{-1})}{\\det(S^{-1}+W^{-1})}`
        """
        algName = "ext_gaap_GaapFlux"
        dependencies = ("base_SdssShape",)
        sfmConfig = self.makeSingleFrameMeasurementConfig(algName, dependencies=dependencies)
        forcedConfig = self.makeForcedMeasurementConfig(algName, dependencies=dependencies)
        # Turn on optimal photometry explicitly
        sfmConfig.plugins[algName].doOptimalPhotometry = True
        forcedConfig.plugins[algName].doOptimalPhotometry = True
        sfmConfig.plugins[algName].gaussianizationMethod = gaussianizationMethod
        forcedConfig.plugins[algName].gaussianizationMethod = gaussianizationMethod

        algMetadata = lsst.daf.base.PropertyList()
        sfmTask = self.makeSingleFrameMeasurementTask(config=sfmConfig, algMetadata=algMetadata)
        forcedTask = self.makeForcedMeasurementTask(config=forcedConfig, algMetadata=algMetadata,
                                                    refSchema=sfmTask.schema)

        refExposure, refCatalog = self.dataset.realize(0.0, sfmTask.schema)
        self.recordPsfShape(refCatalog)
        sfmTask.run(refCatalog, refExposure)

        # Check if the measured values match the expectations from
        # analytical Gaussian integrals
        recordId = 1  # Elliptical Gaussian galaxy
        refRecord = refCatalog[recordId]
        refWcs = self.dataset.exposure.getWcs()
        schema = refRecord.schema
        trueFlux = refRecord["truth_instFlux"]
        intrinsicShapeVector = afwTable.QuadrupoleKey(schema["truth"]).get(refRecord).getParameterVector() \
            - afwTable.QuadrupoleKey(schema["slot_PsfShape"]).get(refRecord).getParameterVector()
        intrinsicShape = afwGeom.Quadrupole(intrinsicShapeVector)
        invIntrinsicShape = self.invertQuadrupole(intrinsicShape)
        # Assert that the measured fluxes agree with analytical expectations.
        for sigma in sfmTask.config.plugins[algName]._sigmas:
            if sigma == "Optimal":
                aperShape = afwTable.QuadrupoleKey(schema[f"{algName}_OptimalShape"]).get(refRecord)
            else:
                aperShape = afwGeom.Quadrupole(sigma**2, sigma**2, 0.0)
                aperShape.transformInPlace(refWcs.linearizeSkyToPixel(refRecord.getCentroid(),
                                                                      geom.arcseconds).getLinear())

            invAperShape = self.invertQuadrupole(aperShape)
            analyticalFlux = trueFlux*(invIntrinsicShape.getDeterminantRadius()
                                       / invIntrinsicShape.convolve(invAperShape).getDeterminantRadius())**2
            for scalingFactor in sfmTask.config.plugins[algName].scalingFactors:
                baseName = sfmTask.plugins[algName].ConfigClass._getGaapResultName(scalingFactor,
                                                                                   sigma, algName)
                instFlux = refRecord.get(f"{baseName}_instFlux")
                self.assertFloatsAlmostEqual(instFlux, analyticalFlux, rtol=5e-3)

        measWcs = self.dataset.makePerturbedWcs(refWcs, randomSeed=15)
        measDataset = self.dataset.transform(measWcs)
        measExposure, truthCatalog = measDataset.realize(0.0, schema)
        measCatalog = forcedTask.generateMeasCat(measExposure, refCatalog, refWcs)
        forcedTask.attachTransformedFootprints(measCatalog, refCatalog, measExposure, refWcs)
        forcedTask.run(measCatalog, measExposure, refCatalog, refWcs)

        fullTransform = afwGeom.makeWcsPairTransform(refWcs, measWcs)
        localTransform = afwGeom.linearizeTransform(fullTransform, refRecord.getCentroid()).getLinear()
        intrinsicShape.transformInPlace(localTransform)
        invIntrinsicShape = self.invertQuadrupole(intrinsicShape)
        measRecord = measCatalog[recordId]

        # Since measCatalog and refCatalog differ only by WCS, the GAaP flux
        # measured through consistent apertures must agree with each other.
        for sigma in forcedTask.config.plugins[algName]._sigmas:
            if sigma == "Optimal":
                aperShape = afwTable.QuadrupoleKey(measRecord.schema[f"{algName}_"
                                                                     "OptimalShape"]).get(measRecord)
            else:
                aperShape = afwGeom.Quadrupole(sigma**2, sigma**2, 0.0)
                aperShape.transformInPlace(measWcs.linearizeSkyToPixel(measRecord.getCentroid(),
                                                                       geom.arcseconds).getLinear())

            invAperShape = self.invertQuadrupole(aperShape)
            analyticalFlux = trueFlux*(invIntrinsicShape.getDeterminantRadius()
                                       / invIntrinsicShape.convolve(invAperShape).getDeterminantRadius())**2
            for scalingFactor in forcedTask.config.plugins[algName].scalingFactors:
                baseName = forcedTask.plugins[algName].ConfigClass._getGaapResultName(scalingFactor,
                                                                                      sigma, algName)
                instFlux = measRecord.get(f"{baseName}_instFlux")
                # The measurement in the measRecord must be consistent with
                # the same in the refRecord in addition to analyticalFlux.
                self.assertFloatsAlmostEqual(instFlux, refRecord.get(f"{baseName}_instFlux"), rtol=5e-3)
                self.assertFloatsAlmostEqual(instFlux, analyticalFlux, rtol=5e-3)

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
        trace = aperShape.getIxx() + aperShape.getIyy()
        distortion = galsim.Shear(e1=(aperShape.getIxx()-aperShape.getIyy())/trace,
                                  e2=2*aperShape.getIxy()/trace)
        gauss = galsim.Gaussian(sigma=aperSigma, flux=2*np.pi*aperSigma**2).shear(distortion)
        weight = gauss.drawImage(image=weight, scale=1.0, method='no_pixel')
        kwarr = scipy.signal.convolve2d(weight.array, kim.array, boundary='fill')
        fluxErrScaling = np.sqrt(np.sum(kwarr*kwarr))
        fluxErrScaling /= np.sqrt(np.pi*aperSigma**2)
        return fluxErrScaling

    def testCorrelatedNoiseError(self, sigmas=[0.6, 0.8], scalingFactors=[1.15, 1.2, 1.25, 1.3, 1.4]):
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

        Notes
        -----
        This unit test tests internal states of the plugin for accuracy and is
        specific to the implementation. It uses private variables as a result
        and intentionally breaks encapsulation.
        """
        gaapConfig = lsst.meas.extensions.gaap.SingleFrameGaapFluxConfig(sigmas=sigmas,
                                                                         scalingFactors=scalingFactors)
        gaapConfig.scaleByFwhm = True

        algorithm, schema = self.makeAlgorithm(gaapConfig)
        exposure, catalog = self.dataset.realize(0.0, schema)
        wcs = exposure.getWcs()
        record = catalog[0]
        center = self.center
        seeing = exposure.getPsf().computeShape(center).getDeterminantRadius()
        for scalingFactor in gaapConfig.scalingFactors:
            targetSigma = scalingFactor*seeing
            modelPsf = afwDetection.GaussianPsf(algorithm.config._modelPsfDimension,
                                                algorithm.config._modelPsfDimension,
                                                targetSigma)
            result = algorithm._gaussianize(exposure, modelPsf, record)
            kernel = result.psfMatchingKernel
            kernelAcf = algorithm._computeKernelAcf(kernel)
            for sigma in gaapConfig.sigmas:
                intrinsicShape = afwGeom.Quadrupole(sigma**2, sigma**2, 0.0)
                intrinsicShape.transformInPlace(wcs.linearizeSkyToPixel(center, geom.arcseconds).getLinear())
                aperShape = afwGeom.Quadrupole(intrinsicShape.getParameterVector()
                                               - [targetSigma**2, targetSigma**2, 0.0])
                fluxErrScaling1 = algorithm._getFluxErrScaling(kernelAcf, aperShape)
                fluxErrScaling2 = self.getFluxErrScaling(kernel, aperShape)

                # The PSF matching kernel is a Gaussian of sigma^2 = (f^2-1)s^2
                # where f is the scalingFactor and s is the original seeing.
                # The integral of ACF of the kernel times the elliptical
                # Gaussian described by aperShape is given below.
                sigma /= wcs.getPixelScale(exposure.getBBox().getCenter()).asArcseconds()
                analyticalValue = ((sigma**2 - (targetSigma)**2)/(sigma**2-seeing**2))**0.5
                self.assertFloatsAlmostEqual(fluxErrScaling1, analyticalValue, rtol=1e-4)
                self.assertFloatsAlmostEqual(fluxErrScaling1, fluxErrScaling2, rtol=1e-4)

            # Try with an elliptical aperture. This is a proxy for
            # optimal aperture, since we do not actually measure anything.
            aperShape = afwGeom.Quadrupole(8, 6, 3)
            fluxErrScaling1 = algorithm._getFluxErrScaling(kernelAcf, aperShape)
            fluxErrScaling2 = self.getFluxErrScaling(kernel, aperShape)
            self.assertFloatsAlmostEqual(fluxErrScaling1, fluxErrScaling2, rtol=1e-4)

    @lsst.utils.tests.methodParameters(noise=(0.001, 0.01, 0.1))
    def testMonteCarlo(self, noise, recordId=1, sigmas=[0.7, 1.0, 1.25],
                       scalingFactors=[1.1, 1.15, 1.2, 1.3, 1.4]):
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
        gaapConfig = lsst.meas.extensions.gaap.SingleFrameGaapFluxConfig(sigmas=sigmas,
                                                                         scalingFactors=scalingFactors)
        gaapConfig.scaleByFwhm = True
        gaapConfig.doPsfPhotometry = True
        gaapConfig.doOptimalPhotometry = True

        algorithm, schema = self.makeAlgorithm(gaapConfig)
        # Make a noiseless exposure and keep measurement record for reference
        exposure, catalog = self.dataset.realize(0.0, schema)
        if gaapConfig.doOptimalPhotometry:
            self.recordPsfShape(catalog)
        recordNoiseless = catalog[recordId]
        totalFlux = recordNoiseless["truth_instFlux"]
        algorithm.measure(recordNoiseless, exposure)

        nSamples = 1024
        catalog = afwTable.SourceCatalog(schema)
        for repeat in range(nSamples):
            exposure, cat = self.dataset.realize(noise*totalFlux, schema, randomSeed=repeat)
            if gaapConfig.doOptimalPhotometry:
                self.recordPsfShape(cat)
            record = cat[recordId]
            algorithm.measure(record, exposure)
            catalog.append(record)

        catalog = catalog.copy(deep=True)
        for baseName in gaapConfig.getAllGaapResultNames():
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
