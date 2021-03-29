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
import lsst.utils.tests
import lsst.daf.base as dafBase
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.geom as geom
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.meas.base as measBase
import lsst.afw.display as afwDisplay
import lsst.meas.extensions.gaap


try:
    type(display)
except NameError:
    display = False
    frame = 1


def makeGalaxyExposure(scale, psfSigma=0.9, flux=1000., galSigma=3.7):
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

    exposure.variance.set(1.0)
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
    def check(self, psfSigma=0.5, flux=1000., scalingFactors=[1.15], forced=False):
        """Check for non-negative values for GAaP instFlux and instFluxErr.
        """
        scale = 0.1*geom.arcseconds

        TaskClass = measBase.ForcedMeasurementTask if forced else measBase.SingleFrameMeasurementTask

        # Create an image of a tiny source
        exposure, center = makeGalaxyExposure(scale, psfSigma, flux, galSigma=0.001)

        measConfig = TaskClass.ConfigClass()
        algName = "ext_gaap_GaapFlux"

        measConfig.plugins.names.add(algName)

        if forced:
            measConfig.copyColumns = {"id": "objectId", "parent": "parentObjectId"}

        algConfig = measConfig.plugins[algName]
        algConfig.scalingFactors = scalingFactors

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
        ss = afwDetection.FootprintSet(exposure.getMaskedImage(), afwDetection.Threshold(0.1))
        fp = ss.getFootprints()[0]
        source.setFootprint(fp)

        task.run(measCat, exposure, *taskRunArgs)

        if display:
            disp = afwDisplay.Display(frame)
            disp.mtv(exposure)
            disp.dot("x", *center, origin=afwImage.PARENT, title="psfSigma=%f" % (psfSigma,))

        self.assertFalse(source.get(algName + "_flag"))  # algorithm succeeded

        # We will check the accuracy of the measured flux in DM-29430.
        # We simply check now if it produces a positive number (non-nan)
        for sF in algConfig.scalingFactors:
            for sigma in algConfig.sigmas:
                baseName = algConfig.getGaapResultName(sF, sigma, algName)
                self.assertTrue((source.get(baseName + "_instFlux") >= 0))
                self.assertTrue((source.get(baseName + "_instFluxErr") >= 0))

    def runGaap(self, forced, scalingFactors=(1.0, 1.1, 1.15, 1.2, 1.5, 2.0), psfSigmas=(1.7, 0.95, 1.3,)):
        for psfSigma in psfSigmas:
            self.check(psfSigma=psfSigma, forced=forced, scalingFactors=scalingFactors)

    def testGaapUnforced(self):
        self.runGaap(False)

    def testGaapForced(self):
        self.runGaap(True)


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
