#
# LSST Data Management System
# Copyright 2017 LSST/AURA.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
#
from __future__ import absolute_import, division, print_function

import math
import unittest
import lsst.utils.tests
import lsst.daf.base as dafBase
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.geom
import lsst.geom as geom # HACK ALERT
import lsst.afw.geom.ellipses as afwEll
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.meas.base as measBase
import lsst.meas.algorithms as measAlg
from lsst.ip.diffim import modelPsfMatch
import numpy as np
import sys
sys.path.append('/home/kannawad/repo/meas_extensions_gaap/')
sys.path.append('/home/kannawad/repo/meas_extensions_gaap/python/')
sys.path.append('/home/kannawad/repo/meas_extensions_gaap/lib/')
import lsst.meas.extensions.gaap  # Load flux.convolved algorithm

import lsst.afw.display as afwDisplay

try:
    type(display)
except NameError:
    display = False
    frame = 1

SIGMA_TO_FWHM = 2.0*math.sqrt(2.0*math.log(2.0))

def makeExposure(bbox, scale, psfFwhm, flux):
    """Make a fake exposure
    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
        Bounding box for image.
    scale : `lsst.afw.geom.Angle`
        Pixel scale.
    psfFwhm : `float`
        PSF FWHM (arcseconds)
    flux : `float`
        PSF flux (ADU)
    Returns
    -------
    exposure : `lsst.afw.image.ExposureF`
        Fake exposure.
    center : `lsst.afw.geom.Point2D`
        Position of fake source.
    """
    image = afwImage.ImageF(bbox)
    image.set(0)
    center = geom.Box2D(bbox).getCenter()
    psfSigma = psfFwhm/SIGMA_TO_FWHM/scale.asArcseconds()
    psfWidth = 2*int(4.0*psfSigma) + 1
    psf = afwDetection.GaussianPsf(psfWidth, psfWidth, psfSigma)
    psfImage = psf.computeImage(center).convertF()
    psfFlux = psfImage.getArray().sum()
    psfImage *= flux/psfFlux

    subImage = afwImage.ImageF(image, psfImage.getBBox(afwImage.PARENT), afwImage.PARENT)
    subImage += psfImage

    exp = afwImage.makeExposure(afwImage.makeMaskedImage(image))
    exp.setPsf(psf)
    exp.getMaskedImage().getVariance().set(1.0)
    exp.getMaskedImage().getMask().set(0)

    cdMatrix = afwGeom.makeCdMatrix(scale=scale)
    exp.setWcs(afwGeom.makeSkyWcs(crpix=center,
                                  crval=geom.SpherePoint(0.0, 0.0, geom.degrees),
                                  cdMatrix=cdMatrix))
    return exp, center

def makeGaussianizedExposure(bbox, scale, psfFwhm, flux, modelPsf=None):
    exposure, center = makeExposure(bbox, scale, psfFwhm, flux)
    pixToGrow = 0 # 2*max(self.psfMatch.kConfig.sizeCellX, self.psfMatch.kConfig.sizeCellY)
    bbox.grow(pixToGrow)

    #origPsf = exposure.getPsf(bbox.getCenter())
    origPsf = exposure.getPsf()

    maskedImage = exposure.getMaskedImage()
    subImage = maskedImage.Factory(maskedImage, bbox)
    subExposure = afwImage.ExposureF(subImage)
    subExposure.setPsf(origPsf)
    if modelPsf is None:
        modelPsf = measAlg.SingleGaussianPsf(width=64, height=64, sigma=SIGMA_TO_FWHM*0.6)
    result = modelPsfMatch.ModelPsfMatchTask().run(exposure=subExposure, referencePsfModel=modelPsf)
    convolved = result.psfMatchedExposure
    convolved.image.array[np.isnan(convolved.image.array)] = 0. ## HACK ALERT
    return convolved, center

class GaapFluxTestCase(lsst.utils.tests.TestCase):

    def check(self, psfFwhm=0.5, flux=1000., forced=False):
        return 0
        bbox = geom.Box2I(geom.Point2I(12345,6789), geom.Extent2I(200,300))

        scale = 0.1*geom.arcseconds

        TaskClass = measBase.ForcedMeasurementTask if forced else measBase.SingleFrameMeasurementTask

        exposure, center = makeExposure(bbox, scale, psfFwhm, flux)

        measConfig = TaskClass.ConfigClass()
        algName = "ext_gaap_GaapFlux"

        measConfig.plugins.names.add(algName)

        if forced:
            measConfig.copyColumns = {"id": "objectId", "parent": "parentObjectId"}

        values = [ii/scale.asArcseconds() for ii in (0.6, 0.8, 1.0, 1.2)]
        algConfig = measConfig.plugins[algName]
        algConfig.seeing = values
#        algConfig.aperture.radii = values

        if forced:
            offset = geom.Extent2D(-12.3, 45.6)
            refWcs = exposure.getWcs().copyAtShiftedPixelOrigin(offset)
            refSchema = afwTable.SourceTable.makeMinimalSchema()
            centroidKey = afwTable.Point2DKey.addFields(refSchema, "my_centroid", doc="centroid", unit="pixel")
            shapeKey = afwTable.QuadrupoleKey.addFields(refSchema, "my_shape", "shape")

            refSchema.getAliasMap().set("slot_Centroid", "my_centroid")
            refSchema.getAliasMap().set("slot_Shape", "my_shape")
            refSchema.addField("my_centroid_flag", type="Flag", doc="centroid flag")
            refSchema.addField("my_shape_flag", type="Flag", doc="shape flag")
            refCat = afwTable.SourceCatalog(refSchema)
            refSource = refCat.addNew()
            refSource.set(centroidKey, center + offset)

            refSource.setCoord(refWcs.pixelToSky(refSource.get(centroidKey)))
            taskInitArgs = (refSchema,)
            taskRunArgs = (refCat, refWcs)
        else:
            taskInitArgs = (afwTable.SourceTable.makeMinimalSchema(),)
            taskRunArgs = ()

        # Activate undeblended measurement with the same configuration
        measConfig.undeblended.names.add(algName)
        measConfig.undeblended[algName] = measConfig.plugins[algName]

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

        disp = afwDisplay.Display(frame)
        disp.mtv(exposure)
        disp.dot("x", *center, origin=afwImage.PARENT, title="psfFwhm=%f" % (psfFwhm,))

        self.assertFalse(source.get(algName + "_flag"))  # algorithm succeeded
        originalSeeing = psfFwhm/scale.asArcseconds()

        for prefix in ("", "undeblended_"):
            if not forced:
                targetSeeing = 0.6
                gaapName = algConfig.getGaapResultName(targetSeeing)

                #print(source.get(prefix + gaapName + "_flux"), source.get(prefix + gaapName + "_fluxErr"), source.get(prefix + gaapName + "_flag"))

    def testGaapFlux(self):
        return 0
        for forced in (False, True):
            for psfFwhm in (0.5, 0.9, 1.3):
                self.check(psfFwhm=psfFwhm, forced=forced)

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