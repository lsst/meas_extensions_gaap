#
# LSST Data Management System
# Copyright 2008-2020 AURA/LSST.
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

import math
from lsst.ip.diffim.modelPsfMatch import ModelPsfMatchConfig
from lsst.ip.diffim.psfMatch import PsfMatchConfigDF
import numpy as np
import time

from lsst.pex.config import Config, Field, ListField, ConfigField, ConfigChoiceField, makeConfigClass
from lsst.pipe.base import Struct
from lsst.meas.base.wrappers import WrappedSingleFramePlugin, WrappedForcedPlugin

import lsst.meas.base
import lsst.meas.algorithms as measAlg
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
from lsst.ip.diffim import modelPsfMatch#, ModelPsfMatchConfig
from lsst.afw.geom.skyWcs import makeWcsPairTransform
from lsst.meas.base.wrappers import WrappedSingleFramePlugin, WrappedForcedPlugin

# from .gaapFlux import GaapFluxAlgorithm, GaapFluxControl, GaapFluxTransform

__all__ = ("GaapFluxPlugin", "GaapFluxConfig", "ForcedGaapFluxPlugin", "ForcedGaapFluxConfig")

SIGMA_TO_FWHM = 2.0*math.sqrt(2.0*(math.log(2.0)))
PLUGIN_NAME = "ext_gaap_GaapFlux"

GaapFluxConfig = makeConfigClass(GaapFluxControl)

class GaapFluxData(Struct):
    def __init__(self, name, schema, seeing, config, metadata):
        aperture = lsst.meas.base.GaapFluxAlgorithm(config.aperture.makeControl(), name, schema)
        Struct.__init__(self, aperture=aperture)

class BaseGaapFluxConfig(ModelPsfMatchConfig):
    seeing = ListField(dtype=float, default=[3,5, 5.0, 6.5], doc="list of target seeings (FWHM, pixels)")

    aperture = ConfigField(dtype=GaapFluxConfig, doc="Gaussian photometry parameters")
    #psfMatchKernel = ConfigChoiceField(
    #                                   doc="kernel type",
    #                                   typemap=dict(
    #                                                AL=PsfMatchConfigAL,
    #                                               ),
    #                                   default="AL",
    #                                  )

    def getGaapResultName(*args):
        return "ext_gaap_GaapFlux"

    def setDefaults(self):
        ModelPsfMatchConfig.setDefaults(self)
        self.kernel.active.alardNGauss = 1
        self.kernel.active.alardDegGaussDeconv = 1
        self.kernel.active.alardDegGauss = [1]
        self.kernel.active.alardSigGauss = [1.0]


class BaseGaapFluxPlugin(lsst.meas.base.BaseMeasurementPlugin):
    @classmethod
    def getExecutionOrder(cls):
        return 300000.2 ## HACK ALERT

    def __init__(self, config, name, schema, metadata):
        """
        GAaP
        """
        lsst.meas.base.BaseMeasurementPlugin.__init__(self, config, name)
        self.seeingKey = schema.addField(name + "_seeing", type="F",
                                         doc="original seeing (Gaussian sigma) at position",
                                         units="pixel")
        self.centroidExtractor = lsst.meas.base.SafeCentroidExtractor(schema, name)

        flagDefs = lsst.meas.base.FlagDefinitionList()
        flagDefs.addFailureFlag("error in running ConvolvedFluxPlugin")
        self.flagHandler = lsst.meas.base.FlagHandler.addFields(schema, name, flagDefs)
        self.gaussianAperture = lsst.meas.base.GaapFluxAlgorithm(config.aperture.makeControl(), 'GaapFlux'+name, schema)

    def convolve(self, exposure, modelPsf, footprint, maxRadius):
        """ Convolve

        Parameters
        ----------
        modelPsf : :cpp.class: `lsst::meas::algorithms::KernelPsf`
            Target PSF to which to match.

        Returns
        -------
        convExp : `lsst.afw.image.Exposure`
            Sub-image containing the source, convolving to the target seeing
        """

        bbox = footprint.getBBox()
        pixToGrow = 8 # 2*max(self.psfMatch.kConfig.sizeCellX, self.psfMatch.kConfig.sizeCellY)
        bbox.grow(pixToGrow)

        #origPsf = exposure.getPsf(bbox.getCenter())
        origPsf = exposure.getPsf()

        maskedImage = exposure.getMaskedImage()
        subImage = maskedImage.Factory(maskedImage, bbox)
        subExposure = afwImage.ExposureF(subImage)
        subExposure.setPsf(origPsf)
        result = modelPsfMatch.ModelPsfMatchTask(config=self.config).run(exposure=subExposure, referencePsfModel=modelPsf)
        convolved = result.psfMatchedExposure
        convolved.image.array[np.isnan(convolved.image.array)] = 0. ## HACK ALERT
        return convolved
        ##return exposure HACK ALERT

    def measure(self, measRecord, exposure):
        return self.measureForced(measRecord, exposure, measRecord, None)

    def measureForced(self, measRecord, exposure, refRecord, refWcs):
        t1 = time.time()
        psf = exposure.getPsf()
        if psf is None:
            raise lsst.meas.base.MeasurementError("No PSF in exposure")

        refCenter = self.centroidExtractor(refRecord, self.flagHandler)

        if refWcs is not None:
            measWcs = exposure.getWcs()
            if measWcs is None:
                raise lsst.meas.base.MeasurementError("No WCS in exposure")
            fullTransform = makeWcsPairTransform(refWcs, measWcs)
            transform = lsst.afw.geom.linearizeTransform(fullTransform, refCenter)
        else:
            transform = lsst.geom.AffineTransform()

        center = refCenter if transform is None else transform(refCenter)
        seeing = psf.computeShape(center).getDeterminantRadius()
        measRecord.set(self.seeingKey, seeing)

        for ii, target in enumerate(self.config.seeing):
            modelPsf = measAlg.SingleGaussianPsf(width=64, height=64, sigma=SIGMA_TO_FWHM*target)
            # TO DO: convert seeing in FWHM to sigma
            # TO DO: Define the aperture shape here, using SafeShapeExtractor - seeing**2
            try:
                maxRadius = 64
                convolved = self.convolve(exposure, modelPsf, measRecord.getFootprint(), maxRadius)
            except RuntimeError:
                convolved = exposure

            #import pdb; pdb.set_trace()
            self.measureAperture(measRecord, convolved, self.gaussianAperture)
        t2 = time.time()
        print(t2-t1)

    def measureAperture(self, measRecord, exposure, aperturePhot):
        """Perform aperture photometry
        Parameters
        ----------
        measRecord : `lsst.afw.table.SourceRecord`
            Record for source to be measured.
        exposure : `lsst.afw.image.Exposure`
            Image to be measured.
        aperturePhot : `lsst.meas.base.GaapFluxAlgorithm`
            Measurement plugin that will do the measurement.
        """
        try:
            aperturePhot.measure(measRecord, exposure)
        except Exception:
            aperturePhot.fail(measRecord)

    def fail(self, measRecord, error=None):
        """ Record failure
        """
        self.flagHandler.handleFailure(measRecord)

    def getBaseNameForSeeing(self, seeing, name=PLUGIN_NAME):
        indices = [ii for ii, target in enumerate(self.seeing) if seeing==target]
        if len(indices) != 1:
            raise RuntimeError(f"Unable to uniquely identify index for seeing {seeing}: {indices}")
        return name + f"_{indices[0]}"

    def getGaapResultName(self, seeing, name=PLUGIN_NAME):
        return self.getBaseNameForSeeing(seeing, name=name)+"_gaap"


def wrapPlugin(Base, PluginClass=BaseGaapFluxPlugin, ConfigClass=BaseGaapFluxConfig,
                name=PLUGIN_NAME, factory=BaseGaapFluxPlugin):
    WrappedConfig = type("GAaPFlux"+ Base.ConfigClass.__name__, (Base.ConfigClass, ConfigClass), {})
    typeDict = dict(AlgClass=PluginClass, ConfigClass=WrappedConfig, factory=factory, getExecutionOrder=PluginClass.getExecutionOrder)
    WrappedPlugin = type("GAaPFlux" + Base.__name__, (Base,), typeDict)

    Base.registry.register(name, WrappedPlugin)
    return WrappedPlugin, WrappedConfig

def wrapPluginForced(Base, PluginClass=BaseGaapFluxPlugin, ConfigClass=BaseGaapFluxConfig,
                        name=PLUGIN_NAME, factory=BaseGaapFluxPlugin):

    def forcedPluginFactory(name, config, schemaMapper, metadata):
        return factory(name, config, schemaMapper.editOutputSchema(), metadata)
    return wrapPlugin(Base, PluginClass=PluginClass, ConfigClass=ConfigClass, name=name,
                        factory=staticmethod(forcedPluginFactory))

GaapFluxPlugin, GaapFluxConfig = wrapPlugin(WrappedSingleFramePlugin)
ForcedGaapFluxPlugin, ForcedGaapFluxConfig = wrapPluginForced(WrappedForcedPlugin)
