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
import os

import lsst.afw.image as afwImage
import lsst.meas.algorithms as measAlg
import lsst.meas.base
import numpy as np
from lsst.afw.geom.skyWcs import makeWcsPairTransform
from lsst.ip.diffim import modelPsfMatch
from lsst.ip.diffim.modelPsfMatch import ModelPsfMatchConfig
from lsst.meas.base.wrappers import (WrappedForcedPlugin,
                                     WrappedSingleFramePlugin)
from lsst.pex.config import ConfigField, Field, ListField, makeConfigClass

# os.environ['LD_LIBRARY_PATH'] += ':/home/kannawad/repo/meas_extensions_gaap/lib'
from .gaapFlux import GaapFluxAlgorithm, GaapFluxControl, GaapFluxTransform

# from lsst.meas.extensions.gaap.gaapFlux import GaapFluxAlgorithm, GaapFluxControl, GaapFluxTransform

__all__ = ("GaapFluxPlugin", "GaapFluxConfig", "ForcedGaapFluxPlugin", "ForcedGaapFluxConfig")

SIGMA_TO_FWHM = 2.0*math.sqrt(2.0*(math.log(2.0)))
PLUGIN_NAME = "ext_gaap_GaapFlux"

GaapFluxConfig = makeConfigClass(GaapFluxControl)


class BaseGaapFluxConfig(ModelPsfMatchConfig):
    scalingFactor = Field(dtype=float, default=1.15, doc="Scale factor to scale the worst seeing")
    aperture = ConfigField(dtype=GaapFluxConfig, doc="Gaussian photometry parameters")

    def getGaapResultName(*args) -> str:
        return "ext_gaap_GaapFlux"

    def setDefaults(self) -> None:
        ModelPsfMatchConfig.setDefaults(self)
        # TODO: The following will move to a config file later in DM-27482
        self.kernel.active.alardNGauss = 1
        self.kernel.active.alardDegGaussDeconv = 1
        self.kernel.active.alardDegGauss = [8]
        self.kernel.active.alardGaussBeta = 1.0
        self.kernel.active.spatialKernelOrder = 0


class BaseGaapFluxPlugin(lsst.meas.base.BaseMeasurementPlugin):
    @classmethod
    def getExecutionOrder(cls) -> float:
        return cls.FLUX_ORDER

    def __init__(self, config, name, schema, metadata) -> None:
        """
        Gaussian Aperture and PSF (GAaP) photometry plugin.
        """
        lsst.meas.base.BaseMeasurementPlugin.__init__(self, config, name)
        self.seeingKey = schema.addField(name, type="F",
                                         doc="original seeing (Gaussian sigma) at position",
                                         units="pixel")
        self.centroidExtractor = lsst.meas.base.SafeCentroidExtractor(schema, name)

        flagDefs = lsst.meas.base.FlagDefinitionList()
        flagDefs.addFailureFlag("error in running ConvolvedFluxPlugin")
        self.flagHandler = lsst.meas.base.FlagHandler.addFields(schema, "GAaP", flagDefs)
        self.gaussianAperture = GaapFluxAlgorithm(config.aperture.makeControl(),
                                                  name, schema)

    def convolve(self, exposure, modelPsf, footprint) -> afwImage.Exposure:
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
        pixToGrow = 8  # Arbitrary for now.
        # An alternate option is 2*max(self.psfMatch.kConfig.sizeCellX, self.psfMatch.kConfig.sizeCellY)
        bbox.grow(pixToGrow)

        # origPsf = exposure.getPsf(bbox.getCenter())
        origPsf = exposure.getPsf()

        maskedImage = exposure.getMaskedImage()
        subImage = maskedImage.Factory(maskedImage, bbox)
        subExposure = afwImage.ExposureF(subImage)
        subExposure.setPsf(origPsf)
        self.config.kernel.active.alardSigGauss = [modelPsf.getSigma()]  # The size has to be set dynamically
        result = modelPsfMatch.ModelPsfMatchTask(config=self.config).run(exposure=subExposure,
                                                                         referencePsfModel=modelPsf)
        # TODO: DM-27407 will re-Gaussianize the exposure to make the PSF even more Gaussian-like
        convolved = result.psfMatchedExposure
        convolved.image.array[np.isnan(convolved.image.array)] = 0.
        return convolved

    def measure(self, measRecord, exposure):
        return self.measureForced(measRecord, exposure, measRecord, None)

    def measureForced(self, measRecord, exposure, refRecord, refWcs) -> None:
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
        seeing = psf.computeShape(center).getTraceRadius()
        target = self.config.scalingFactor*seeing

        modelPsf = measAlg.SingleGaussianPsf(width=64, height=64, sigma=target)
        try:
            convolved = self.convolve(exposure, modelPsf, measRecord.getFootprint())
        except RuntimeError:
            convolved = exposure

        self.measureAperture(measRecord, convolved, self.gaussianAperture)

    def measureAperture(self, measRecord, exposure, aperturePhot) -> None:
        """Perform aperture photometry.

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

    def fail(self, measRecord, error=None) -> None:
        """ Record failure
        """
        self.flagHandler.handleFailure(measRecord)


def wrapPlugin(Base, PluginClass=BaseGaapFluxPlugin, ConfigClass=BaseGaapFluxConfig,
               name=PLUGIN_NAME, factory=BaseGaapFluxPlugin):
    WrappedConfig = type("GAaPFlux" + Base.ConfigClass.__name__, (Base.ConfigClass, ConfigClass), {})
    typeDict = dict(AlgClass=PluginClass, ConfigClass=WrappedConfig, factory=factory,
                    getExecutionOrder=PluginClass.getExecutionOrder)
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
