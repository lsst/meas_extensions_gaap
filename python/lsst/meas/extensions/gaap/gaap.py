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

import numpy as np
import itertools
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.meas.base
from lsst.meas.base.wrappers import GenericPlugin
from lsst.meas.base import SdssShapeAlgorithm
from lsst.ip.diffim.modelPsfMatch import ModelPsfMatchTask
from lsst.pex.config import ListField
from lsst.pex.config.configurableField import ConfigurableField

__all__ = ("GaapFluxPlugin", "GaapFluxConfig", "ForcedGaapFluxPlugin", "ForcedGaapFluxConfig")

PLUGIN_NAME = "ext_gaap_GaapFlux"


class BaseGaapFluxConfig(lsst.meas.base.BaseMeasurementPluginConfig):
    scalingFactors = ListField(dtype=float, default=[1.0, 1.15],
                               doc="List of scale factors to scale the seeing")
    sigmas = ListField(dtype=float, default=[4.0, 5.0],
                       doc="List of sigmas for Gaussian apertures")
    modelPsfMatch = ConfigurableField(doc="PSF Gaussianization Task", target=ModelPsfMatchTask)

    def setDefaults(self) -> None:
        # TODO: The following will move to a config file later in DM-27482
        self.modelPsfMatch.kernel.active.alardNGauss = 1
        self.modelPsfMatch.kernel.active.alardDegGaussDeconv = 1
        self.modelPsfMatch.kernel.active.alardDegGauss = [8]
        self.modelPsfMatch.kernel.active.alardGaussBeta = 1.0
        self.modelPsfMatch.kernel.active.spatialKernelOrder = 0


class BaseGaapFluxPlugin(GenericPlugin):

    ConfigClass = BaseGaapFluxConfig

    @classmethod
    def getExecutionOrder(cls) -> float:
        return cls.FLUX_ORDER

    def getGaapResultName(self, sF: float, sigma: float) -> str:
        return "_".join((self.name, str(sF).replace(".", "_")+"x", str(sigma).replace(".", "_")))

    def __init__(self, config, name, schema, metadata) -> None:
        """Gaussian Aperture and PSF (GAaP) photometry plugin.
        """
        GenericPlugin.__init__(self, config, name, schema, metadata)
        for sF, sigma in itertools.product(self.config.scalingFactors, self.config.sigmas):
            baseName = self.getGaapResultName(sF, sigma)
            schema.addField(schema.join(baseName, "instFlux"), type="F", doc="GAaP Flux")
            schema.addField(schema.join(baseName, "instFluxErr"), type="F", doc="GAaP Flux error")

    def convolve(self, exposure: afwImage.Exposure, modelPsf: afwDetection.GaussianPsf,
                 footprint: afwDetection.Footprint) -> afwImage.Exposure:
        """Convolve the ``exposure`` to make the PSF same as ``modelPsf``.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Original (full) exposure containing all the sources.
        modelPsf : `lsst.afw.detection.GaussianPsf`
            Target PSF to which to match.
        footprint : `lsst.afw.detection.Footprint`
            Footprint of the source to measure.

        Returns
        -------
        convExp : `lsst.afw.image.Exposure`
            Sub-image containing the source, convolving to the target seeing
        """

        bbox = footprint.getBBox()
        pixToGrow = self.config.modelPsfMatch.kernel.active.detectionConfig.fpGrowPix
        # This is conservative for now.
        # An alternate option is 2*max(self.psfMatch.kConfig.sizeCellX, self.psfMatch.kConfig.sizeCellY)
        bbox.grow(pixToGrow)  # gets too big

        origPsf = exposure.getPsf()

        maskedImage = exposure.getMaskedImage()
        subImage = maskedImage.Factory(maskedImage, bbox)
        subExposure = afwImage.ExposureF(subImage)
        subExposure.setPsf(origPsf)

        config = self.config.modelPsfMatch
        config.setDefaults()
        config.kernel.active.alardSigGauss = [modelPsf.getSigma()]  # The size has to be set dynamically
        task = config.target(config=config)
        result = task.run(exposure=subExposure, referencePsfModel=modelPsf)
        # TODO: DM-27407 will re-Gaussianize the exposure to make the PSF even more Gaussian-like
        convolved = result.psfMatchedExposure
        # Setting nans to zeros in a hacky way! DM-28740 will fix this properly.
        convolved.image.array[np.isnan(convolved.image.array)] = 0.0
        return convolved

    def measure(self, measRecord: lsst.afw.table.SourceRecord, exposure: afwImage.Exposure,
                center: lsst.geom.Point2D) -> None:
        """Measure the GAaP flux
        """

        psf = exposure.getPsf()
        if psf is None:
            raise lsst.meas.base.MeasurementError("No PSF in exposure")

        seeing = psf.computeShape(center).getTraceRadius()
        for sF in self.config.scalingFactors:
            if sF < 1.0:
                # Do not allow the PSF matching to go into deconvolution mode.
                raise lsst.meas.base.FatalAlgorithmError("The scaling factor has to be greater than"
                                                         "or equal to 1")

            target = sF*seeing
            modelPsf = afwDetection.GaussianPsf(65, 65, target)
            try:
                convolved = self.convolve(exposure, modelPsf, measRecord.getFootprint())
            except RuntimeError:
                continue

            for sigma in self.config.sigmas:
                effShape = afwGeom.Quadrupole(sigma**2, sigma**2, 0.0)
                # Assume effShape is elliptical, although it is explicitly circular
                if target**2 >= min(effShape.getIxx(), effShape.getIyy()):
                    raise lsst.meas.base.MeasurementError("The GaussianPsf was larger than the effective"
                                                          "aperture")

                aperShape = afwGeom.Quadrupole(effShape.getIxx()-target**2,
                                               effShape.getIyy()-target**2,
                                               effShape.getIxy())
                fluxResult = SdssShapeAlgorithm.computeFixedMomentsFlux(convolved.getMaskedImage(),
                                                                        aperShape, center)
                fluxScaling = (effShape.getDeterminantRadius()/aperShape.getDeterminantRadius())**2

                # Copy result to record
                baseName = self.getGaapResultName(sF, sigma)
                instFluxKey = measRecord.schema.join(baseName, "instFlux")
                instFluxErrKey = measRecord.schema.join(baseName, "instFluxErr")
                measRecord.set(instFluxKey, fluxScaling*fluxResult.instFlux)
                measRecord.set(instFluxErrKey, fluxScaling*fluxResult.instFluxErr)

    def fail(self, measRecord: lsst.afw.table.SourceRecord, error=None) -> None:
        """Record failure

        Called by the measurement framework when it catches an exception.

        Parameters
        ----------
        measRecord : `lsst.afw.table.SourceRecord`
            Record for source on which the measurement failed.
        error : `Exception`, optional
            Error that occurred, or None.
        """
        GenericPlugin.fail(self, measRecord, error)


GaapFluxConfig = BaseGaapFluxConfig
GaapFluxPlugin = BaseGaapFluxPlugin.makeSingleFramePlugin(PLUGIN_NAME)
"""Single-frame version of `GaapFluxPlugin`.
"""

ForcedGaapFluxConfig = BaseGaapFluxConfig
ForcedGaapFluxPlugin = BaseGaapFluxPlugin.makeForcedPlugin(PLUGIN_NAME)
"""Forced version of `GaapFluxPlugin`.
"""
