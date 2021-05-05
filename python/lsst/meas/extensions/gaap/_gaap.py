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

from __future__ import annotations

__all__ = ("GaapFluxPlugin", "GaapFluxConfig", "ForcedGaapFluxPlugin", "ForcedGaapFluxConfig")

from typing import Optional, Generator
import itertools
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.meas.base as measBase
from lsst.meas.base.fluxUtilities import FluxResultKey
import lsst.pex.config as pexConfig
from lsst.ip.diffim import ModelPsfMatchTask
from lsst.pex.exceptions import RuntimeError as pexRuntimeError
import scipy.signal

PLUGIN_NAME = "ext_gaap_GaapFlux"


class GaapConvolutionError(pexRuntimeError):
    """Collection of any unexpected errors in GAaP during PSF Gaussianization.

    The PSF Gaussianization procedure using `modelPsfMatchTask` may throw
    exceptions for certain target PSFs. Such errors are caught until all
    measurements are at least attempted. The complete traceback information
    is lost, but unique error messages are preserved.

    Parameters
    ----------
    errors : `dict` [`str`, `Exception`]
        The values are exceptions raised, while the keys are the loop variables
        (in `str` format) where the exceptions were raised.
    """
    def __init__(self, errors: dict[str, Exception], *args, **kwds):
        message = "Problematic scaling factors = "
        message += ", ".join(errors)
        message += " Errors: "
        message += " | ".join(set(msg.__repr__() for msg in errors.values()))  # msg.cpp.what() misses type
        super().__init__(message, *args, **kwds)


class BaseGaapFluxConfig(measBase.BaseMeasurementPluginConfig):
    """Configuration parameters for Gaussian Aperture and PSF (GAaP) plugin.
    """
    def _greaterThanOrEqualToUnity(x: float) -> bool:  # noqa: N805
        """Returns True if the input ``x`` is greater than 1.0, else False.
        """
        return x >= 1

    def _isOdd(x: int) -> bool:  # noqa: N805
        """Returns True if the input ``x`` is positive and odd, else False.
        """
        return (x%2 == 1) & (x > 0)

    sigmas = pexConfig.ListField(
        dtype=float,
        default=[4.0, 5.0],
        doc="List of sigmas (in pixels) of circular Gaussian apertures to apply on "
            "pre-seeing galaxy images. These should be somewhat larger than the PSF "
            "(determined by ``scalingFactors``) to avoid measurement failures.")

    scalingFactors = pexConfig.ListField(
        dtype=float,
        default=[1.15],
        itemCheck=_greaterThanOrEqualToUnity,
        doc="List of factors with which the seeing should be scaled to obtain the "
            "sigma values of the target Gaussian PSF. The factor should not be less "
            "than unity to avoid the PSF matching task to go into deconvolution mode "
            "and should ideally be slightly greater than unity.")

    modelPsfMatch = pexConfig.ConfigurableField(
        target=ModelPsfMatchTask,
        doc="PSF Gaussianization Task")

    modelPsfDimension = pexConfig.Field(
        dtype=int,
        default=65,
        check=_isOdd,
        doc="The dimensions (width and height) of the target PSF image in pixels. Must be odd.")

    # scaleByFwm is the only config field of modelPsfMatch Task that we allow
    # the user to set without explicitly setting the modelPsfMatch config.
    @property
    def scaleByFwhm(self) -> bool:
        return self.modelPsfMatch.kernel.active.scaleByFwhm

    @scaleByFwhm.setter
    def scaleByFwhm(self, value) -> None:
        self.modelPsfMatch.kernel.active.scaleByFwhm = value

    def setDefaults(self) -> None:
        # TODO: DM-27482 might change these values.
        self.modelPsfMatch.kernel.active.alardNGauss = 1
        self.modelPsfMatch.kernel.active.alardDegGaussDeconv = 1
        self.modelPsfMatch.kernel.active.alardDegGauss = [8]
        self.modelPsfMatch.kernel.active.alardGaussBeta = 1.0
        self.modelPsfMatch.kernel.active.spatialKernelOrder = 0

    def validate(self):
        super().validate()
        self.modelPsfMatch.validate()
        assert self.modelPsfMatch.kernel.active.alardNGauss == 1

    @staticmethod
    def _getGaapResultName(sF: float, sigma: float, name: Optional[str] = None) -> str:
        """Return the base name for GAaP fields

        For example, for a scaling factor of 1.15 for seeing and sigma of the
        effective Gaussian aperture of 4.0 pixels, the returned value would be
        "ext_gaap_GaapFlux_1_15x_4_0".

        Notes
        -----
        Being a static method, this does not check if measurements correspond
        to the input arguments. Instead, users should use
        `getAllGaapResultNames` to obtain the full list of base names.

        This is not a config-y thing, but is placed here to make the fieldnames
        from GAaP measurements available outside the plugin.

        Parameters
        ----------
        sF : `float`
            The factor by which the trace radius of the PSF must be scaled.
        sigma : `float`
            Sigma of the effective Gaussian aperture (PSF-convolved explicit
            aperture).
        name : `str`, optional
            The exact registered name of the GAaP plugin, typically either
            "ext_gaap_GaapFlux" or "undeblended_ext_gaap_GaapFlux". If ``name``
            is None, then only the middle part (1_15x_4_0 in the example)
            without the leading underscore is returned.

        Returns
        -------
        baseName : `str`
            Base name for GAaP field.
        """
        suffix = "_".join((str(sF).replace(".", "_")+"x", str(sigma).replace(".", "_")))
        if name is None:
            return suffix
        return "_".join((name, suffix))

    def getAllGaapResultNames(self, name: Optional[str] = None) -> Generator[str]:
        """Generate the base names for all of the GAaP fields.

        For example, if the plugin is configured with `scalingFactors` = [1.15]
        and `sigmas` = [4.0, 5.0] the returned expression would yield
        ("ext_gaap_GaapFlux_1_15x_4_0", "ext_gaap_GaapFlux_1_15x_5_0") when
        called with ``name`` = "ext_gaap_GaapFlux".

        Parameters
        ----------
        name : `str`, optional
            The exact registered name of the GAaP plugin, typically either
            "ext_gaap_GaapFlux" or "undeblended_ext_gaap_GaapFlux". If ``name``
            is None, then only the middle parts (("1_15x_4_0", "1_15x_5_0"),
            for example) without the leading underscores are returned.

        Returns
        -------
        baseNames : `generator`
            A generator expression yielding all the base names.
        """
        scalingFactors = self.scalingFactors
        sigmas = self.sigmas
        baseNames = (self._getGaapResultName(sF, sigma, name)
                     for sF, sigma in itertools.product(scalingFactors, sigmas))
        return baseNames


class BaseGaapFluxPlugin(measBase.GenericPlugin):
    """Gaussian-Aperture and PSF flux (GAaP) base plugin

    Parameters
    ----------
    config : `BaseGaapFluxConfig`
        Plugin configuration.
    name : `str`
        Plugin name, for registering.
    schema : `lsst.afw.table.Schema`
        The schema for the measurement output catalog. New fields will be added
        to hold measurements produced by this plugin.
    metadata : `lsst.daf.base.PropertySet`
        Plugin metadata that will be attached to the output catalog.

    Raises
    ------
    GaapConvolutionError
        Raised if the PSF Gaussianization fails for one or more target PSFs.
    lsst.meas.base.FatalAlgorithmError
        Raised if the Exposure does not contain a PSF model.
    """

    ConfigClass = BaseGaapFluxConfig

    def __init__(self, config, name, schema, metadata) -> None:
        super().__init__(config, name, schema, metadata)

        # Flag definitions for each variant of GAaP measurement
        flagDefs = measBase.FlagDefinitionList()
        for sF, sigma in itertools.product(self.config.scalingFactors, self.config.sigmas):
            baseName = self.ConfigClass._getGaapResultName(sF, sigma, name)
            baseString = f"with {sigma} aperture after scaling the seeing by {sF}"
            schema.addField(schema.join(baseName, "instFlux"), type="D",
                            doc="GAaP Flux " + baseString)
            schema.addField(schema.join(baseName, "instFluxErr"), type="D",
                            doc="GAaP Flux error " + baseString)

            # Remove the prefix_ since FlagHandler prepends it
            middleName = self.ConfigClass._getGaapResultName(sF, sigma)
            flagDefs.add(schema.join(middleName, "flag_bigpsf"), ("The Gaussianized PSF is "
                                                                  "bigger than the aperture"
                                                                  ))
        self.flagHandler = measBase.FlagHandler.addFields(schema, name, flagDefs)
        self.EdgeFlagKey = schema.addField(schema.join(name, "flag_edge"), type="Flag",
                                           doc="Source is too close to the edge")

        self.psfMatchTask = self.config.modelPsfMatch.target(config=self.config.modelPsfMatch)

    @classmethod
    def getExecutionOrder(cls) -> float:
        # Docstring inherited.
        return cls.FLUX_ORDER

    @staticmethod
    def _computeKernelAcf(kernel: lsst.afw.math.Kernel) -> lsst.afw.image.Image:  # noqa: F821
        """Compute the auto-correlation function of ``kernel``.

        Parameters
        ----------
        kernel : `~lsst.afw.math.Kernel`
            The kernel for which auto-correlation function is to be computed.

        Returns
        -------
        acfImage : `~lsst.afw.image.Image`
            The two-dimensional auto-correlation function of ``kernel``.
        """
        kernelImage = afwImage.ImageD(kernel.getDimensions())
        kernel.computeImage(kernelImage, False)
        acfArray = scipy.signal.correlate2d(kernelImage.array, kernelImage.array, boundary='fill')
        acfImage = afwImage.ImageD(acfArray)
        return acfImage

    @staticmethod
    def _getFluxErrScaling(kernelAcf: lsst.afw.image.Image,  # noqa: F821
                           aperShape: lsst.afw.geom.Quadrupole) -> float:  # noqa: F821
        """Calculate the value by which the standard error has to be scaled due
        to noise correlations.

        This calculates the correction to apply to the naively computed
        `instFluxErr` to account for correlations in the pixel noise introduced
        in the PSF-Gaussianization step.
        This method performs the integral in Eq. A17 of Kuijken et al. (2015).

        The returned value equals
        :math:`\\int\\mathrm{d}x C^G(x) \\exp(-x^T Q^{-1}x/4)`
        where :math: `Q` is ``aperShape`` and :math: `C^G(x)` is ``kernelAcf``.

        Parameters
        ----------
        kernelAcf : `~lsst.afw.image.Image`
            The auto-correlation function (ACF) of the PSF matching kernel.
        aperShape : `~lsst.afw.geom.Quadrupole`
            The shape parameter of the Gaussian function which was used to
            measure GAaP flux.

        Returns
        -------
        fluxErrScaling : `float`
            The factor by which the standard error on GAaP flux must be scaled.
        """
        aperShapeX2 = afwGeom.Quadrupole(2*aperShape.getParameterVector())
        corrFlux = measBase.SdssShapeAlgorithm.computeFixedMomentsFlux(kernelAcf, aperShapeX2,
                                                                       kernelAcf.getBBox().getCenter())
        fluxErrScaling = (0.5*corrFlux.instFlux)**0.5
        return fluxErrScaling

    def _convolve(self, exposure: afwImage.Exposure, modelPsf: afwDetection.GaussianPsf,
                  measRecord: lsst.afw.table.SourceRecord) -> lsst.pipe.base.Struct:  # noqa: F821
        """Convolve the ``exposure`` to make the PSF same as ``modelPsf``.

        Parameters
        ----------
        exposure : `~lsst.afw.image.Exposure`
            Original (full) exposure containing all the sources.
        modelPsf : `~lsst.afw.detection.GaussianPsf`
            Target PSF to which to match.
        measRecord : `~lsst.afw.tabe.SourceRecord`
            Record for the source to be measured.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            ``result`` is the Struct returned by `modelPsfMatch` task. Notably,
            it contains a ``psfMatchedExposure``, which is the exposure
            containing the source, convolved to the target seeing and
            ``psfMatchingKernel``, the kernel that ``exposure`` was convolved
            by to obtain ``psfMatchedExposure``. Typically, the bounding box of
            ``psfMatchedExposure`` is larger than that of the footprint.
        """
        footprint = measRecord.getFootprint()
        bbox = footprint.getBBox()

        # The kernelSize is guaranteed to be odd, say 2N+1 pixels (N=10 by
        # default). The flux inside the footprint is smeared by N pixels on
        # either side, which is region of interest. The PSF matching sets
        # NO_DATA mask bit in the outermost N pixels. To account for these nans
        # along the edges, the subExposure needs to be expanded by another
        # N pixels. So grow the bounding box initially by 2N pixels on either
        # side.
        pixToGrow = self.config.modelPsfMatch.kernel.active.kernelSize - 1
        bbox.grow(pixToGrow)

        # The bounding box may become too big and go out of bounds for sources
        # near the edge. Clip the subExposure to the exposure's bounding box.
        # Set the flag_edge marking that the bbox of the footprint could not
        # be grown fully but do not set it as a failure.
        if not exposure.getBBox().contains(bbox):
            bbox.clip(exposure.getBBox())
            measRecord.setFlag(self.EdgeFlagKey, True)

        subExposure = exposure[bbox]

        # The size parameter of the basis has to be set dynamically.
        # `basisSigmaGauss` is a keyword argument in DM-28955.
        task = self.psfMatchTask
        # The modelPsfTask requires the modification made in DM-28955.
        result = task.run(exposure=subExposure, referencePsfModel=modelPsf,
                          basisSigmaGauss=[modelPsf.getSigma()])
        # TODO: DM-27407 will re-Gaussianize the exposure to make the PSF even
        # more Gaussian-like

        # Do not let the variance plane be rescaled since we handle it
        # carefully later using _getFluxScaling method
        result.psfMatchedExposure.variance.array = subExposure.variance.array

        # N pixels around the edges will have NO_DATA mask bit set,
        # where 2N+1 is the kernelSize. Set N number of pixels to erode without
        # reusing pixToGrow, as pixToGrow can be anything in principle.
        pixToErode = self.config.modelPsfMatch.kernel.active.kernelSize//2
        result.psfMatchedExposure = result.psfMatchedExposure[bbox.erodedBy(pixToErode)]
        return result

    def measure(self, measRecord: lsst.afw.table.SourceRecord, exposure: afwImage.Exposure,  # noqa: F821
                center: lsst.geom.Point2D) -> None:  # noqa: F821
        # Docstring inherited.
        psf = exposure.getPsf()
        if psf is None:
            raise measBase.FatalAlgorithmError("No PSF in exposure")

        seeing = psf.computeShape(center).getTraceRadius()
        errorCollection = dict()
        for sF in self.config.scalingFactors:
            targetSigma = sF*seeing
            stampSize = self.config.modelPsfDimension
            targetPsf = afwDetection.GaussianPsf(stampSize, stampSize, targetSigma)
            try:
                result = self._convolve(exposure, targetPsf, measRecord)

            except Exception as error:
                errorCollection[str(sF)] = error
                continue

            convolved = result.psfMatchedExposure
            kernelAcf = self._computeKernelAcf(result.psfMatchingKernel)

            for sigma in self.config.sigmas:
                baseName = self.ConfigClass._getGaapResultName(sF, sigma, self.name)
                if targetSigma >= sigma:
                    flagKey = measRecord.schema.join(baseName, "flag_bigpsf")
                    measRecord.set(flagKey, 1)
                    continue

                aperSigma2 = sigma**2 - targetSigma**2
                aperShape = afwGeom.Quadrupole(aperSigma2, aperSigma2, 0.0)
                fluxResult = measBase.SdssShapeAlgorithm.computeFixedMomentsFlux(convolved.getMaskedImage(),
                                                                                 aperShape, center)
                # Calculate the pre-factor in Eq. A16 of Kuijken et al. (2015)
                # to scale the flux. Include an extra factor of 0.5 to undo
                # the normalization factor of 2 in `computeFixedMomentsFlux`.
                fluxScaling = 0.5*sigma**2/aperSigma2

                # Calculate the integral in Eq. A17 of Kuijken et al. (2015)
                # ``fluxErrScaling`` contains the factors not captured by
                # ``fluxScaling`` and `instFluxErr`. It is 1 theoretically
                # if ``kernelAcf`` is a Dirac-delta function.
                fluxErrScaling = self._getFluxErrScaling(kernelAcf, aperShape)

                # Scale the fluxResult and copy result to record
                fluxResult.instFlux *= fluxScaling
                fluxResult.instFluxErr *= fluxScaling*fluxErrScaling
                fluxResultKey = FluxResultKey(measRecord.schema[baseName])
                fluxResultKey.set(measRecord, fluxResult)

        # Raise GaapConvolutionError before exiting the plugin
        # if the collection of errors is not empty
        if errorCollection:
            raise GaapConvolutionError(errorCollection)


GaapFluxConfig = BaseGaapFluxConfig
GaapFluxPlugin = BaseGaapFluxPlugin.makeSingleFramePlugin(PLUGIN_NAME)
"""Single-frame version of `GaapFluxPlugin`.
"""

ForcedGaapFluxConfig = BaseGaapFluxConfig
ForcedGaapFluxPlugin = BaseGaapFluxPlugin.makeForcedPlugin(PLUGIN_NAME)
"""Forced version of `GaapFluxPlugin`.
"""
