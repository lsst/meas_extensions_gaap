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

__all__ = ("SingleFrameGaapFluxPlugin", "SingleFrameGaapFluxConfig",
           "ForcedGaapFluxPlugin", "ForcedGaapFluxConfig")

from typing import Generator, Optional, Union
from functools import partial
import itertools
import logging
import lsst.afw.detection as afwDetection
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
from lsst.pex.exceptions import InvalidParameterError
import scipy.signal
from ._gaussianizePsf import GaussianizePsfTask

PLUGIN_NAME = "ext_gaap_GaapFlux"


class GaapConvolutionError(measBase.MeasurementError):
    """Raised when there is an error in GAaP convolution.
    """


class NoPixelError(measBase.MeasurementError):
    """Raised when the footprint has no pixels.
    """


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
        default=[0.7, 1.0],
        doc="List of sigmas (in arcseconds) of circular Gaussian apertures to apply on "
            "pre-seeing galaxy images. These should be somewhat larger than the PSF "
            "(determined by ``scalingFactors``) to avoid measurement failures."
    )

    scalingFactors = pexConfig.ListField(
        dtype=float,
        default=[1.15],
        itemCheck=_greaterThanOrEqualToUnity,
        doc="List of factors with which the seeing should be scaled to obtain the "
            "sigma values of the target Gaussian PSF. The factor should not be less "
            "than unity to avoid the PSF matching task to go into deconvolution mode "
            "and should ideally be slightly greater than unity. The runtime of the "
            "plugin scales linearly with the number of elements in the list."
    )

    _modelPsfMatch = pexConfig.ConfigurableField(
        target=GaussianizePsfTask,
        doc="PSF Gaussianization Task"
    )

    _modelPsfDimension = pexConfig.Field(
        dtype=int,
        default=65,
        check=_isOdd,
        doc="The dimensions (width and height) of the target PSF image in pixels. Must be odd."
    )

    doPsfPhotometry = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Perform PSF photometry after PSF-Gaussianization to validate Gaussianization accuracy? "
            "This does not produce consistent color estimates. If setting it to `True`, it must be done so "
            "prior to registering the plugin for aperture correction if ``registerForApCorr`` is also `True`."
    )

    doOptimalPhotometry = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Perform optimal photometry with near maximal SNR using an adaptive elliptical aperture? "
            "This requires a shape algorithm to have been run previously."
    )

    registerForApCorr = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Register measurements for aperture correction? "
            "The aperture correction registration is done when the plugin is instatiated and not "
            "during import because the column names are derived from the configuration rather than being "
            "static. Sometimes you want to turn this off, e.g., when you use aperture corrections derived "
            "from somewhere else through a 'proxy' mechanism."
    )

    # scaleByFwm is the only config field of modelPsfMatch Task that we allow
    # the user to set without explicitly setting the modelPsfMatch config.
    # It is intended to abstract away the underlying implementation.
    @property
    def scaleByFwhm(self) -> bool:
        """Config parameter of the PSF Matching task.
        Scale kernelSize, alardGaussians by input Fwhm?
        """
        return self._modelPsfMatch.kernel.active.scaleByFwhm

    @scaleByFwhm.setter
    def scaleByFwhm(self, value: bool) -> None:
        self._modelPsfMatch.kernel.active.scaleByFwhm = value

    @property
    def gaussianizationMethod(self) -> str:
        """Type of convolution to use for PSF-Gaussianization."""
        return self._modelPsfMatch.convolutionMethod

    @gaussianizationMethod.setter
    def gaussianizationMethod(self, value: str) -> None:
        self._modelPsfMatch.convolutionMethod = value

    @property
    def _sigmas(self) -> list:
        """List of values set in ``sigmas`` along with special apertures such
        as "PsfFlux" and "Optimal" if applicable.
        """
        return self.sigmas.list() + ["PsfFlux"]*self.doPsfPhotometry + ["Optimal"]*self.doOptimalPhotometry

    def setDefaults(self) -> None:
        # Docstring inherited
        super().setDefaults()
        self._modelPsfMatch.kernel.active.alardNGauss = 1
        self._modelPsfMatch.kernel.active.alardDegGauss = [4]
        self._modelPsfMatch.kernel.active.alardGaussBeta = 1.0
        self._modelPsfMatch.kernel.active.spatialKernelOrder = 0
        self.scaleByFwhm = True

    def validate(self):
        super().validate()
        self._modelPsfMatch.validate()
        assert self._modelPsfMatch.kernel.active.alardNGauss == 1

    @staticmethod
    def _getGaapResultName(scalingFactor: float, sigma: Union[float, str], name: Optional[str] = None) -> str:
        """Return the base name for GAaP fields

        For example, for a scaling factor of 1.15 for seeing and sigma of the
        effective Gaussian aperture of 0.7 arcsec, the returned value would be
        "ext_gaap_GaapFlux_1_15x_0_7".

        Notes
        -----
        Being a static method, this does not check if measurements correspond
        to the input arguments. Instead, users should use
        `getAllGaapResultNames` to obtain the full list of base names.

        This is not a config-y thing, but is placed here to make the fieldnames
        from GAaP measurements available outside the plugin.

        Parameters
        ----------
        scalingFactor : `float`
            The factor by which the trace radius of the PSF must be scaled.
        sigma : `float` or `str`
            Sigma of the effective Gaussian aperture (PSF-convolved explicit
            aperture) or "PsfFlux" for PSF photometry post PSF-Gaussianization.
        name : `str`, optional
            The exact registered name of the GAaP plugin, typically either
            "ext_gaap_GaapFlux" or "undeblended_ext_gaap_GaapFlux". If ``name``
            is None, then only the middle part (1_15x_0_7 in the example)
            without the leading underscore is returned.

        Returns
        -------
        baseName : `str`
            Base name for GAaP field.
        """
        suffix = "_".join((str(scalingFactor).replace(".", "_")+"x", str(sigma).replace(".", "_")))
        if name is None:
            return suffix
        return "_".join((name, suffix))

    def getAllGaapResultNames(self, name: Optional[str] = PLUGIN_NAME) -> Generator[str]:
        """Generate the base names for all of the GAaP fields.

        For example, if the plugin is configured with `scalingFactors` = [1.15]
        and `sigmas` = [0.7, 1.0] the returned expression would yield
        ("ext_gaap_GaapFlux_1_15x_0_7", "ext_gaap_GaapFlux_1_15x_1_0") when
        called with ``name`` = "ext_gaap_GaapFlux". It will also generate
        "ext_gaap_GaapFlux_1_15x_PsfFlux" if `doPsfPhotometry` is True.

        Parameters
        ----------
        name : `str`, optional
            The exact registered name of the GAaP plugin, typically either
            "ext_gaap_GaapFlux" or "undeblended_ext_gaap_GaapFlux". If ``name``
            is None, then only the middle parts (("1_15x_0_7", "1_15x_1_0"),
            for example) without the leading underscores are returned.

        Returns
        -------
        baseNames : `generator`
            A generator expression yielding all the base names.
        """
        scalingFactors = self.scalingFactors
        sigmas = self._sigmas
        baseNames = (self._getGaapResultName(scalingFactor, sigma, name)
                     for scalingFactor, sigma in itertools.product(scalingFactors, sigmas))
        return baseNames


class BaseGaapFluxMixin:
    """Mixin base class for Gaussian-Aperture and PSF (GAaP) photometry
    algorithm.

    This class does almost all the heavy-lifting for its two derived classes,
    SingleFrameGaapFluxPlugin and ForcedGaapFluxPlugin which simply adapt it to
    the slightly different interfaces for single-frame and forced measurement.
    This class implements the GAaP algorithm and is intended for code reuse
    by the two concrete derived classes by including this mixin class.

    Parameters
    ----------
    config : `BaseGaapFluxConfig`
        Plugin configuration.
    name : `str`
        Plugin name, for registering.
    schema : `lsst.afw.table.Schema`
        The schema for the measurement output catalog. New fields will be added
        to hold measurements produced by this plugin.
    logName : `str`, optional
        Name to use when logging errors. This is typically provided by the
        measurement framework.

    Raises
    ------
    GaapConvolutionError
        Raised if the PSF Gaussianization fails for one or more target PSFs.
    lsst.meas.base.FatalAlgorithmError
        Raised if the Exposure does not contain a PSF model.
    """

    ConfigClass = BaseGaapFluxConfig
    hasLogName = True

    def __init__(self, config: BaseGaapFluxConfig, name, schema, logName=None) -> None:
        # Flag definitions for each variant of GAaP measurement
        flagDefs = measBase.FlagDefinitionList()
        for scalingFactor, sigma in itertools.product(config.scalingFactors, config.sigmas):
            baseName = self.ConfigClass._getGaapResultName(scalingFactor, sigma, name)
            doc = f"GAaP Flux with {sigma} aperture after multiplying the seeing by {scalingFactor}"
            measBase.FluxResultKey.addFields(schema, name=baseName, doc=doc)

            # Remove the prefix_ since FlagHandler prepends it
            middleName = self.ConfigClass._getGaapResultName(scalingFactor, sigma)
            flagDefs.add(schema.join(middleName, "flag_bigPsf"), "The Gaussianized PSF is "
                                                                 "bigger than the aperture")
            flagDefs.add(schema.join(middleName, "flag"), "Generic failure flag for this set of config "
                                                          "parameters. ")

        # PSF photometry
        if config.doPsfPhotometry:
            for scalingFactor in config.scalingFactors:
                baseName = self.ConfigClass._getGaapResultName(scalingFactor, "PsfFlux", name)
                doc = f"GAaP Flux with PSF aperture after multiplying the seeing by {scalingFactor}"
                measBase.FluxResultKey.addFields(schema, name=baseName, doc=doc)

                # Remove the prefix_ since FlagHandler prepends it
                middleName = self.ConfigClass._getGaapResultName(scalingFactor, "PsfFlux")
                flagDefs.add(schema.join(middleName, "flag"), "Generic failure flag for this set of config "
                                                              "parameters. ")

        if config.doOptimalPhotometry:
            # Add fields to hold the optimal aperture shape
            # OptimalPhotometry case will fetch the aperture shape from here.
            self.optimalShapeKey = afwTable.QuadrupoleKey.addFields(schema, schema.join(name, "OptimalShape"),
                                                                    doc="Pre-seeing aperture used for "
                                                                        "optimal GAaP photometry")
            for scalingFactor in config.scalingFactors:
                baseName = self.ConfigClass._getGaapResultName(scalingFactor, "Optimal", name)
                docstring = f"GAaP Flux with optimal aperture after multiplying the seeing by {scalingFactor}"
                measBase.FluxResultKey.addFields(schema, name=baseName, doc=docstring)

                # Remove the prefix_ since FlagHandler prepends it
                middleName = self.ConfigClass._getGaapResultName(scalingFactor, "Optimal")
                flagDefs.add(schema.join(middleName, "flag_bigPsf"), "The Gaussianized PSF is "
                                                                     "bigger than the aperture")
                flagDefs.add(schema.join(middleName, "flag"), "Generic failure flag for this set of config "
                                                              "parameters. ")

        if config.registerForApCorr:
            for baseName in config.getAllGaapResultNames(name):
                measBase.addApCorrName(baseName)

        for scalingFactor in config.scalingFactors:
            flagName = self.ConfigClass._getGaapResultName(scalingFactor, "flag_gaussianization")
            flagDefs.add(flagName, "PSF Gaussianization failed when trying to scale by this factor.")

        self.log = logging.getLogger(logName)
        self.flagHandler = measBase.FlagHandler.addFields(schema, name, flagDefs)
        self.EdgeFlagKey = schema.addField(schema.join(name, "flag_edge"), type="Flag",
                                           doc="Source is too close to the edge")
        self.NoPixelKey = schema.addField(schema.join(name, "flag_no_pixel"), type="Flag",
                                          doc="No pixels in the footprint")
        self._failKey = schema.addField(name + '_flag', type="Flag", doc="Set for any fatal failure")

        self.psfMatchTask = config._modelPsfMatch.target(config=config._modelPsfMatch)

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
        aperShapeX2 = aperShape.convolve(aperShape)
        corrFlux = measBase.SdssShapeAlgorithm.computeFixedMomentsFlux(kernelAcf, aperShapeX2,
                                                                       kernelAcf.getBBox().getCenter())
        fluxErrScaling = (0.5*corrFlux.instFlux)**0.5
        return fluxErrScaling

    def _gaussianize(self, exposure: afwImage.Exposure, modelPsf: afwDetection.GaussianPsf,
                     measRecord: lsst.afw.table.SourceRecord) -> lsst.pipe.base.Struct:  # noqa: F821
        """Modify the ``exposure`` so that its PSF is a Gaussian.

        Compute the convolution kernel to make the PSF same as ``modelPsf``
        and return the Gaussianized exposure in a struct.

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
        # either side, which is region of interest. So grow the bounding box
        # initially by N pixels on either side.
        pixToGrow = self.config._modelPsfMatch.kernel.active.kernelSize//2
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
        result = self.psfMatchTask.run(exposure=subExposure, center=measRecord.getCentroid(),
                                       targetPsfModel=modelPsf,
                                       basisSigmaGauss=[modelPsf.getSigma()])
        # TODO: DM-27407 will re-Gaussianize the exposure to make the PSF even
        # more Gaussian-like

        # Do not let the variance plane be rescaled since we handle it
        # carefully later using _getFluxScaling method
        result.psfMatchedExposure.variance.array = subExposure.variance.array
        return result

    def _measureFlux(self, measRecord: lsst.afw.table.SourceRecord,
                     exposure: afwImage.Exposure, kernelAcf: afwImage.Image,
                     center: lsst.geom.Point2D, aperShape: afwGeom.Quadrupole,
                     baseName: str, fluxScaling: Optional[float] = None) -> None:
        """Measure the flux and populate the record.

        Parameters
        ----------
        measRecord : `~lsst.afw.table.SourceRecord`
            Catalog record for the source being measured.
        exposure : `~lsst.afw.image.Exposure`
            Subexposure containing the deblended source being measured.
            The PSF attached to it should nominally be an
            `lsst.afw.Detection.GaussianPsf` object, but not enforced.
        kernelAcf : `~lsst.afw.image.Image`
            An image representating the auto-correlation function of the
            PSF-matching kernel.
        center : `~lsst.geom.Point2D`
            The centroid position of the source being measured.
        aperShape : `~lsst.afw.geom.Quadrupole`
            The shape parameter of the post-seeing Gaussian aperture.
            It should be a valid quadrupole if ``fluxScaling`` is specified.
        baseName : `str`
            The base name of the GAaP field.
        fluxScaling : `float`, optional
            The multiplication factor by which the measured flux has to be
            scaled. If `None` or unspecified, the pre-factor in Eq. A16
            of Kuijken et al. (2015) is computed and applied.
        """
        if fluxScaling is None:
            # Calculate the pre-factor in Eq. A16 of Kuijken et al. (2015)
            # to scale the flux. Include an extra factor of 0.5 to undo
            # the normalization factor of 2 in `computeFixedMomentsFlux`.
            try:
                aperShape.normalize()
                # Calculate the pre-seeing aperture.
                preseeingShape = aperShape.convolve(exposure.getPsf().computeShape(center))
                fluxScaling = 0.5*preseeingShape.getArea()/aperShape.getArea()
            except (InvalidParameterError, ZeroDivisionError):
                self._setFlag(measRecord, baseName, "bigPsf")
                return

        # Calculate the integral in Eq. A17 of Kuijken et al. (2015)
        # ``fluxErrScaling`` contains the factors not captured by
        # ``fluxScaling`` and `instFluxErr`. It is 1 theoretically
        # if ``kernelAcf`` is a Dirac-delta function.
        fluxErrScaling = self._getFluxErrScaling(kernelAcf, aperShape)

        fluxResult = measBase.SdssShapeAlgorithm.computeFixedMomentsFlux(exposure.getMaskedImage(),
                                                                         aperShape, center)

        # Scale the quantities in fluxResult and copy result to record
        fluxResult.instFlux *= fluxScaling
        fluxResult.instFluxErr *= fluxScaling*fluxErrScaling
        fluxResultKey = measBase.FluxResultKey(measRecord.schema[baseName])
        fluxResultKey.set(measRecord, fluxResult)

    def _gaussianizeAndMeasure(self, measRecord: lsst.afw.table.SourceRecord,
                               exposure: afwImage.Exposure,
                               center: lsst.geom.Point2D) -> None:
        """Measure the properties of a source on a single image.

        The image may be from a single epoch, or it may be a coadd.

        Parameters
        ----------
        measRecord : `~lsst.afw.table.SourceRecord`
            Record describing the object being measured. Previously-measured
            quantities may be retrieved from here, and it will be updated
            in-place with the outputs of this plugin.
        exposure : `~lsst.afw.image.ExposureF`
            The pixel data to be measured, together with the associated PSF,
            WCS, etc. All other sources in the image should have been replaced
            by noise according to deblender outputs.
        center : `~lsst.geom.Point2D`
            Centroid location of the source being measured.

        Raises
        ------
        GaapConvolutionError
            Raised if the PSF Gaussianization fails for any of the target PSFs.
        lsst.meas.base.FatalAlgorithmError
            Raised if the Exposure does not contain a PSF model.
        NoPixelError
            Raised if the footprint has no pixels.

        Notes
        -----
        This method is the entry point to the mixin from the concrete derived
        classes.
        """
        # First make sure we have a PSF.
        if (psf := exposure.getPsf()) is None:
            raise measBase.FatalAlgorithmError("No PSF in exposure")

        # Raise errors if the plugin would fail for this record for all
        # scaling factors and sigmas.
        if measRecord.getFootprint().getArea() == 0:
            self._setFlag(measRecord, self.name, "no_pixel")
            self._setScalingAndSigmaFlags(measRecord, self.config.scalingFactors)
            raise NoPixelError("No good pixels in footprint", 1)

        psfSigma = psf.computeShape(center).getTraceRadius()
        if not (psfSigma > 0):  # This captures NaN and negative values.
            center = measRecord.getCentroid()
            self.log.debug("Invalid PSF sigma; cannot solve for PSF matching kernel in GAaP for (%f, %f): %s",
                           center.getX(), center.getY(), "GAaP Convolution Error")
            self._setScalingAndSigmaFlags(
                measRecord,
                self.config.scalingFactors,
                specificFlag="flag_gaussianization",
            )
            raise GaapConvolutionError("Failed to solve for PSF matching kernel", 1)
        else:
            errorCollection = dict()

        wcs = exposure.getWcs()

        for scalingFactor in self.config.scalingFactors:
            targetSigma = scalingFactor*psfSigma
            # If this target PSF is bound to fail for all apertures,
            # set the flags and move on without PSF Gaussianization.
            if self._isAllFailure(measRecord, scalingFactor, targetSigma):
                continue

            stampSize = self.config._modelPsfDimension
            targetPsf = afwDetection.GaussianPsf(stampSize, stampSize, targetSigma)
            try:
                result = self._gaussianize(exposure, targetPsf, measRecord)
            except Exception as error:
                errorCollection[str(scalingFactor)] = error
                continue

            convolved = result.psfMatchedExposure
            kernelAcf = self._computeKernelAcf(result.psfMatchingKernel)

            measureFlux = partial(self._measureFlux, measRecord, convolved, kernelAcf, center)
            # Computing shape is inexpensive and position-independent for a
            # GaussianPsf
            psfShape = targetPsf.computeShape(center)

            if self.config.doPsfPhotometry:
                baseName = self.ConfigClass._getGaapResultName(scalingFactor, "PsfFlux", self.name)
                aperShape = psfShape
                measureFlux(aperShape, baseName, fluxScaling=1)

            if self.config.doOptimalPhotometry:
                baseName = self.ConfigClass._getGaapResultName(scalingFactor, "Optimal", self.name)
                optimalShape = measRecord.get(self.optimalShapeKey)
                aperShape = afwGeom.Quadrupole(optimalShape.getParameterVector()
                                               - psfShape.getParameterVector())
                measureFlux(aperShape, baseName)

            # Iterate over pre-defined circular apertures
            for sigma in self.config.sigmas:
                baseName = self.ConfigClass._getGaapResultName(scalingFactor, sigma, self.name)
                if sigma <= targetSigma * wcs.getPixelScale(center).asArcseconds():
                    # Raise when the aperture is invalid
                    self._setFlag(measRecord, baseName, "bigPsf")
                    continue

                intrinsicShape = afwGeom.Quadrupole(sigma**2, sigma**2, 0.0)  # in sky coordinates
                intrinsicShape.transformInPlace(wcs.linearizeSkyToPixel(center,
                                                                        lsst.geom.arcseconds).getLinear())
                aperShape = afwGeom.Quadrupole(intrinsicShape.getParameterVector()
                                               - psfShape.getParameterVector())
                measureFlux(aperShape, baseName)

        # Raise GaapConvolutionError before exiting the plugin
        # if the collection of errors is not empty
        if errorCollection:
            message = "Problematic scaling factors = "
            message += ", ".join(errorCollection)
            message += " Errors: "
            message += " | ".join(set(msg.__repr__() for msg in errorCollection.values()))
            center = measRecord.getCentroid()
            self.log.debug("Failed to solve for PSF matching kernel in GAaP for (%f, %f): %s",
                           center.getX(), center.getY(), message)
            self._setScalingAndSigmaFlags(
                measRecord,
                errorCollection.keys(),
                specificFlag="flag_gaussianization",
            )
            raise GaapConvolutionError("Failed to solve for PSF matching kernel", 1)

    @staticmethod
    def _setFlag(measRecord, baseName, flagName=None):
        """Set the GAaP flag determined by ``baseName`` and ``flagName``.

        A convenience method to set {baseName}_flag_{flagName} to True.
        This also automatically sets the generic {baseName}_flag to True.
        To set the general plugin flag indicating measurement failure,
        use _failKey directly.

        Parameters
        ----------
        measRecord : `~lsst.afw.table.SourceRecord`
            Record describing the source being measured.
        baseName : `str`
            The base name of the GAaP field for which the flag must be set.
        flagName : `str`, optional
            The name of the specific flag to set along with the general flag.
            If unspecified, only the general flag corresponding to ``baseName``
            is set. For now, the only value that can be specified is "bigPsf".
        """
        if flagName is not None:
            specificFlagKey = measRecord.schema.join(baseName, f"flag_{flagName}")
            measRecord.set(specificFlagKey, True)
        genericFlagKey = measRecord.schema.join(baseName, "flag")
        measRecord.set(genericFlagKey, True)

    def _setScalingAndSigmaFlags(self, measRecord, scalingFactors, specificFlag=None):
        """Set a full suite of flags for scalingFactors/sigmas.

        Parameters
        ----------
        measRecord : `~lsst.afw.table.SourceRecord`
            Record describing the source being measured.
        scalingFactors : `list` [`float`]
            List of scaling factors.
        specificFlag : `str`, optional
            Specific type of flag to set if needed.
        """
        for scalingFactor in scalingFactors:
            if specificFlag is not None:
                flagName = self.ConfigClass._getGaapResultName(scalingFactor, specificFlag,
                                                               self.name)
                measRecord.set(flagName, True)
            for sigma in self.config._sigmas:
                baseName = self.ConfigClass._getGaapResultName(scalingFactor, sigma, self.name)
                self._setFlag(measRecord, baseName)

    def _isAllFailure(self, measRecord, scalingFactor, targetSigma) -> bool:
        """Check if all measurements would result in failure.

        If all of the pre-seeing apertures are smaller than size of the
        target PSF for the given ``scalingFactor``, then set the
        `flag_bigPsf` for all fields corresponding to ``scalingFactor``
        and move on instead of spending computational effort in
        Gaussianizing the exposure.

        Parameters
        ----------
        measRecord : `~lsst.afw.table.SourceRecord`
            Record describing the source being measured.
        scalingFactor : `float`
            The multiplicative factor by which the seeing is scaled.
        targetSigma : `float`
            Sigma (in pixels) of the target circular Gaussian PSF.

        Returns
        -------
        allFailure : `bool`
            A boolean value indicating whether all measurements would fail.

        Notes
        ----
        If doPsfPhotometry is set to True, then this will always return False.
        """
        if self.config.doPsfPhotometry:
            return False

        allFailure = targetSigma >= max(self.config.sigmas)
        # If measurements would fail on all circular apertures, and if
        # optimal elliptical aperture is used, check if that would also fail.
        if self.config.doOptimalPhotometry and allFailure:
            optimalShape = measRecord.get(self.optimalShapeKey)
            aperShape = afwGeom.Quadrupole(optimalShape.getParameterVector()
                                           - [targetSigma**2, targetSigma**2, 0.0])
            allFailure = (aperShape.getIxx() <= 0) or (aperShape.getIyy() <= 0) or (aperShape.getArea() <= 0)

        # Set all failure flags if allFailure is True.
        if allFailure:
            if self.config.doOptimalPhotometry:
                baseName = self.ConfigClass._getGaapResultName(scalingFactor, "Optimal", self.name)
                self._setFlag(measRecord, baseName, "bigPsf")
            for sigma in self.config.sigmas:
                baseName = self.ConfigClass._getGaapResultName(scalingFactor, sigma, self.name)
                self._setFlag(measRecord, baseName, "bigPsf")

        return allFailure

    def fail(self, measRecord, error=None):
        """Record a measurement failure.

        This default implementation simply records the failure in the source
        record and is inherited by the SingleFrameGaapFluxPlugin and
        ForcedGaapFluxPlugin.

        Parameters
        ----------
        measRecord : `lsst.afw.table.SourceRecord`
            Catalog record for the source being measured.
        error : `Exception`
            Error causing failure, or `None`.
        """
        # We only need to set the failKey if no error was specified which
        # signifies that the flagging was already handled.
        if error is None:
            measRecord.set(self._failKey, True)


class SingleFrameGaapFluxConfig(BaseGaapFluxConfig,
                                measBase.SingleFramePluginConfig):
    """Config for SingleFrameGaapFluxPlugin."""


@measBase.register(PLUGIN_NAME)
class SingleFrameGaapFluxPlugin(BaseGaapFluxMixin, measBase.SingleFramePlugin):
    """Gaussian Aperture and PSF photometry algorithm in single-frame mode.

    Parameters
    ----------
    config : `GaapFluxConfig`
        Plugin configuration.
    name : `str`
        Plugin name, for registering.
    schema : `lsst.afw.table.Schema`
        The schema for the measurement output catalog. New fields will be added
        to hold measurements produced by this plugin.
    metadata : `lsst.daf.base.PropertySet`
        Plugin metadata that will be attached to the output catalog.
    logName : `str`, optional
        Name to use when logging errors. This will be provided by the
        measurement framework.

    Notes
    -----
    This plugin must be run in forced mode to produce consistent colors across
    the different bandpasses.
    """
    ConfigClass = SingleFrameGaapFluxConfig

    def __init__(self, config, name, schema, metadata, logName=None):
        BaseGaapFluxMixin.__init__(self, config, name, schema, logName=logName)
        measBase.SingleFramePlugin.__init__(self, config, name, schema, metadata, logName=logName)

    @classmethod
    def getExecutionOrder(cls) -> float:
        # Docstring inherited
        return cls.FLUX_ORDER

    def measure(self, measRecord, exposure):
        # Docstring inherited.
        center = measRecord.getCentroid()
        if self.config.doOptimalPhotometry:
            # The adaptive shape is set to post-seeing aperture.
            # Convolve with the PSF shape to obtain pre-seeing aperture.
            # Refer to pg. 30-31 of Kuijken et al. (2015) for this heuristic.
            # psfShape = measRecord.getPsfShape()  # TODO: DM-30229
            psfShape = afwTable.QuadrupoleKey(measRecord.schema["slot_PsfShape"]).get(measRecord)
            optimalShape = measRecord.getShape().convolve(psfShape)
            # Record the aperture used for optimal photometry
            measRecord.set(self.optimalShapeKey, optimalShape)
        self._gaussianizeAndMeasure(measRecord, exposure, center)


class ForcedGaapFluxConfig(BaseGaapFluxConfig, measBase.ForcedPluginConfig):
    """Config for ForcedGaapFluxPlugin."""


@measBase.register(PLUGIN_NAME)
class ForcedGaapFluxPlugin(BaseGaapFluxMixin, measBase.ForcedPlugin):
    """Gaussian Aperture and PSF (GAaP) photometry plugin in forced mode.

    This is the GAaP plugin to run for consistent colors across the bandpasses.

    Parameters
    ----------
    config : `GaapFluxConfig`
        Plugin configuration.
    name : `str`
        Plugin name, for registering.
    schemaMapper : `lsst.afw.table.SchemaMapper`
        A mapping from reference catalog fields to output catalog fields.
        Output fields will be added to the output schema.
        for the measurement output catalog. New fields will be added
        to hold measurements produced by this plugin.
    metadata : `lsst.daf.base.PropertySet`
        Plugin metadata that will be attached to the output catalog.
    logName : `str`, optional
        Name to use when logging errors. This will be provided by the
        measurement framework.
    """
    ConfigClass = ForcedGaapFluxConfig

    def __init__(self, config, name, schemaMapper, metadata, logName=None):
        schema = schemaMapper.editOutputSchema()
        BaseGaapFluxMixin.__init__(self, config, name, schema, logName=logName)
        measBase.ForcedPlugin.__init__(self, config, name, schemaMapper, metadata, logName=logName)

    @classmethod
    def getExecutionOrder(cls) -> float:
        # Docstring inherited.
        return cls.FLUX_ORDER

    def measure(self, measRecord, exposure, refRecord, refWcs):
        # Docstring inherited.
        wcs = exposure.getWcs()
        center = wcs.skyToPixel(refWcs.pixelToSky(refRecord.getCentroid()))
        if self.config.doOptimalPhotometry:
            # The adaptive shape is set to post-seeing aperture.
            # Convolve it with the PSF shape to obtain pre-seeing aperture.
            # Refer to pg. 30-31 of Kuijken et al. (2015) for this heuristic.
            # psfShape = refRecord.getPsfShape()  # TODO: DM-30229
            psfShape = afwTable.QuadrupoleKey(refRecord.schema["slot_PsfShape"]).get(refRecord)
            optimalShape = refRecord.getShape().convolve(psfShape)
            if not (wcs == refWcs):
                measFromSky = wcs.linearizeSkyToPixel(measRecord.getCentroid(), lsst.geom.radians)
                skyFromRef = refWcs.linearizePixelToSky(refRecord.getCentroid(), lsst.geom.radians)
                measFromRef = measFromSky*skyFromRef
                optimalShape.transformInPlace(measFromRef.getLinear())
            # Record the intrinsic aperture used for optimal photometry.
            measRecord.set(self.optimalShapeKey, optimalShape)
        self._gaussianizeAndMeasure(measRecord, exposure, center)
