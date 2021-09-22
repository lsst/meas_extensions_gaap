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

__all__ = ("GaussianizePsfTask", "GaussianizePsfConfig")

import numpy as np
import scipy.signal

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
from lsst.ip.diffim import diffimLib
from lsst.ip.diffim.makeKernelBasisList import makeKernelBasisList
from lsst.ip.diffim.modelPsfMatch import ModelPsfMatchConfig, ModelPsfMatchTask
from lsst.ip.diffim.modelPsfMatch import sigma2fwhm, nextOddInteger
import lsst.log as log
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase


class GaussianizePsfConfig(ModelPsfMatchConfig):
    """Configuration for model-to-model Psf matching."""

    convolutionMethod = pexConfig.ChoiceField(
        dtype=str,
        doc="Which type of convolution to use",
        default="auto",
        allowed={
            "direct": "Brute-force real-space convolution",
            "fft": "Convolve using FFTs (generally faster)",
            "auto": "Choose the faster method between 'direct' and 'fft'",
            "overlap-add": "Convolve using the overlap-add method",
        }
    )


class GaussianizePsfTask(ModelPsfMatchTask):
    """Task to make the PSF at a source Gaussian.

    This is a specialized version of `lsst.ip.diffim.ModelPsfMatchTask` for
    use within the Gaussian-Aperture and PSF (GAaP) photometry plugin. The
    `run` method has a different signature and is optimized for multiple calls.
    The optimization includes treating PSF as spatially constant within the
    footprint of a source and substituting `lsst.afw.math.convolution` method
    with scipy.signal's. The PSF is evaluated at the centroid of the source.
    Unlike `lsst.ip.diffim.ModelPsfMatchTask`, the assessment of the fit from
    residuals is not made. This is assessed via `PsfFlux` in the GAaP plugin.

    See also
    --------
    ModelPsfMatchTask
    """
    ConfigClass = GaussianizePsfConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kConfig = self.config.kernel.active
        self.ps = pexConfig.makePropertySet(self.kConfig)

    def run(self, exposure: lsst.afw.image.Exposure, center: lsst.geom.Point2D,  # noqa: F821
            targetPsfModel: lsst.afw.detection.GaussianPsf,  # noqa: F821
            kernelSum=1.0, basisSigmaGauss=None) -> pipeBase.Struct:
        """Make the PSF of an exposure match a model PSF.

        Parameters
        ----------
        exposure : `~lsst.afw.image.Exposure`
            A (sub-)exposure containing a single (deblended) source being
            measured; it must return a valid PSF model via exposure.getPsf().
        center : `~lsst.geom.Point2D`
            The centroid position of the source being measured.
        targetPsfModel : `~lsst.afw.detection.GaussianPsf`
            The model GaussianPsf to which the PSF at the source must be
            matched to.
        kernelSum : `float`, optional
            A multipicative factor to apply to the kernel sum.
        basisSigmaGauss: `list` [`float`], optional
            The sigma (in pixels) of the Gaussian in the Alard-Lupton basis set
            used to express the kernel in. This is used only if ``scaleByFwhm``
            is set to False. If it is not provided, then it defaults to
            `config.alardSigGauss`.

        Returns
        -------
        result : `struct`
            - ``psfMatchedExposure`` : the Psf-matched Exposure.
                This has the same parent bbox, wcs as ``exposure``
                and ``targetPsfModel`` as its Psf.
            - ``psfMatchingKernel`` : Psf-matching kernel.
            - ``kernelCellSet`` : SpatialCellSet used to solve for the
                Psf-matching kernel.
            - ``metadata`` : Metadata generated in making Alard-Lupton basis
                set.
        """
        maskedImage = exposure.getMaskedImage()

        result = self._buildCellSet(exposure, center, targetPsfModel)
        kernelCellSet = result.kernelCellSet
        targetPsfModel = result.targetPsfModel
        fwhmScience = exposure.getPsf().computeShape(center).getDeterminantRadius()*sigma2fwhm
        fwhmModel = targetPsfModel.getSigma()*sigma2fwhm  # This works only because it is a `GaussianPsf`.
        self.log.debug("Ratio of GAaP model to science PSF = %f", fwhmModel/fwhmScience)

        basisList = makeKernelBasisList(self.kConfig, fwhmScience, fwhmModel,
                                        basisSigmaGauss=basisSigmaGauss,
                                        metadata=self.metadata)
        spatialSolution, psfMatchingKernel, backgroundModel = self._solve(kernelCellSet, basisList)

        kParameters = np.array(psfMatchingKernel.getKernelParameters())
        kParameters[0] = kernelSum
        psfMatchingKernel.setKernelParameters(kParameters)

        bbox = exposure.getBBox()
        psfMatchedExposure = afwImage.ExposureD(bbox, exposure.getWcs())
        psfMatchedExposure.setPsf(targetPsfModel)

        # Normalize the psf-matching kernel while convolving since its
        # magnitude is meaningless when PSF-matching one model to another.
        kernelImage = afwImage.ImageD(psfMatchingKernel.getDimensions())
        psfMatchingKernel.computeImage(kernelImage, False)

        if self.config.convolutionMethod == "overlap-add":
            # The order of image arrays is important if mode="same", since the
            # returned image array has the same dimensions as the first one.
            psfMatchedImageArray = scipy.signal.oaconvolve(maskedImage.image.array, kernelImage.array,
                                                           mode="same")
        else:
            convolutionMethod = self.config.convolutionMethod
            if convolutionMethod == "auto":
                # Decide if the convolution is faster in real-space or in
                # Fourier space? scipy.signal.convolve uses this under the
                # hood, but we call here for logging purposes.
                convolutionMethod = scipy.signal.choose_conv_method(maskedImage.image.array,
                                                                    kernelImage.array)
                self.log.debug("Using %s method for convolution.", convolutionMethod)

            # The order of image arrays is important if mode="same", since the
            # returned array has the same dimensions as the first argument.
            psfMatchedImageArray = scipy.signal.convolve(maskedImage.image.array, kernelImage.array,
                                                         method=convolutionMethod, mode="same")

        psfMatchedImage = afwImage.ImageD(psfMatchedImageArray)
        psfMatchedExposure.setImage(psfMatchedImage)

        return pipeBase.Struct(psfMatchedExposure=psfMatchedExposure,
                               psfMatchingKernel=psfMatchingKernel,
                               kernelCellSet=kernelCellSet,
                               metadata=self.metadata,
                               )

    def _buildCellSet(self, exposure, center, targetPsfModel) -> pipeBase.Struct:
        """Build a SpatialCellSet with one cell for use with the solve method.

        This builds a SpatialCellSet containing a single cell and a single
        candidate, centered at the location of the source.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            The science exposure that will be convolved; must contain a Psf.
        center : `lsst.geom.Point2D`
            The centroid of the source being measured.
        targetPsfModel : `~lsst.afw.detection.GaussianPsf`
            Psf model to match to.

        Returns
        -------
        result : `struct`
            - ``kernelCellSet`` : a SpatialCellSet to be used by self._solve
            - ``targetPsfModel`` : Validated and/or modified
                target model used to populate the SpatialCellSet

        Notes
        -----
        If the target Psf model and science Psf model have different
        dimensions, adjust the targetPsfModel (the model to which the
        exposure PSF will be matched) to match that of the science Psf.
        If the science Psf dimensions vary across the image,
        as is common with a WarpedPsf, either pad or clip
        (depending on config.padPsf) the dimensions to be constant.
        """
        sizeCellX = self.kConfig.sizeCellX
        sizeCellY = self.kConfig.sizeCellY

        scienceBBox = exposure.getBBox()
        scienceBBox.grow(geom.Extent2I(sizeCellX, sizeCellY))
        sciencePsfModel = exposure.getPsf()
        dimenR = targetPsfModel.getDimensions()

        # Have the size of the region much larger than the bbox, so that
        # the ``kernelCellSet`` has only one instance of `SpatialCell`.
        regionSize = 10*max(scienceBBox.getWidth(), scienceBBox.getHeight())
        kernelCellSet = afwMath.SpatialCellSet(geom.Box2I(scienceBBox), regionSize)

        # Survey the PSF dimensions of the Spatial Cell Set
        # to identify the minimum enclosed or maximum bounding square BBox.
        scienceMI = self._makePsfMaskedImage(sciencePsfModel, center)
        psfWidth, psfHeight = scienceMI.getBBox().getDimensions()
        psfSize = max(psfWidth, psfHeight)

        if self.config.doAutoPadPsf:
            minPsfSize = nextOddInteger(self.kConfig.kernelSize*self.config.autoPadPsfTo)
            paddingPix = max(0, minPsfSize - psfSize)
        else:
            if self.config.padPsfBy % 2 != 0:
                raise ValueError("Config padPsfBy (%i pixels) must be even number." %
                                 self.config.padPsfBy)
            paddingPix = self.config.padPsfBy

        if paddingPix > 0:
            self.log.debug("Padding Science PSF from (%d, %d) to (%d, %d) pixels",
                           psfSize, psfSize, paddingPix + psfSize, paddingPix + psfSize)
            psfSize += paddingPix

        # Check that PSF is larger than the matching kernel.
        maxKernelSize = psfSize - 1
        if maxKernelSize % 2 == 0:
            maxKernelSize -= 1
        if self.kConfig.kernelSize > maxKernelSize:
            message = """
                Kernel size (%d) too big to match Psfs of size %d.
                Please reconfigure by setting one of the following:
                1) kernel size to <= %d
                2) doAutoPadPsf=True
                3) padPsfBy to >= %s
                """ % (self.kConfig.kernelSize, psfSize,
                       maxKernelSize, self.kConfig.kernelSize - maxKernelSize)
            raise ValueError(message)

        dimenS = geom.Extent2I(psfSize, psfSize)

        if (dimenR != dimenS):
            try:
                targetPsfModel = targetPsfModel.resized(psfSize, psfSize)
            except Exception as e:
                self.log.warning("Zero padding or clipping the target PSF model of type %s and dimensions %s"
                                 " to the science Psf dimensions %s because: %s",
                                 targetPsfModel.__class__.__name__, dimenR, dimenS, e)
            dimenR = dimenS

        # Make the target kernel image, at location of science subimage.
        targetMI = self._makePsfMaskedImage(targetPsfModel, center, dimensions=dimenR)

        # Make the kernel image we are going to convolve.
        scienceMI = self._makePsfMaskedImage(sciencePsfModel, center, dimensions=dimenR)

        # The image to convolve is the science image, to the target Psf.
        kc = diffimLib.makeKernelCandidate(center.getX(), center.getY(), scienceMI, targetMI, self.ps)
        kernelCellSet.insertCandidate(kc)

        return pipeBase.Struct(kernelCellSet=kernelCellSet,
                               targetPsfModel=targetPsfModel,
                               )

    def _solve(self, kernelCellSet, basisList):
        """Solve for the PSF matching kernel

        Parameters
        ----------
        kernelCellSet : `~lsst.afw.math.SpatialCellSet`
            A SpatialCellSet to use in determining the matching kernel
            (typically as provided by _buildCellSet).
        basisList : `list` [`~lsst.afw.math.kernel.FixedKernel`]
            A sequence of Kernels to be used in the decomposition of the kernel
            (typically as provided by makeKernelBasisList).

        Returns
        -------
        spatialSolution : `~lsst.ip.diffim.KernelSolution`
            Solution of convolution kernels.
        psfMatchingKernel : `~lsst.afw.math.LinearCombinationKernel`
            Spatially varying Psf-matching kernel.
        backgroundModel : `~lsst.afw.math.Function2D`
            Spatially varying background-matching function.

        Raises
        ------
        RuntimeError
            Raised if unable to determine PSF matching kernel.
        """
        # Visitor for the single kernel fit.
        if self.useRegularization:
            singlekv = diffimLib.BuildSingleKernelVisitorF(basisList, self.ps, self.hMat)
        else:
            singlekv = diffimLib.BuildSingleKernelVisitorF(basisList, self.ps)

        # Visitor for the kernel sum rejection.
        ksv = diffimLib.KernelSumVisitorF(self.ps)

        try:
            # Make sure there are no uninitialized candidates as
            # active occupants of Cell.
            kernelCellSet.visitCandidates(singlekv, 1)

            # Reject outliers in kernel sum.
            ksv.resetKernelSum()
            ksv.setMode(diffimLib.KernelSumVisitorF.AGGREGATE)
            kernelCellSet.visitCandidates(ksv, 1)
            ksv.processKsumDistribution()
            ksv.setMode(diffimLib.KernelSumVisitorF.REJECT)
            kernelCellSet.visitCandidates(ksv, 1)

            regionBBox = kernelCellSet.getBBox()
            spatialkv = diffimLib.BuildSpatialKernelVisitorF(basisList, regionBBox, self.ps)
            kernelCellSet.visitCandidates(spatialkv, 1)
            spatialkv.solveLinearEquation()
            spatialKernel, spatialBackground = spatialkv.getSolutionPair()
            spatialSolution = spatialkv.getKernelSolution()
        except Exception as e:
            self.log.error("ERROR: Unable to calculate psf matching kernel")
            log.getLogger(f"TRACE1.{self.log.name}._solve").debug("%s", e)
            raise e

        self._diagnostic(kernelCellSet, spatialSolution, spatialKernel, spatialBackground)

        return spatialSolution, spatialKernel, spatialBackground

    def _makePsfMaskedImage(self, psfModel, center, dimensions=None) -> afwImage.MaskedImage:
        """Make a MaskedImage of a PSF model of specified dimensions.

        Parameters
        ----------
        psfModel : `~lsst.afw.detection.Psf`
            The PSF model whose image is requested.
        center : `~lsst.geom.Point2D`
            The location at which the PSF image is requested.
        dimensions : `~lsst.geom.Box2I`, optional
            The bounding box of the PSF image.

        Returns
        -------
        kernelIm : `~lsst.afw.image.MaskedImage`
            Image of the PSF.
        """
        rawKernel = psfModel.computeKernelImage(center).convertF()
        if dimensions is None:
            dimensions = rawKernel.getDimensions()
        if rawKernel.getDimensions() == dimensions:
            kernelIm = rawKernel
        else:
            # Make an image of proper size.
            kernelIm = afwImage.ImageF(dimensions)
            bboxToPlace = geom.Box2I(geom.Point2I((dimensions.getX() - rawKernel.getWidth())//2,
                                                  (dimensions.getY() - rawKernel.getHeight())//2),
                                     rawKernel.getDimensions())
            kernelIm.assign(rawKernel, bboxToPlace)

        kernelMask = afwImage.Mask(dimensions, 0x0)
        kernelVar = afwImage.ImageF(dimensions, 1.0)
        return afwImage.MaskedImageF(kernelIm, kernelMask, kernelVar)
