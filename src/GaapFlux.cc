// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#include "ndarray/eigen.h"

#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/detection/GaussianPsf.h"
#include "lsst/geom/Box.h"
#include "lsst/afw/geom/ellipses/Ellipse.h"
#include "lsst/afw/table/Source.h"
#include "lsst/meas/base/GaussianFlux.h"
#include "lsst/meas/base/SdssShape.h"

#include "lsst/meas/extensions/gaap/GaapFlux.h"

namespace lsst {
namespace meas {
namespace extensions{
namespace gaap {
namespace {
lsst::meas::base::FlagDefinitionList flagDefinitions;
}  // namespace

base::FlagDefinition const GaapFluxAlgorithm::FAILURE = flagDefinitions.addFailureFlag();

base::FlagDefinitionList const& GaapFluxAlgorithm::getFlagDefinitions() { return flagDefinitions; }

GaapFluxAlgorithm::GaapFluxAlgorithm(Control const& ctrl, std::string const& name,
                                             afw::table::Schema& schema)
        : _ctrl(ctrl),
          _instFluxResultKey(base::FluxResultKey::addFields(schema, name, "instFlux from Gaap Flux algorithm")),
          _centroidExtractor(schema, name),
          _shapeExtractor(schema, name) {
    _flagHandler = base::FlagHandler::addFields(schema, name, getFlagDefinitions());
}

void GaapFluxAlgorithm::measure(afw::table::SourceRecord& measRecord,
                                    afw::image::Exposure<float> const& exposure) const {
    geom::Point2D centroid = _centroidExtractor(measRecord, _flagHandler);
    afw::geom::ellipses::Quadrupole measShape = _shapeExtractor(measRecord, _flagHandler);
    /* TODO: DM-27408 will compute the shape for optimality, and pick a shape so that
       shape measurement failure doesn't lead to GAaP flux measurement failure.
    */

    double p = std::static_pointer_cast<const afw::detection::GaussianPsf>(exposure.getPsf())->getSigma();
    double p2 = std::pow(p, 2);
    /*  TODO: Cast the pointer safely in DM-27604.
        If we cannot guarantee that the PSF is Gaussian, it is better to use dynamic_pointer_cast,
        check if it is empty (it will be if the PSF was not a GaussianPsf object) and
        throw an exception
    */

    // TODO: Need to check that this doesn't make Ixx, Iyy negative in DM-27605
    afw::geom::ellipses::Quadrupole shape = afw::geom::ellipses::Quadrupole(measShape.getIxx()-p2, measShape.getIyy()-p2, measShape.getIxy());

    double scaleFactor = std::pow(measShape.getDeterminant()/shape.getDeterminant(), 0.5);

    base::FluxResult result =
            base::SdssShapeAlgorithm::computeFixedMomentsFlux(exposure.getMaskedImage(), shape, centroid);

    result.instFlux *= scaleFactor;
    result.instFluxErr *= scaleFactor;

    measRecord.set(_instFluxResultKey, result);
    // TODO: DM-27088 will estimate the error correctly due to pixel correlations
    _flagHandler.setValue(measRecord, FAILURE.number, false);
}

void GaapFluxAlgorithm::fail(afw::table::SourceRecord& measRecord, base::MeasurementError* error) const {
    _flagHandler.handleFailure(measRecord, error);
}

}  // namespace gaap
}  // namespace extensions
}  // namespace meas
}  // namespace lsst
