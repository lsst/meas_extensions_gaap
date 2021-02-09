// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2015 AURA/LSST.
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

#ifndef LSST_MEAS_EXTENSIONS_GAAP_H
#define LSST_MEAS_EXTENSIONS_GAAP_H

#include "lsst/pex/config.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/meas/base/Algorithm.h"
#include "lsst/meas/base/FluxUtilities.h"
#include "lsst/meas/base/CentroidUtilities.h"
#include "lsst/meas/base/FlagHandler.h"
#include "lsst/meas/base/InputUtilities.h"
#include "lsst/meas/base/Transform.h"

namespace lsst {
namespace meas {
namespace extensions {
namespace gaap {

/**
 *  @brief A C++ control class to handle GaapFluxAlgorithm's configuration
 */
class GaapFluxControl {
public:


    LSST_CONTROL_FIELD(background, double, "FIXME! NEVER DOCUMENTED!");
    // Undocumented relic from GaussianFlux

    /**
     *  @brief Default constructor
     *
     *  All control classes should define a default constructor that sets all fields to their default values.
     */

    GaapFluxControl() : background(0.0) {}
};

/**
 *  @brief A measurement algorithm that estimates instFlux using an elliptical Gaussian weight.
 *
 *  This algorithm computes instFlux as the dot product of an elliptical Gaussian weight function
 *  with the image.  The size and ellipticity of the weight function are determined using the
 *  SdssShape algorithm, or retreived from a named field.
 */
class GaapFluxAlgorithm : public base::SimpleAlgorithm {
public:
    // Structures and routines to manage flaghandler
    static base::FlagDefinitionList const& getFlagDefinitions();
    static base::FlagDefinition const FAILURE;

    /// A typedef to the Control object for this algorithm, defined above.
    /// The control object contains the configuration parameters for this algorithm.
    typedef GaapFluxControl Control;

    GaapFluxAlgorithm(Control const& ctrl, std::string const& name, afw::table::Schema& schema);

    virtual void measure(afw::table::SourceRecord& measRecord,
                         afw::image::Exposure<float> const& exposure) const;

    virtual void fail(afw::table::SourceRecord& measRecord, base::MeasurementError* error = nullptr) const;
private:
    Control _ctrl;
    base::FluxResultKey _instFluxResultKey;
    base::FlagHandler _flagHandler;
    base::SafeCentroidExtractor _centroidExtractor;
    base::SafeShapeExtractor _shapeExtractor;
};

class GaapFluxTransform : public base::FluxTransform {
public:
    typedef GaapFluxControl Control;
    GaapFluxTransform(Control const& ctrl, std::string const& name, afw::table::SchemaMapper& mapper)
            : base::FluxTransform{name, mapper} {}
};

}  // namespace gaap
}  // namespace extensions
}  // namespace meas
}  // namespace lsst

#endif  // !LSST_MEAS_EXTENSIONS_GAAP_H
