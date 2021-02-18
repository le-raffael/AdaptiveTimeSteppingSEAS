#ifndef RATEANDSTATEBASE_20201026_H
#define RATEANDSTATEBASE_20201026_H

#include "config.h"

#include "form/FacetInfo.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include "mneme/storage.hpp"
#include "mneme/view.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

class RateAndStateBase {
public:
    static constexpr std::size_t TangentialComponents = DomainDimension - 1u;
    constexpr static std::size_t NumQuantitiesCompact = TangentialComponents + 1u;
    constexpr static std::size_t NumQuantitiesExtended = 2 * TangentialComponents + 1u;
    constexpr static std::size_t NumInternalQuantities = 2 + 3 * TangentialComponents;
    
    RateAndStateBase(std::shared_ptr<Curvilinear<DomainDimension>> cl);

    std::size_t block_size() { 
        return (extendedFormulation) ?  
                space_.numBasisFunctions() * NumQuantitiesExtended :
                space_.numBasisFunctions() * NumQuantitiesCompact;
    }
    std::size_t scratch_mem_size() const { return 1; }

    void begin_preparation(std::size_t numFaultFaces);
    void prepare(std::size_t faultNo, FacetInfo const& info, LinearAllocator<double>&);
    void end_preparation() {}

    auto state_prototype(std::size_t numLocalElements) const {
        return FiniteElementFunction<DomainDimension - 1u>(space_.clone(), NumInternalQuantities,
                                                           numLocalElements);
    }

    void setExtendedFormulation(bool is) { extendedFormulation = is; }

    auto const& space() const { return space_; }

protected:
    std::shared_ptr<Curvilinear<DomainDimension>> cl_;

    bool extendedFormulation = false;

    // Basis
    NodalRefElement<DomainDimension - 1u> space_;
    std::vector<Managed<Matrix<double>>> geoE_q;

    struct Coords {
        using type = std::array<double, DomainDimension>;
    };
    using fault_t = mneme::MultiStorage<mneme::DataLayout::SoA, Coords>;
    mneme::StridedView<fault_t> fault_;
};

} // namespace tndm

#endif // RATEANDSTATEBASE_20201026_H
