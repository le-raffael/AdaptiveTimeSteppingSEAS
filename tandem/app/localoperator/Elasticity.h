#ifndef ELASTICITY_20200929_H
#define ELASTICITY_20200929_H

#include "config.h"

#include "form/DGCurvilinearCommon.h"
#include "form/FacetInfo.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include "mneme/storage.hpp"
#include "mneme/view.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility> 
#include <vector>

namespace tndm {

class Elasticity : public DGCurvilinearCommon<DomainDimension> {
public:
    using base = DGCurvilinearCommon<DomainDimension>;
    constexpr static std::size_t Dim = DomainDimension;
    constexpr static std::size_t NumQuantities = DomainDimension;

    Elasticity(std::shared_ptr<Curvilinear<DomainDimension>> cl, functional_t<1> lam,
               functional_t<1> mu);

    std::size_t block_size() const { return space_.numBasisFunctions() * NumQuantities; }

    void begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                           std::size_t numLocalFacets);
    void prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch);
    void prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch);
    void prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch);
    void prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch);

    bool assemble_volume(std::size_t elNo, Matrix<double>& A00,
                         LinearAllocator<double>& scratch) const;
    bool assemble_skeleton(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           Matrix<double>& A01, Matrix<double>& A10, Matrix<double>& A11,
                           LinearAllocator<double>& scratch) const;
    bool assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           LinearAllocator<double>& scratch) const;

    bool rhs_volume(std::size_t elNo, Vector<double>& B, LinearAllocator<double>& scratch) const;
    /**
     * Evaluate the vector b of Au - b = 0 on all facet quadrature points of an element 
     * @param fctNo index of the element among all in the space
     * @param info contains info about the type of the facet (e.g. DBC, fault)
     * @param B0 vector on one side of the facet
     * @param B1 vector on the other side of the facet
     * @param scratch some spare memory
     */
    bool rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      Vector<double>& B1, LinearAllocator<double>& scratch) const;

    /**
     * Evaluate the vector b of Au - b = 0 for the facets facet quadrature points of an element
     * on the fault if the fault is inside the domain (non symmetric setting) 
     * @param fctNo index of the element among all in the space
     * @param info contains info about the type of the facet (e.g. DBC, fault)
     * @param B0 vector on one side of the fault
     * @param B1 vector on the other side of the fault
     * @param scratch some spare memory
     */
    bool rhs_skeleton_only_slip(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      Vector<double>& B1, LinearAllocator<double>& scratch) const;

    /**
     * Evaluate the vector b of Au - b = 0 on all facet quadrature points of an element at 
     * the domain boundary
     * @param fctNo index of the element among all in the space
     * @param info contains info about the type of the facet (e.g. DBC, fault)
     * @param B0 vector on the domain boundary
     * @param scratch some spare memory
     */
    bool rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      LinearAllocator<double>& scratch) const;
    /**
     * Evaluate the vector b of Au - b = 0 for the facets facet quadrature points of an element
     * on the fault if the fault is at the domain boundary (symmetric setting) 
     * @param fctNo index of the element among all in the space
     * @param info contains info about the type of the facet (e.g. DBC, fault)
     * @param B0 vector on the boundary fault 
     * @param scratch some spare memory
     */
    bool rhs_boundary_only_slip(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      LinearAllocator<double>& scratch) const;

    TensorBase<Matrix<double>> tractionResultInfo() const;
    void traction_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                           Vector<double const>& u1, Matrix<double>& result) const;
    void traction_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                           Matrix<double>& result) const;
    void traction_skeleton_onlySlip(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                           Vector<double const>& u1, Matrix<double>& result) const;
    void traction_boundary_onlySlip(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                           Matrix<double>& result) const;
    /**
     * Calculate the derivative of the traction w.r.t. to the displacement if the fault is not symmetric
     * @param fctNo index of the quadrature point
     * @param info needed to implement the boundary conditions
     * @param result store the calculated traction tau
     * */
    void derivative_traction_skeleton(std::size_t fctNo, FacetInfo const& info, Tensor<double,4>& result) const;
 
    /**
     * Calculate the derivative of the traction w.r.t. to the displacement if the fault is symmetric
     * @param fctNo index of the quadrature point
     * @param info needed to implement the boundary conditions
     * @param result store the calculated traction tau
     * */
    void derivative_traction_boundary(std::size_t fctNo, FacetInfo const& info, Tensor<double,4>& result) const;

    FiniteElementFunction<DomainDimension> solution_prototype(std::size_t numLocalElements) const {
        return FiniteElementFunction<DomainDimension>(space_.clone(), NumQuantities,
                                                      numLocalElements);
    }

    FiniteElementFunction<DomainDimension>
    coefficients_prototype(std::size_t numLocalElements) const {
        return FiniteElementFunction<DomainDimension>(materialSpace_.clone(), 2, numLocalElements);
    }
    void coefficients_volume(std::size_t elNo, Matrix<double>& C, LinearAllocator<double>&) const;

    void set_force(functional_t<NumQuantities> fun) {
        fun_force = make_volume_functional(std::move(fun));
    }
    void set_force(volume_functional_t fun) { fun_force = std::move(fun); }
    void set_dirichlet(functional_t<NumQuantities> fun) {
        fun_dirichlet = make_facet_functional(std::move(fun));
    }
    void set_dirichlet(functional_t<NumQuantities> fun,
                       std::array<double, DomainDimension> const& refNormal) {
        fun_dirichlet = make_facet_functional(std::move(fun), refNormal);
    }
    void set_dirichlet(facet_functional_t fun) { fun_dirichlet = std::move(fun); }
    void set_slip(functional_t<NumQuantities> fun,
                  std::array<double, DomainDimension> const& refNormal) {
        fun_slip = make_facet_functional(std::move(fun), refNormal);
    }
    void set_slip(facet_functional_t fun) { fun_slip = std::move(fun); }

    double penalty(FacetInfo const& info) const {
        return std::max(base::penalty[info.up[0]], base::penalty[info.up[1]]);
    }
private:
    bool bc_skeleton(std::size_t fctNo, BC bc, double f_q_raw[]) const;
    bool bc_boundary(std::size_t fctNo, BC bc, double f_q_raw[]) const;

    // Ref elements
    ModalRefElement<DomainDimension> space_;
    NodalRefElement<DomainDimension> materialSpace_;

    // Matrices
    Managed<Matrix<double>> E_Q;
    Managed<Tensor<double, 3u>> Dxi_Q;
    std::vector<Managed<Matrix<double>>> E_q;
    std::vector<Managed<Tensor<double, 3u>>> Dxi_q;

    Managed<Matrix<double>> matE_Q_T;
    Managed<Matrix<double>> matMinv;
    std::vector<Managed<Matrix<double>>> matE_q_T;

    // Input
    volume_functional_t fun_lam;
    volume_functional_t fun_mu;
    volume_functional_t fun_force;
    facet_functional_t fun_dirichlet;
    facet_functional_t fun_slip;

    // Precomputed data
    struct lam {
        using type = double;
    };
    struct mu {
        using type = double;
    };
    struct lam_W_J {
        using type = double;
    };
    struct mu_W_J {
        using type = double;
    };
    struct lam_q_0 {
        using type = double;
    };
    struct mu_q_0 {
        using type = double;
    };
    struct lam_q_1 {
        using type = double;
    };
    struct mu_q_1 {
        using type = double;
    };

    using material_vol_t = mneme::MultiStorage<mneme::DataLayout::SoA, lam, mu>;
    mneme::StridedView<material_vol_t> material;

    using vol_pre_t = mneme::MultiStorage<mneme::DataLayout::SoA, lam_W_J, mu_W_J>;
    mneme::StridedView<vol_pre_t> volPre;

    using fct_pre_t = mneme::MultiStorage<mneme::DataLayout::SoA, lam_q_0, mu_q_0, lam_q_1, mu_q_1>;
    mneme::StridedView<fct_pre_t> fctPre;

    // Options
    constexpr static double epsilon = -1.0;
};

} // namespace tndm

#endif // ELASTICITY_20200929_H
