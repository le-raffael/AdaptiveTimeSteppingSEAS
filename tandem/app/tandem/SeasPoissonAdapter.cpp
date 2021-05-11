#include "SeasPoissonAdapter.h"

#include "kernels/poisson/tensor.h"
#include "kernels/poisson_adapter/kernel.h"
#include "kernels/poisson_adapter/tensor.h"

#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "tandem/SeasAdapterBase.h"
#include "tensor/Managed.h"
#include "tensor/Utility.h"

#include <cassert>

namespace tndm {
SeasPoissonAdapter::SeasPoissonAdapter(std::shared_ptr<Curvilinear<Dim>> cl,
                                       std::shared_ptr<DGOperatorTopo> topo,
                                       std::unique_ptr<RefElement<Dim - 1u>> space,
                                       std::unique_ptr<Poisson> local_operator,
                                       std::array<double, Dim> const& up,
                                       std::array<double, Dim> const& ref_normal)
    : SeasAdapterBase(std::move(cl), topo, std::move(space),
                      local_operator->facetQuadratureRule().points(), up, ref_normal),
      dgop_(std::make_unique<DGOperator<Poisson>>(std::move(topo), std::move(local_operator))),
      linear_solver_(*dgop_) {}

void SeasPoissonAdapter::slip(std::size_t faultNo, Vector<double const>& state,
                              Matrix<double>& slip_q) const {
    assert(slip_q.shape(0) == 1);
    assert(slip_q.shape(1) == poisson_adapter::tensor::slip_q::size());
    poisson_adapter::kernel::evaluate_slip krnl;

    krnl.e_q_T = e_q_T.data();
    krnl.slip = state.data();
    krnl.slip_q = slip_q.data();
    krnl.execute();

    /* Slip in the Poisson solver is defined as [[u]] := u^- - u^+.
     * In the friction solver the sign of slip S is flipped, that is, S = -[[u]].
     */
    for (std::size_t i = 0; i < nq_; ++i) {
        if (!fault_[faultNo].template get<SignFlipped>()[i]) {
            slip_q(0, i) = -slip_q(0, i);
        }
    }
}

TensorBase<Matrix<double>> SeasPoissonAdapter::traction_info() const {
    return TensorBase<Matrix<double>>(poisson_adapter::tensor::traction::Shape[0], 2);
}



void SeasPoissonAdapter::traction(std::size_t faultNo, Matrix<double>& traction,
                                  LinearAllocator<double>&) const {
    std::fill(traction.data(), traction.data() + traction.size(), 0.0);

    double grad_u_raw[poisson::tensor::grad_u::Size];
    auto grad_u = Matrix<double>(grad_u_raw, dgop_->lop().tractionResultInfo());
    assert(grad_u.size() == poisson::tensor::grad_u::Size);

    auto fctNo = faultMap_.fctNo(faultNo);
    auto const& info = dgop_->topo().info(fctNo);
    auto u0 = linear_solver_.x().get_block(handle_, info.up[0]);
    auto u1 = linear_solver_.x().get_block(handle_, info.up[1]);
    if (info.up[0] == info.up[1]) {
        dgop_->lop().traction_boundary(fctNo, info, u0, grad_u);
    } else {
        dgop_->lop().traction_skeleton(fctNo, info, u0, u1, grad_u);
    }
    poisson_adapter::kernel::evaluate_traction krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.grad_u = grad_u_raw;
    krnl.minv = minv.data();
    krnl.traction = &traction(0, 1);
    krnl.n_unit_q = fault_[faultNo].template get<UnitNormal>().data()->data();
    krnl.w = dgop_->lop().facetQuadratureRule().weights().data();
    krnl.execute();
}

void SeasPoissonAdapter::traction_onlySlip(std::size_t faultNo, Matrix<double>& traction,
                                  LinearAllocator<double>&) const {
    std::fill(traction.data(), traction.data() + traction.size(), 0.0);

    double grad_u_raw[poisson::tensor::grad_u::Size];
    auto grad_u = Matrix<double>(grad_u_raw, dgop_->lop().tractionResultInfo());
    assert(grad_u.size() == poisson::tensor::grad_u::Size);

    auto fctNo = faultMap_.fctNo(faultNo);
    auto const& info = dgop_->topo().info(fctNo);
    auto u0 = linear_solver_.x().get_block(handle_, info.up[0]);
    auto u1 = linear_solver_.x().get_block(handle_, info.up[1]);
    if (info.up[0] == info.up[1]) {
        dgop_->lop().traction_boundary_onlySlip(fctNo, info, u0, grad_u);
    } else {
        dgop_->lop().traction_skeleton_onlySlip(fctNo, info, u0, u1, grad_u);
    }
    poisson_adapter::kernel::evaluate_traction krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.grad_u = grad_u_raw;
    krnl.minv = minv.data();
    krnl.traction = &traction(0, 1);
    krnl.n_unit_q = fault_[faultNo].template get<UnitNormal>().data()->data();
    krnl.w = dgop_->lop().facetQuadratureRule().weights().data();
    krnl.execute();
}




void SeasPoissonAdapter::dtau_du(std::size_t faultNo, Matrix<double>& dtau_du, 
                                LinearAllocator<double>&) const {

    double Dgrad_u_Du_raw[poisson::tensor::Dgrad_u_Du::Size];
    auto tensorBase =  TensorBase<Tensor3<double>>(poisson::tensor::Dgrad_u_Du::Shape[0], poisson::tensor::Dgrad_u_Du::Shape[1], poisson::tensor::Dgrad_u_Du::Shape[2]);
    auto Dgrad_u_Du = Tensor3<double>(Dgrad_u_Du_raw, tensorBase);

    assert(Dgrad_u_Du.size() == poisson::tensor::Dgrad_u_Du::Size);

    auto fctNo = faultMap_.fctNo(faultNo);
    auto const& info = dgop_->topo().info(fctNo);
    auto u0 = linear_solver_.x().get_block(handle_, info.up[0]);
    auto u1 = linear_solver_.x().get_block(handle_, info.up[1]);
    if (info.up[0] == info.up[1]) {
        dgop_->lop().derivative_traction_boundary(fctNo, info, Dgrad_u_Du);    
    } else {
        dgop_->lop().derivative_traction_skeleton(fctNo, info, Dgrad_u_Du);    
    }
    poisson_adapter::kernel::evaluate_derivative_traction_dU krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.Dgrad_u_Du = Dgrad_u_Du_raw;
    krnl.minv = minv.data();
    krnl.dtau_du = &dtau_du(0, 0);
    krnl.n_unit_q = fault_[faultNo].template get<UnitNormal>().data()->data();
    krnl.w = dgop_->lop().facetQuadratureRule().weights().data();
    krnl.execute(); 
}

void SeasPoissonAdapter::dtau_dS(std::size_t faultNo, Matrix<double>& dtau_dS, 
                                LinearAllocator<double>&) const {

    auto fctNo = faultMap_.fctNo(faultNo);
    auto const& info = dgop_->topo().info(fctNo);
    auto u0 = linear_solver_.x().get_block(handle_, info.up[0]);
    auto u1 = linear_solver_.x().get_block(handle_, info.up[1]);
    poisson_adapter::kernel::evaluate_derivative_traction_dS krnl;
    krnl.e_q_T = e_q_T.data();
    if (info.up[0] == info.up[1]) {
        krnl.c00 = -0.5 * dgop_->lop().penalty(info);
    } else {
        krnl.c00 = -dgop_->lop().penalty(info);
    }
    krnl.minv = minv.data();
    krnl.dtau_dS = &dtau_dS(0, 0);
    krnl.n_unit_q = fault_[faultNo].template get<UnitNormal>().data()->data();
    krnl.w = dgop_->lop().facetQuadratureRule().weights().data();
    krnl.execute(); 
}

} // namespace tndm
