#include "Elasticity.h"
#include "config.h"
#include "kernels/elasticity/init.h"
#include "kernels/elasticity/kernel.h"
#include "kernels/elasticity/tensor.h"

#include "basis/WarpAndBlend.h"
#include "form/BC.h"
#include "form/DGCurvilinearCommon.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "util/LinearAllocator.h"

#include <cassert>

namespace tensor = tndm::elasticity::tensor;
namespace init = tndm::elasticity::init;
namespace kernel = tndm::elasticity::kernel;

namespace tndm {
 
Elasticity::Elasticity(std::shared_ptr<Curvilinear<DomainDimension>> cl, functional_t<1> lam,
                       functional_t<1> mu)
    : DGCurvilinearCommon<DomainDimension>(std::move(cl), MinQuadOrder()), space_(PolynomialDegree),
      materialSpace_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>()),
      fun_lam(make_volume_functional(std::move(lam))),
      fun_mu(make_volume_functional(std::move(mu))), fun_force(zero_volume_function),
      fun_dirichlet(zero_facet_function), fun_slip(zero_facet_function) {

    E_Q = space_.evaluateBasisAt(volRule.points());
    Dxi_Q = space_.evaluateGradientAt(volRule.points());
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto points = cl_->facetParam(f, fctRule.points());
        E_q.emplace_back(space_.evaluateBasisAt(points));
        Dxi_q.emplace_back(space_.evaluateGradientAt(points));
        matE_q_T.emplace_back(materialSpace_.evaluateBasisAt(points, {1, 0}));
    }

    matMinv = materialSpace_.inverseMassMatrix();
    matE_Q_T = materialSpace_.evaluateBasisAt(volRule.points(), {1, 0});
}

void Elasticity::begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                                   std::size_t numLocalFacets) {
    base::begin_preparation(numElements, numLocalElements, numLocalFacets);

    material.setStorage(
        std::make_shared<material_vol_t>(numElements * materialSpace_.numBasisFunctions()), 0u,
        numElements, materialSpace_.numBasisFunctions());

    volPre.setStorage(std::make_shared<vol_pre_t>(numElements * volRule.size()), 0u, numElements,
                      volRule.size());

    fctPre.setStorage(std::make_shared<fct_pre_t>(numLocalFacets * fctRule.size()), 0u,
                      numLocalFacets, fctRule.size());
}

void prepare_skeleton(std::size_t fctNo, FacetInfo const& info, LinearAllocator<double>& scratch) {}
void prepare_boundary(std::size_t fctNo, FacetInfo const& info, LinearAllocator<double>& scratch) {}

void Elasticity::prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume(elNo, scratch);

    double lam_Q_raw[tensor::lam_Q::size()];
    auto lam_Q = Matrix<double>(lam_Q_raw, 1, volRule.size());
    fun_lam(elNo, lam_Q);

    double mu_Q_raw[tensor::mu_Q::size()];
    auto mu_Q = Matrix<double>(mu_Q_raw, 1, volRule.size());
    fun_mu(elNo, mu_Q);

    kernel::project_material krnl;
    krnl.matE_Q_T = matE_Q_T.data();
    krnl.lam = material[elNo].get<lam>().data();
    krnl.lam_Q = lam_Q_raw;
    krnl.mu = material[elNo].get<mu>().data();
    krnl.mu_Q = mu_Q_raw;
    krnl.W = volRule.weights().data();
    krnl.matMinv = matMinv.data();
    krnl.execute();
}
void Elasticity::prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                                  LinearAllocator<double>& scratch) {
    base::prepare_skeleton(fctNo, info, scratch);

    kernel::precomputeSurface krnl;
    for (unsigned side = 0; side < 2; ++side) {
        krnl.matE_q_T(side) = matE_q_T[info.localNo[side]].data();
    }
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.mu_q(1) = fctPre[fctNo].get<mu_q_1>().data();

    for (unsigned side = 0; side < 2; ++side) {
        krnl.lam = material[info.up[side]].get<lam>().data();
        krnl.mu = material[info.up[side]].get<mu>().data();
        krnl.execute(side);
    }
}
void Elasticity::prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                                  LinearAllocator<double>& scratch) {
    base::prepare_boundary(fctNo, info, scratch);

    kernel::precomputeSurface krnl;
    krnl.matE_q_T(0) = matE_q_T[info.localNo[0]].data();
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.lam = material[info.up[0]].get<lam>().data();
    krnl.mu = material[info.up[0]].get<mu>().data();
    krnl.execute(0);
}

void Elasticity::prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume_post_skeleton(elNo, scratch);

    auto lam_field = material[elNo].get<lam>();
    auto mu_field = material[elNo].get<mu>();

    kernel::precomputeVolume krnl_pre;
    krnl_pre.matE_Q_T = matE_Q_T.data();
    krnl_pre.J = vol[elNo].template get<AbsDetJ>().data();
    krnl_pre.W = volRule.weights().data();
    krnl_pre.lam = lam_field.data();
    krnl_pre.lam_W_J = volPre[elNo].template get<lam_W_J>().data();
    krnl_pre.mu = mu_field.data();
    krnl_pre.mu_W_J = volPre[elNo].template get<mu_W_J>().data();
    krnl_pre.execute();

    assert(lam_field.size() == mu_field.size());
    double max_mat = std::numeric_limits<double>::lowest();
    for (std::size_t i = 0, n = lam_field.size(); i < n; ++i) {
        max_mat = std::max(max_mat, lam_field[i] + 2.0 * mu_field[i]);
    }

    base::penalty[elNo] *=
        max_mat * (PolynomialDegree + 1) * (PolynomialDegree + DomainDimension) / DomainDimension;
}

bool Elasticity::assemble_volume(std::size_t elNo, Matrix<double>& A00,
                                 LinearAllocator<double>& scratch) const {
    double Dx_Q[tensor::Dx_Q::size()];

    assert(volRule.size() == tensor::W::Shape[0]);
    assert(Dxi_Q.shape(0) == tensor::Dxi_Q::Shape[0]);
    assert(Dxi_Q.shape(1) == tensor::Dxi_Q::Shape[1]);
    assert(Dxi_Q.shape(2) == tensor::Dxi_Q::Shape[2]);

    kernel::Dx_Q dxKrnl;
    dxKrnl.Dx_Q = Dx_Q;
    dxKrnl.Dxi_Q = Dxi_Q.data();
    dxKrnl.G = vol[elNo].get<JInv>().data()->data();
    dxKrnl.execute();

    kernel::assembleVolume krnl;
    krnl.A = A00.data();
    krnl.delta = init::delta::Values;
    krnl.Dx_Q = Dx_Q;
    krnl.lam_W_J = volPre[elNo].get<lam_W_J>().data();
    krnl.mu_W_J = volPre[elNo].get<mu_W_J>().data();
    krnl.execute();
    return true;
}

bool Elasticity::assemble_skeleton(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                   Matrix<double>& A01, Matrix<double>& A10, Matrix<double>& A11,
                                   LinearAllocator<double>& scratch) const {
    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::E_q::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::E_q::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::Dxi_q::Shape[0][0]);
    assert(Dxi_q[0].shape(1) == tensor::Dxi_q::Shape[0][1]);
    assert(Dxi_q[0].shape(2) == tensor::Dxi_q::Shape[0][2]);

    double Dx_q0[tensor::Dx_q::size(0)];
    double Dx_q1[tensor::Dx_q::size(1)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.Dx_q(1) = Dx_q1;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.g(1) = fct[fctNo].get<JInv1>().data()->data();
    for (unsigned side = 0; side < 2; ++side) {
        dxKrnl.Dxi_q(side) = Dxi_q[info.localNo[side]].data();
        dxKrnl.execute(side);
    }

    double traction_op_q0[tensor::traction_op_q::size(0)];
    double traction_op_q1[tensor::traction_op_q::size(1)];

    kernel::assembleTractionOp tOpKrnl;
    tOpKrnl.delta = init::delta::Values;
    tOpKrnl.Dx_q(0) = Dx_q0;
    tOpKrnl.Dx_q(1) = Dx_q1;
    tOpKrnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    tOpKrnl.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
    tOpKrnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    tOpKrnl.mu_q(1) = fctPre[fctNo].get<mu_q_1>().data();
    tOpKrnl.n_q = fct[fctNo].get<Normal>().data()->data();
    tOpKrnl.traction_op_q(0) = traction_op_q0;
    tOpKrnl.traction_op_q(1) = traction_op_q1;
    tOpKrnl.execute(0);
    tOpKrnl.execute(1);

    kernel::assembleSurface krnl;
    krnl.c00 = -0.5;
    krnl.c01 = -krnl.c00;
    krnl.c10 = epsilon * 0.5;
    krnl.c11 = -krnl.c10;
    krnl.c20 = penalty(info);
    krnl.c21 = -krnl.c20;
    krnl.a(0, 0) = A00.data();
    krnl.a(0, 1) = A01.data();
    krnl.a(1, 0) = A10.data();
    krnl.a(1, 1) = A11.data();
    krnl.delta = init::delta::Values;
    for (unsigned side = 0; side < 2; ++side) {
        krnl.E_q(side) = E_q[info.localNo[side]].data();
    }
    krnl.nl_q = fct[fctNo].get<NormalLength>().data();
    krnl.traction_op_q(0) = traction_op_q0;
    krnl.traction_op_q(1) = traction_op_q1;
    krnl.w = fctRule.weights().data();
    krnl.execute(0, 0);
    krnl.execute(0, 1);
    krnl.execute(1, 0);
    krnl.execute(1, 1);

    return true;
}

bool Elasticity::assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                   LinearAllocator<double>& scratch) const {
    if (info.bc == BC::Natural) {
        return false;
    }

    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::E_q::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::E_q::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::Dxi_q::Shape[0][0]);
    assert(Dxi_q[0].shape(1) == tensor::Dxi_q::Shape[0][1]);
    assert(Dxi_q[0].shape(2) == tensor::Dxi_q::Shape[0][2]);

    double Dx_q0[tensor::Dx_q::size(0)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    dxKrnl.execute(0);

    double traction_op_q0[tensor::traction_op_q::size(0)];

    kernel::assembleTractionOp tOpKrnl;
    tOpKrnl.delta = init::delta::Values;
    tOpKrnl.Dx_q(0) = Dx_q0;
    tOpKrnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    tOpKrnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    tOpKrnl.n_q = fct[fctNo].get<Normal>().data()->data();
    tOpKrnl.traction_op_q(0) = traction_op_q0;
    tOpKrnl.execute(0);

    kernel::assembleSurface krnl;
    krnl.c00 = -1.0;
    krnl.c10 = epsilon;
    krnl.c20 = penalty(info);
    krnl.a(0, 0) = A00.data();
    krnl.delta = init::delta::Values;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.nl_q = fct[fctNo].get<NormalLength>().data();
    krnl.traction_op_q(0) = traction_op_q0;
    krnl.w = fctRule.weights().data();
    krnl.execute(0, 0);

    return true;
}

bool Elasticity::rhs_volume(std::size_t elNo, Vector<double>& B,
                            LinearAllocator<double>& scratch) const {
    assert(tensor::b::Shape[0] == tensor::A::Shape[0]);

    double F_Q_raw[tensor::F_Q::size()];
    assert(tensor::F_Q::Shape[1] == volRule.size());

    auto F_Q = Matrix<double>(F_Q_raw, NumQuantities, volRule.size());
    fun_force(elNo, F_Q);

    kernel::rhsVolume rhs;
    rhs.E_Q = E_Q.data();
    rhs.F_Q = F_Q_raw;
    rhs.J = vol[elNo].get<AbsDetJ>().data();
    rhs.W = volRule.weights().data();
    rhs.b = B.data();
    rhs.execute();
    return true;
}

bool Elasticity::bc_skeleton(std::size_t fctNo, BC bc, double f_q_raw[]) const {
    assert(tensor::f_q::Shape[1] == fctRule.size());
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    if (bc == BC::Fault) {
        fun_slip(fctNo, f_q, false);
    } else if (bc == BC::Dirichlet) {
        fun_dirichlet(fctNo, f_q, false);
    } else {
        return false;
    }
    return true;
}

bool Elasticity::bc_boundary(std::size_t fctNo, BC bc, double f_q_raw[]) const {
    assert(tensor::f_q::Shape[1] == fctRule.size());
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    if (bc == BC::Fault) {
        fun_slip(fctNo, f_q, true);
        for (std::size_t q = 0; q < tensor::f_q::Shape[1]; ++q) {
            for (std::size_t p = 0; p < NumQuantities; ++p) {
                f_q(p, q) *= 0.5;
            }
        }
    } else if (bc == BC::Dirichlet) {
        fun_dirichlet(fctNo, f_q, true);
    } else {
        return false;
    }
    return true;
}

bool Elasticity::rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                              Vector<double>& B1, LinearAllocator<double>& scratch) const {
    double Dx_q[tensor::Dx_q::size(0)];
    double f_q_raw[tensor::f_q::size()];
    if (!bc_skeleton(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = 0.5 * epsilon;
    rhs.c20 = penalty(info);
    rhs.Dx_q(0) = Dx_q;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.f_q = f_q_raw;
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    rhs.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.nl_q = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.execute();

    rhs.b = B1.data();
    rhs.c20 *= -1.0;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[1]].data();
    rhs.E_q(0) = E_q[info.localNo[1]].data();
    rhs.g(0) = fct[fctNo].get<JInv1>().data()->data();
    rhs.lam_q(0) = fctPre[fctNo].get<lam_q_1>().data();
    rhs.mu_q(0) = fctPre[fctNo].get<mu_q_1>().data();
    rhs.execute();

    return true;
}

bool Elasticity::rhs_skeleton_only_slip(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                           Vector<double>& B1, LinearAllocator<double>& scratch) const {
    double Dx_q[tensor::Dx_q::size(0)];
    double f_q_raw[tensor::f_q::size()];
    assert(tensor::f_q::Shape[1] == fctRule.size());
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    if (info.bc == BC::Fault) {
        fun_slip(fctNo, f_q, false);
    } else {
        return false;
    }

    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = 0.5 * epsilon;
    rhs.c20 = penalty(info);
    rhs.Dx_q(0) = Dx_q;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.f_q = f_q_raw;
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    rhs.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.nl_q = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.execute();

    rhs.b = B1.data();
    rhs.c20 *= -1.0;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[1]].data();
    rhs.E_q(0) = E_q[info.localNo[1]].data();
    rhs.g(0) = fct[fctNo].get<JInv1>().data()->data();
    rhs.lam_q(0) = fctPre[fctNo].get<lam_q_1>().data();
    rhs.mu_q(0) = fctPre[fctNo].get<mu_q_1>().data();
    rhs.execute();

    return true;
}


bool Elasticity::rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                              LinearAllocator<double>& scratch) const {
    double Dx_q[tensor::Dx_q::size(0)];
    double f_q_raw[tensor::f_q::size()];
    if (!bc_boundary(fctNo, info.bc, f_q_raw)) {
        return false;
    }
    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = epsilon;
    rhs.c20 = penalty(info);
    rhs.Dx_q(0) = Dx_q;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.f_q = f_q_raw;
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    rhs.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.nl_q = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.execute();

    return true;
}

bool Elasticity::rhs_boundary_only_slip(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                           LinearAllocator<double>& scratch) const {

    double Dx_q[tensor::Dx_q::size(0)];
    double f_q_raw[tensor::f_q::size()];

    assert(tensor::f_q::Shape[1] == fctRule.size());
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    if (info.bc == BC::Fault) {
        fun_slip(fctNo, f_q, true);
        for (std::size_t q = 0; q < tensor::f_q::Shape[1]; ++q) {
            for (std::size_t p = 0; p < NumQuantities; ++p) {
                f_q(p, q) *= 0.5;
            }
        }
    } else {
        return false;
    }

    //  calculate b only for the slip at the fault
    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = epsilon;
    rhs.c20 = penalty(info);
    rhs.Dx_q(0) = Dx_q;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.f_q = f_q_raw;
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    rhs.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.nl_q = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.execute();

    return true;
}

void Elasticity::coefficients_volume(std::size_t elNo, Matrix<double>& C,
                                     LinearAllocator<double>&) const {
    auto const coeff_lam = material[elNo].get<lam>();
    auto const coeff_mu = material[elNo].get<mu>();
    assert(coeff_lam.size() == C.shape(0));
    assert(coeff_mu.size() == C.shape(0));
    assert(2 == C.shape(1));
    for (std::size_t i = 0; i < C.shape(0); ++i) {
        C(i, 0) = coeff_lam[i];
        C(i, 1) = coeff_mu[i];
    }
}

TensorBase<Matrix<double>> Elasticity::tractionResultInfo() const {
    return TensorBase<Matrix<double>>(tensor::traction_q::Shape[0], tensor::traction_q::Shape[1]);
}

void Elasticity::traction_skeleton(std::size_t fctNo, FacetInfo const& info,
                                   Vector<double const>& u0, Vector<double const>& u1,
                                   Matrix<double>& result) const {
    assert(result.size() == tensor::traction_q::size());
    assert(u0.size() == tensor::u::size(0));
    assert(u1.size() == tensor::u::size(1));

    double Dx_q0[tensor::Dx_q::size(0)];
    double Dx_q1[tensor::Dx_q::size(1)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.Dx_q(1) = Dx_q1;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.g(1) = fct[fctNo].get<JInv1>().data()->data();
    for (unsigned side = 0; side < 2; ++side) {
        dxKrnl.Dxi_q(side) = Dxi_q[info.localNo[side]].data();
        dxKrnl.execute(side);
    }

    double f_q_raw[tensor::f_q::size()];
    bc_skeleton(fctNo, info.bc, f_q_raw);

    kernel::compute_traction krnl;
    krnl.c00 = -penalty(info);
    krnl.Dx_q(0) = Dx_q0;
    krnl.Dx_q(1) = Dx_q1;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.E_q(1) = E_q[info.localNo[1]].data();
    krnl.f_q = f_q_raw;
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.mu_q(1) = fctPre[fctNo].get<mu_q_1>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.traction_q = result.data();
    krnl.u(0) = u0.data();
    krnl.u(1) = u1.data();
    krnl.execute();
}

void Elasticity::traction_boundary(std::size_t fctNo, FacetInfo const& info,
                                   Vector<double const>& u0, Matrix<double>& result) const {

    assert(result.size() == tensor::traction_q::size());
    assert(u0.size() == tensor::u::size(0));

    double Dx_q0[tensor::Dx_q::size(0)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    dxKrnl.execute(0);

    double f_q_raw[tensor::f_q::size()];
    bc_boundary(fctNo, info.bc, f_q_raw);

    kernel::compute_traction_bnd krnl;
    krnl.c00 = -penalty(info);
    krnl.Dx_q(0) = Dx_q0;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.f_q = f_q_raw;
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.traction_q = result.data();
    krnl.u(0) = u0.data();
    krnl.execute();
}


void Elasticity::traction_skeleton_onlySlip(std::size_t fctNo, FacetInfo const& info,
                                   Vector<double const>& u0, Vector<double const>& u1,
                                   Matrix<double>& result) const {
    assert(result.size() == tensor::traction_q::size());
    assert(u0.size() == tensor::u::size(0));
    assert(u1.size() == tensor::u::size(1));

    double Dx_q0[tensor::Dx_q::size(0)];
    double Dx_q1[tensor::Dx_q::size(1)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.Dx_q(1) = Dx_q1;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.g(1) = fct[fctNo].get<JInv1>().data()->data();
    for (unsigned side = 0; side < 2; ++side) {
        dxKrnl.Dxi_q(side) = Dxi_q[info.localNo[side]].data();
        dxKrnl.execute(side);
    }

    double f_q_raw[tensor::f_q::size()];
    assert(tensor::f_q::Shape[1] == fctRule.size());
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    if (info.bc == BC::Fault) {
        fun_slip(fctNo, f_q, false);
    } else {
        for(std::size_t i = 0; i < tensor::f_q::size(); i++) f_q_raw[i] = 0.0;
    }

    kernel::compute_traction krnl;
    krnl.c00 = -penalty(info);
    krnl.Dx_q(0) = Dx_q0;
    krnl.Dx_q(1) = Dx_q1;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.E_q(1) = E_q[info.localNo[1]].data();
    krnl.f_q = f_q_raw;
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.mu_q(1) = fctPre[fctNo].get<mu_q_1>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.traction_q = result.data();
    krnl.u(0) = u0.data();
    krnl.u(1) = u1.data();
    krnl.execute();
}

void Elasticity::traction_boundary_onlySlip(std::size_t fctNo, FacetInfo const& info,
                                   Vector<double const>& u0, Matrix<double>& result) const {

    assert(result.size() == tensor::traction_q::size());
    assert(u0.size() == tensor::u::size(0));

    double Dx_q0[tensor::Dx_q::size(0)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    dxKrnl.execute(0);

    double f_q_raw[tensor::f_q::size()];
    assert(tensor::f_q::Shape[1] == fctRule.size());
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    if (info.bc == BC::Fault) {
        fun_slip(fctNo, f_q, true);
        for (std::size_t q = 0; q < tensor::f_q::Shape[1]; ++q) {
            for (std::size_t p = 0; p < NumQuantities; ++p) {
                f_q(p, q) *= 0.5;
            }
        }
    } else {
        for(std::size_t i = 0; i < tensor::f_q::size(); i++) f_q_raw[i] = 0.0;
    }

    kernel::compute_traction_bnd krnl;
    krnl.c00 = -penalty(info);
    krnl.Dx_q(0) = Dx_q0;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.f_q = f_q_raw;
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.traction_q = result.data();
    krnl.u(0) = u0.data();
    krnl.execute();
}


void Elasticity::derivative_traction_skeleton(std::size_t fctNo, FacetInfo const& info, Tensor<double,4>& result) const {
    assert(result.size() == tensor::D_traction_q_Du::size());

    double Dx_q0[tensor::Dx_q::size(0)];
    double Dx_q1[tensor::Dx_q::size(1)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.Dx_q(1) = Dx_q1;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.g(1) = fct[fctNo].get<JInv1>().data()->data();
    for (unsigned side = 0; side < 2; ++side) {
        dxKrnl.Dxi_q(side) = Dxi_q[info.localNo[side]].data();
        dxKrnl.execute(side);
    }

    kernel::D_traction_q_Du krnl;
    krnl.c00 = -penalty(info);
    krnl.delta = init::delta::Values;
    krnl.Dx_q(0) = Dx_q0;
    krnl.Dx_q(1) = Dx_q1;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.E_q(1) = E_q[info.localNo[1]].data();
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.mu_q(1) = fctPre[fctNo].get<mu_q_1>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.D_traction_q_Du = result.data();
    krnl.execute();
}


void Elasticity::derivative_traction_boundary(std::size_t fctNo, FacetInfo const& info, Tensor<double,4>& result) const {
    assert(result.size() == tensor::traction_q::size());

    double Dx_q0[tensor::Dx_q::size(0)];

    kernel::Dx_q dxKrnl;
    dxKrnl.Dx_q(0) = Dx_q0;
    dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    dxKrnl.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    dxKrnl.execute(0);

    kernel::D_traction_q_Du_bnd krnl;
    krnl.c00 = -penalty(info);
    krnl.delta = init::delta::Values;
    krnl.Dx_q(0) = Dx_q0;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl.mu_q(0) = fctPre[fctNo].get<mu_q_0>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.D_traction_q_Du = result.data();
    krnl.execute();
}


} // namespace tndm
