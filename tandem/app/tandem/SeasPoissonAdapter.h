#ifndef SEASPOISSONADAPTER_20201102_H
#define SEASPOISSONADAPTER_20201102_H

#include "common/PetscBlockVector.h"
#include "common/PetscLinearSolver.h"
#include "geometry/Curvilinear.h"
#include "localoperator/Poisson.h"
#include "tandem/SeasAdapterBase.h"

#include "form/BoundaryMap.h"
#include "form/DGOperator.h"
#include "form/DGOperatorTopo.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"
#include "util/LinearAllocator.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

namespace tndm {

class SeasPoissonAdapter : public SeasAdapterBase {
public:
    using local_operator_t = Poisson;
    constexpr static std::size_t Dim = local_operator_t::Dim;
    constexpr static std::size_t NumQuantities = local_operator_t::NumQuantities;
    using time_functional_t =
        std::function<std::array<double, NumQuantities>(std::array<double, Dim + 1u> const&)>;

    /**
     * Constructor (most is done in the constructor of the AdapterBase class)
     * @param cl describes mesh and basis functions
     * @param topo contains metadata about the mesh
     * @param space evaluates integrals on one element
     * @param local_operator solves DG problem (together with the three previous params)
     * @param up some characteristic of the fault elements
     * @param ref_normal normal vector on each node 
     */
    SeasPoissonAdapter(std::shared_ptr<Curvilinear<Dim>> cl, std::shared_ptr<DGOperatorTopo> topo,
                       std::unique_ptr<RefElement<Dim - 1u>> space,
                       std::unique_ptr<local_operator_t> local_operator,
                       std::array<double, Dim> const& up,
                       std::array<double, Dim> const& ref_normal);

    /** 
     * Sette function for the boundary functional
     * @param fun functional to be set
     */
    void set_boundary(time_functional_t fun) { fun_boundary = std::move(fun); }

    /**
     * Solve the DG problem, result is kept in the linear solver object
     * @param time current simulation time
     * @param state current solution vector 
     */
    template <typename BlockVector> void solve(double time, BlockVector& state) {
        auto in_handle = state.begin_access_readonly();
        dgop_->lop().set_slip(
            [this, &state, &in_handle](std::size_t fctNo, Matrix<double>& f_q, bool) {
                auto faultNo = this->faultMap_.bndNo(fctNo);
                auto state_block = state.get_block(in_handle, faultNo);
                this->slip(faultNo, state_block, f_q);
            });
        dgop_->lop().set_dirichlet(
            [this, time](std::array<double, Dim> const& x) {
                std::array<double, Dim + 1u> xt;
                std::copy(x.begin(), x.end(), xt.begin());
                xt.back() = time;
                return this->fun_boundary(xt);
            },
            ref_normal_);
        linear_solver_.update_rhs(*dgop_);
        linear_solver_.solve();
        state.end_access_readonly(in_handle);
    }

    /**
     * prepares the tensor of the traction
     * @return the tensor has nbf rows (number of base functions in one fault element) and 2 columns (sigma=0, tau)
     */
    TensorBase<Matrix<double>> traction_info() const;

    /**
     * set the slip function in the DG solver (not executed yet, functional updated for later, called by the fault index)
     * @param state_access handler to safely access the state vector
     */
    template <class Func> void begin_traction(Func state_access) {
        handle_ = linear_solver_.x().begin_access_readonly();
        dgop_->lop().set_slip([this, state_access](std::size_t fctNo, Matrix<double>& f_q, bool) {
            auto faultNo = this->faultMap_.bndNo(fctNo);
            auto state_block = state_access(faultNo);
            this->slip(faultNo, state_block, f_q);
        });
    }

    /**
     * Calculates the traction for one fault element
     * @param faultNo index of the considered fault element
     * @param traction  result matrix with the calculated traction (cols: nbf, rows: (sigma=0,tau))
     * @param . this scratch thingy
     */
    void traction(std::size_t faultNo, Matrix<double>& traction, LinearAllocator<double>&) const;

    /**
     * terminate access handler for the state vector
     */
    void end_traction() { linear_solver_.x().end_access_readonly(handle_); }

    /**
     * get the current displacement as locally present in the DG solver
     * @return displacement in finite element format
     */
    auto displacement() const { return dgop_->solution(linear_solver_.x()); }

    /**
     * get the current solution vector in the linear solver (contains U)
     * @return block vector with current solution
     */
    auto& getSolutionLinearSystem(){ return linear_solver_.x();}

    /**
     * get number of fault elements
     * @return this
     */
    std::size_t numLocalElements() const { return dgop_->numLocalElements(); }

    /** 
     * calculate the derivative of the traction w.r.t the displacement
     * @param faultNo index if the fault
     * @param dtau_du result tensor with dimensions (nbf, nbf)
     * @param . this scratch thingy
     */
    void dtau_du(std::size_t faultNo, Matrix<double>& dtau_du, LinearAllocator<double>&) const;

private:
    /**
    * transform the slip from the nodes to the quadrature points 
    * @param faultNo index of the current element
    * @param state current solution vector on the element (first nbf elements contain the slip in nodal representation)
    * @param s_q result vector with slip at the quadrature points [1, nq]
    */
    void slip(std::size_t faultNo, Vector<double const>& state, Matrix<double>& s_q) const;

    std::unique_ptr<DGOperator<local_operator_t>> dgop_;
    PetscLinearSolver linear_solver_;

    time_functional_t fun_boundary =
        [](std::array<double, Dim + 1u> const& x) -> std::array<double, NumQuantities> {
        return {};
    };
    PetscBlockVector::const_handle handle_;
};

} // namespace tndm

#endif // SEASPOISSONADAPTER_20201102_H
