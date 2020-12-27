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

#include "kernels/poisson/tensor.h"
#include "kernels/poisson_adapter/kernel.h"
#include "kernels/poisson_adapter/tensor.h"

#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "tandem/SeasAdapterBase.h"
#include "tensor/Managed.h"
#include "tensor/Utility.h"


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
     * Set the function for the boundary functional
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
     * Solve the DG problem, with a unit vector to set up the Jacobian
     * @param state unit vector e 
     * @param result contains A^{-1}e
     */
    template <typename BlockVector> void solveUnitVector(BlockVector& state, BlockVector& result) {
        auto in_handle = state.begin_access_readonly();

        // set the unit vector as slip and transform to quadrature points
        dgop_->lop().set_slip(
            [this, &state, &in_handle](std::size_t fctNo, Matrix<double>& f_q, bool) {
                auto faultNo = this->faultMap_.bndNo(fctNo);
                auto state_block = state.get_block(in_handle, faultNo);
                this->slip(faultNo, state_block, f_q);
            });

        // solve
        linear_solver_.update_rhsOnlySlip(*dgop_);
        linear_solver_.solve();
        state.end_access_readonly(in_handle);

        // transform values in the solution vector x_ from size Nbf to nq
        BlockVector x_nq(nq_, this->numLocalElements(), topo_->comm());
        dgop_->transform_Nbf_to_nq(linear_solver_.x(), x_nq); 

        // extract values on fault and write to solution vector
        auto handleWrite = result.begin_access();
        auto handleRead = x_nq.begin_access_readonly();
        for (int faultNo = 0; faultNo < faultMap_.size(); faultNo++){
            auto fctNo = faultMap_.fctNo(faultNo);
            auto const& info = dgop_->topo().info(fctNo);
            auto u0 = x_nq.get_block(handleRead, info.up[0]);
            auto u1 = x_nq.get_block(handleRead, info.up[1]);
            auto slip = result.get_block(handleWrite, faultNo);
            int nbf = 0.5 * slip.shape()[0];
            // transform uX from quadrature to nodal basis (manual matrix vector product)
            // ?? is e_q the correct transform ?? 
            for (int i = 0; i < nbf; i++){
                for (int  j = 0; j < nq_; j++){
                    slip(i) -= e_q(i,j) * u0(j);                
                    slip(i) -= e_q(i,j) * u1(j);
                }
            }
        }
        result.end_access(handleWrite); 
        x_nq.end_access_readonly(handleRead); 
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
     * terminate access handler for the solution of the linear system
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
     * get number of elements in the mesh
     * @return this
     */
    std::size_t numLocalElements() const { return dgop_->numLocalElements(); }

    /**
     * get the dimensions of the matrix dtau/dU to assemble the Jacobian
     * @return Tensor base of Dtau/DU in one element [nbf, nbf]
     */
        TensorBase<Matrix<double>> getBaseDtauDu(){
            TensorBase<Matrix<double>> tensorBase(poisson_adapter::tensor::dtau_du::Shape[0],
                                          poisson_adapter::tensor::dtau_du::Shape[1]);
            return tensorBase;
        }

    /** 
     * calculate the derivative of the traction w.r.t the displacement
     * @param faultNo index of the fault
     * @param dtau_du result tensor with dimensions (nbf, 2*nbf)
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
