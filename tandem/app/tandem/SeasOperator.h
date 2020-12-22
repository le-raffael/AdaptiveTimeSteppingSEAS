#ifndef SEASOPERATOR_20201001_H
#define SEASOPERATOR_20201001_H

#include "form/BoundaryMap.h"
#include "geometry/Curvilinear.h"

#include "tandem/RateAndStateBase.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace tndm {

template <typename LocalOperator, typename SeasAdapter> class SeasOperator {
public:

    constexpr static std::size_t Dim = SeasAdapter::Dim;

    constexpr static std::size_t NumQuantities = SeasAdapter::NumQuantities;

    using time_functional_t = typename SeasAdapter::time_functional_t;

    /**
     * Creator function, prepare all operators
     * @param localOperator method to solve the fault (e.g. rate and state)
     * @param seas_adapter handler for the space domain (e.g. discontinuous Galerkin with Poisson/Elasticity solver)
     */
    SeasOperator(std::unique_ptr<LocalOperator> localOperator,
                 std::unique_ptr<SeasAdapter> seas_adapter)
        : lop_(std::move(localOperator)), adapter_(std::move(seas_adapter)) {
        scratch_size_ = lop_->scratch_mem_size();
        scratch_size_ += adapter_->scratch_mem_size();
        scratch_mem_ = std::make_unique<double[]>(scratch_size_);

        auto scratch = make_scratch();
        adapter_->begin_preparation(numLocalElements());
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->prepare(faultNo, scratch);
        }
        adapter_->end_preparation();

        lop_->begin_preparation(numLocalElements());
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto fctNo = adapter_->faultMap().fctNo(faultNo);
            lop_->prepare(faultNo, adapter_->topo().info(fctNo), scratch);
        }
        lop_->end_preparation();
    }

    /**
     * Initialize the system 
     *  - apply initial conditions on the local operator
     *  - solve the system once
     * @param vector solution vector to be initialized (has a block vector format)
     */
    template <class BlockVector> void initial_condition(BlockVector& vector) {
        auto scratch = make_scratch();
        auto access_handle = vector.begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto B = vector.get_block(access_handle, faultNo);
            lop_->pre_init(faultNo, B, scratch);
        }
        vector.end_access(access_handle);

        adapter_->solve(0.0, vector);

        access_handle = vector.begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&vector, &access_handle](std::size_t faultNo) {
            return vector.get_block(const_cast<typename BlockVector::const_handle>(access_handle),
                                    faultNo);
        });
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto B = vector.get_block(access_handle, faultNo);
            lop_->init(faultNo, traction, B, scratch);
        }
        adapter_->end_traction();
        vector.end_access(access_handle);
    }

    /**
     * Solve the system for a given timestep
     *  - first solve the DG problem
     *  - then solve the ODE in the rate and state problem
     * @param time current simulation time
     * @param state current solution vector
     * @param result next solution vector (to be filled in the execution of the function)
     */
    template <typename BlockVector> void rhs(double time, BlockVector& state, BlockVector& result) {
        adapter_->solve(time, state);

        auto scratch = make_scratch();
        auto in_handle = state.begin_access_readonly();
        auto out_handle = result.begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto state_block = state.get_block(in_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            double VMax = lop_->rhs(faultNo, time, traction, state_block, result_block, scratch);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        evaluation_rhs_count++;
    }

    /**
     * write solution vector to Finite difference format (see state() in rate and state)
     * @param solution vector in block format
     * @return finite difference object with the solution
     */
    template <typename BlockVector> auto state(BlockVector& vector) {
        auto soln = lop_->state_prototype(numLocalElements());
        auto& values = soln.values();

        auto scratch = make_scratch();
        auto in_handle = vector.begin_access_readonly();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&vector, &in_handle](std::size_t faultNo) {
            return vector.get_block(in_handle, faultNo);
        });
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto value_matrix = values.subtensor(slice{}, slice{}, faultNo);
            auto state_block = vector.get_block(in_handle, faultNo);
            lop_->state(faultNo, traction, state_block, value_matrix, scratch);
        }
        adapter_->end_traction();
        vector.end_access_readonly(in_handle);
        return soln;
    }

    /**
     * see SeasXXXXAdapter -> set_boundary
     * @param fun functional to be used at the boundary
     */
    void set_boundary(time_functional_t fun) { adapter_->set_boundary(std::move(fun)); }

    /**
     * Puts the counter of the number of evaluations of the right-hand side to zero
     */
    void reset_rhs_count() { evaluation_rhs_count = 0; };

    /**
     *  Getter functions
     */
    std::size_t block_size() const { return lop_->block_size(); }
    
    MPI_Comm comm() const { return adapter_->topo().comm(); }
    
    BoundaryMap const& faultMap() const { return adapter_->faultMap(); }
    
    std::size_t numLocalElements() const { return adapter_->faultMap().size(); }

    SeasAdapter& adapter() const { return *adapter_; }

    size_t rhs_count() {return evaluation_rhs_count; };

    double VMax() const { return VMax_; }
    
    LocalOperator& lop() {return *lop_; }

    /**
     * Allocate memory in scratch
     */
    auto make_scratch() const {
        return LinearAllocator<double>(scratch_mem_.get(), scratch_mem_.get() + scratch_size_);
    }

    private:

    std::unique_ptr<LocalOperator> lop_;    // on fault: rate and state instance (handles ageing law and slip_rate)
    std::unique_ptr<SeasAdapter> adapter_;  // on domain: DG solver (handles traction and mechanical solver)
    std::unique_ptr<double[]> scratch_mem_; // memory allocated, not sure for what
    std::size_t scratch_size_;              // size of this memory
    double VMax_ = 0.0;                     // metrics: maximal velocity among all fault elements
    size_t evaluation_rhs_count = 0;        // metrics: counts the number of calls of the rhs function in one time step
};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
