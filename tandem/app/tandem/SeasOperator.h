#ifndef SEASOPERATOR_20201001_H
#define SEASOPERATOR_20201001_H

#include "GlobalVariables.h"

#include "form/BoundaryMap.h"
#include "geometry/Curvilinear.h"

#include "tandem/RateAndStateBase.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"
#include "util/Scratch.h"

#include <petscts.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petscsnes.h>

#include <Eigen/Dense>

#include <mpi.h>

#include "Config.h"

#include <cstddef>
#include <memory>
#include <utility>

namespace tndm {

template <typename LocalOperator, typename SeasAdapter> class SeasOperator {
public:

    constexpr static std::size_t Dim = SeasAdapter::Dim;

    constexpr static std::size_t NumQuantities = SeasAdapter::NumQuantities;

    using time_functional_t = typename SeasAdapter::time_functional_t;

    struct solverParametersASandEQ {
        int (*customErrorFct)(TS,NormType,PetscInt*,PetscReal*);
        int (*customNewtonFct)(SNES,Vec);

        bool checkAS;                     // true if it is a period of aseismic slip, false if it is an eartquake    
        bool needRegularizationCompactDAE=false;  // true if it is the first step of the compact DAE (need to set stol = 0)
        bool useImplicitSolver=true;      // true if the solver is implicit (e.g. BDF)

        double time_eq = 0.;              // stores the time at the beginning of the earthquake 

        double FMax = 0.0;                // Maximum residual - only for output
        double ratio_addition = 0.0;      // ratio (sigma * df/dS) / (df/dV) - only for output
        double KSP_iteration_count = 0.;  // counts the average number of iteration for the Jacobian matrix solver in the Newton iteration

        std::optional<tndm::SolverConfigSpecific> current_solver_cfg;

        Vec * previous_solution;

        Vec NewtonTolerances;             // used as stopping criterion of the Newton iteration

        std::shared_ptr<PetscBlockVector> state_current;
        std::shared_ptr<PetscBlockVector> state_as;
        std::shared_ptr<PetscBlockVector> state_eq;

        Formulations current_formulation;
    };

    /**
     * COnstructor function, prepare all operators
     * @param localOperator method to solve the fault (e.g. rate and state)
     * @param seas_adapter handler for the space domain (e.g. discontinuous Galerkin with Poisson/Elasticity solver)
     * @param cfg the struct with the solver configuration
     */
    SeasOperator(std::unique_ptr<LocalOperator> localOperator,
                 std::unique_ptr<SeasAdapter> seas_adapter, 
                 const std::optional<tndm::SolverConfigGeneral>& cfg)
        : lop_(std::move(localOperator)), adapter_(std::move(seas_adapter)), 
        solverGenCfg_(cfg), solverEqCfg_(cfg->solver_earthquake), solverAsCfg_(cfg->solver_aseismicslip), 
        scratch_(lop_->scratch_mem_size() + adapter_->scratch_mem_size(), ALIGNMENT) {

        scratch_.reset();
        adapter_->begin_preparation(numLocalElements());
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->prepare(faultNo, scratch_);
        }
        adapter_->end_preparation();

        lop_->begin_preparation(numLocalElements());
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto fctNo = adapter_->faultMap().fctNo(faultNo);
            lop_->prepare(faultNo, adapter_->topo().info(fctNo), scratch_);
        }
        lop_->end_preparation();
    }

    /**
     * Destructor
     */
    ~SeasOperator(){}

    /**
     * Initialize the system 
     *  - apply initial conditions on the local operator
     *  - solve the system once
     * @param vector solution vector to be initialized [S,psi]
     */
    template <class BlockVector> void initial_condition(BlockVector& vector) {
        scratch_.reset();
        auto access_handle = vector.begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto B = vector.get_block(access_handle, faultNo);
            lop_->pre_init(faultNo, B, scratch_);
        }
        vector.end_access(access_handle);
        size_t blockSize = this->block_size();
        size_t nbf = lop_->space().numBasisFunctions();

        adapter_->solve(0.0, vector);

        access_handle = vector.begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&vector, &access_handle](std::size_t faultNo) {
            return vector.get_block(const_cast<typename BlockVector::const_handle>(access_handle),
                                    faultNo);
        });
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);

            auto B = vector.get_block(access_handle, faultNo);
            lop_->initCompact(faultNo, traction, B, scratch_);
        }
        adapter_->end_traction();
        vector.end_access(access_handle);

        // Initialize the Jacobian matrix
        JacobianQuantities_ = new PetscBlockVector(14 * lop_->space().numBasisFunctions(), numLocalElements(), comm());
    }

    /**
     * generally adapt the size of a solution vector at a formulation switch
     * @param time                current simulation time
     * @param previousFormulation used formulation before the change 
     * @param nextFormulation     used formulation after the change 
     * @param previousVec         solution vector to be written from 
     * @param nextVec             solution vector to be written to 
     */
    template <class BlockVector> void changeProblemSize(double time, 
                tndm::Formulations previousFormulation, tndm::Formulations nextFormulation,
                BlockVector&       previousVec,         BlockVector&       nextVec) {

        if(previousFormulation != tndm::SECOND_ORDER_ODE) adapter_->solve(time, previousVec);

        scratch_.reset();
        auto previous_handle = previousVec.begin_access_readonly();
        auto next_handle     = nextVec.begin_access();
        auto traction        = Managed<Matrix<double>>(adapter_->traction_info());

        if(previousFormulation != tndm::SECOND_ORDER_ODE) adapter_->begin_traction([&previousVec, &previous_handle](std::size_t faultNo) {
            return previousVec.get_block(const_cast<typename BlockVector::const_handle>(previous_handle),
                                    faultNo);
        });
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            if(previousFormulation != tndm::SECOND_ORDER_ODE)  adapter_->traction(faultNo, traction, scratch_);

            auto P = previousVec.get_block(previous_handle, faultNo);
            auto N = nextVec.get_block(next_handle, faultNo);

            lop_->changeProblemSize(faultNo, traction, previousFormulation, nextFormulation, P, N, scratch_);            
        }
        adapter_->end_traction();
        previousVec.end_access_readonly(previous_handle);
        nextVec.end_access(next_handle);
    }

    /**
     * Solve the system for a given timestep
     *  - first solve the DG problem
     *  - then solve the ODE in the rate and state problem
     * @param time current simulation time
     * @param state current solution vector
     * @param result next solution vector
     * @param PetscCall evaluate the Jacobian only if the function is called by the Petsc time solver
     */
    template <typename BlockVector> void rhsCompactODE(double time, BlockVector& state, BlockVector& result, bool PetscCall) {
        time += solverParameters_.time_eq;
        adapter_->solve(time, state);

        scratch_.reset();
        auto in_handle = state.begin_access_readonly();
        auto out_handle = result.begin_access();

        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        auto outJac_handle = JacobianQuantities_->begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);

            auto state_block = state.get_block(in_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities_->get_block(outJac_handle, faultNo);

            double VMax = lop_->rhsCompactODE(faultNo, time, traction, state_block, result_block, scratch_);

            lop_->getJacobianQuantitiesCompact(faultNo, time, traction, state_block, result_block, JacobianQuantities_block, scratch_);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities_->end_access(outJac_handle);

        // bool printMatrices = true;
        // if (PetscCall) testJacobianMatricesCompactODE(state, result, time, printMatrices);

        evaluation_rhs_count++;
    }

    /**
     * Solve the extended system for a given timestep
     *  - do not solve the DG problem!!
     *  - Set up the rhs of the velocity
     *  - then solve the ODE in the rate and state problem
     * @param time current simulation time
     * @param state current solution vector
     * @param result next solution vector
     * @param PetscCall evaluate the Jacobian only if the function is called by the Petsc time solver
     */
    template <typename BlockVector> void rhsExtendedODE(double time, BlockVector& state, BlockVector& result, bool PetscCall) {
        time += solverParameters_.time_eq;
        if (error_extended_ODE_ >= 0) adapter_->solve(time, state); // only to estimate the error

        scratch_.reset();
        auto in_handle = state.begin_access_readonly();
        auto out_handle = result.begin_access();

        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        if (error_extended_ODE_ >= 0) {
            adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
                return state.get_block(in_handle, faultNo);
            });
        }
        VMax_ = 0.0;
        if (error_extended_ODE_ >= 0) error_extended_ODE_ = 0.0;

        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            if (error_extended_ODE_ >= 0) adapter_->traction(faultNo, traction, scratch_);

            auto state_block = state.get_block(in_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);

            double VMax = lop_->rhsExtendedODE(faultNo, time, state_block, result_block, scratch_, solverParameters_.checkAS);

            if (error_extended_ODE_ >= 0) error_extended_ODE_ = std::max(error_extended_ODE_, 
                lop_->applyMaxFrictionLaw(faultNo, time, traction, state_block, scratch_));

            VMax_ = std::max(VMax_, VMax);
        }
        result.end_access(out_handle);
        state.end_access_readonly(in_handle);

        if (RateAndStateBase::TangentialComponents == 2) calculateSigma(state, time);

        in_handle = state.begin_access_readonly();
        auto outJac_handle = JacobianQuantities_->begin_access();
        auto sigma_handle  = sigmaHelperVector_->begin_access_readonly();

        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto JacobianQuantities_block = JacobianQuantities_->get_block(outJac_handle, faultNo);
            auto state_block  = state.get_block(in_handle,    faultNo);
            auto sigma_block = sigmaHelperVector_->get_block(sigma_handle, faultNo);            

            lop_->getJacobianQuantities2ndOrderODE(faultNo, time, sigma_block, state_block, JacobianQuantities_block, scratch_);
        }

        if (error_extended_ODE_ >= 0) adapter_->end_traction();
        state.end_access_readonly(in_handle);
        sigmaHelperVector_->end_access_readonly(sigma_handle);
        JacobianQuantities_->end_access(outJac_handle);

        // calculate the misssing entry dV/dt
        calculateSlipRateExtendedODE(state, result, time);

        // if (PetscCall) testJacobianMatricesExtendedODE(state, result, time, false);

        evaluation_rhs_count++;
    }


    /**
     * Solve the system for a given timestep
     *  - first solve the DG problem
     *  - then solve the ODE in the rate and state problem
     * @param time current simulation time
     * @param state current solution vector
     * @param state_der derivatives of the current solution vector
     * @param result evaluation of the function F()
     */
    template <typename BlockVector> void lhsCompactDAE(double time, BlockVector& state, BlockVector& state_der, BlockVector& result) {
        time += solverParameters_.time_eq;
        adapter_->solve(time, state);

        scratch_.reset();
        auto in_handle = state.begin_access_readonly();
        auto in_der_handle = state_der.begin_access_readonly();
        auto out_handle = result.begin_access();
        
        auto outJac_handle = JacobianQuantities_->begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);

            auto state_block = state.get_block(in_handle, faultNo);
            auto state_der_block = state_der.get_block(in_der_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities_->get_block(outJac_handle, faultNo);

            double VMax = lop_->lhsCompactDAE(faultNo, time, traction, state_block, state_der_block, result_block, scratch_);

            lop_->getJacobianQuantitiesCompact(faultNo, time, traction, state_block, state_der_block, JacobianQuantities_block, scratch_);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state_der.end_access_readonly(in_der_handle);
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities_->end_access(outJac_handle);


        evaluation_rhs_count++;
    }

    /**
     * Solve the system for a given timestep
     *  - first solve the DG problem
     *  - then set up the DAE in the rate and state problem
     * @param time current simulation time
     * @param state current solution vector
     * @param state_der derivatives of the current solution vector
     * @param result evaluation of the function F()
     * @param PetscCall evaluate the Jacobian only if the function is called by the Petsc time solver
     */
    template <typename BlockVector> void lhsExtendedDAE(double time, BlockVector& state, BlockVector& state_der, BlockVector& result, bool PetscCall) {
        time += solverParameters_.time_eq;
        adapter_->solve(time, state);

        scratch_.reset();
        auto in_handle = state.begin_access_readonly();
        auto in_der_handle = state_der.begin_access_readonly();
        auto out_handle = result.begin_access();
        
        auto outJac_handle = JacobianQuantities_->begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);

            auto state_block = state.get_block(in_handle, faultNo);
            auto state_der_block = state_der.get_block(in_der_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities_->get_block(outJac_handle, faultNo);

            double VMax = lop_->lhsExtendedDAE(faultNo, time, traction, state_block, state_der_block, result_block, scratch_);

            lop_->getJacobianQuantitiesExtended(faultNo, time, traction, state_block, JacobianQuantities_block, scratch_);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state_der.end_access_readonly(in_der_handle);
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities_->end_access(outJac_handle);

        bool printMatrices = true;
        if (PetscCall) testJacobianMatricesExtendedDAE(state, state_der, result, time, printMatrices);

        evaluation_rhs_count++;
    }

    /**
     * write solution vector to Finite difference format (see state() in rate and state)
     * @param vector vector in block format
     * @return finite difference object with the solution
     */
    template <typename BlockVector> auto state(BlockVector& vector) {
        auto soln = lop_->state_prototype(numLocalElements());
        auto& values = soln.values();

        scratch_.reset();
        auto in_handle = vector.begin_access_readonly();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&vector, &in_handle](std::size_t faultNo) {
            return vector.get_block(in_handle, faultNo);
        });
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);

            auto value_matrix = values.subtensor(slice{}, slice{}, faultNo);
            auto state_block = vector.get_block(in_handle, faultNo);
            lop_->state(faultNo, traction, state_block, value_matrix, scratch_);
        }
        adapter_->end_traction();
        vector.end_access_readonly(in_handle);
        return soln;
    }

    /** *
     * Calculate the maximum value of df/dpsi
     * @param time current simulation time
     * @param state current solution vector
     */
    template <typename BlockVector> double calculateMaxFactorErrorPSI(double time, BlockVector& state) {
        time += solverParameters_.time_eq;
        auto in_handle = state.begin_access_readonly();

        double max_dfdpsi = 1e9;
        double current_dfdpsi = 0.0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto state_block = state.get_block(in_handle, faultNo);
            current_dfdpsi = lop_->calculateMaxFactorErrorPSI(faultNo, time, state_block);

            max_dfdpsi = std::min(max_dfdpsi, current_dfdpsi);
        }
        state.end_access_readonly(in_handle);
        return max_dfdpsi; 
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

    /**        std::cout << std::endl;

     * Set up the constant part dV/dt for the extended ODE formulation
     */
    void initialize_secondOrderDerivative() {
        // ----------------- general parameters ----------------- //
        size_t blockSize          = this->block_size();
        size_t numFaultElements   = this->numLocalElements();
        size_t nbf                = lop_->space().numBasisFunctions();
        size_t tractionDim        = adapter_->getNumberQuantities();
        size_t faultDim           = RateAndStateBase::TangentialComponents;
        size_t tractionBlockSize  = tractionDim * nbf;
        size_t totalSize          = blockSize         * numFaultElements;
        size_t numFaultNodes      = nbf               * numFaultElements;
        size_t tractionSizeGlobal = tractionBlockSize * numFaultElements;

        // ----------------- Initialize vector ------------------//
        dtau_dt_ = new PetscBlockVector(tractionBlockSize, numFaultElements, comm());
        sigmaHelperVector_ = new PetscBlockVector(nbf, numFaultElements, comm());
        sigmaHelperVector_->set_zero();

        // ----------------- Calculate dtau/dt ----------------- //
                                            
        // vector to initialize other vectors to 0
        PetscBlockVector zeroVector = PetscBlockVector(blockSize, numFaultElements, this->comm());
            zeroVector.set_zero();

        // evaluate tau for the zero vector at time t=1
        double time = 1.0; 
        adapter_->solve(time, zeroVector);

        scratch_.reset();
        auto in_handle = zeroVector.begin_access_readonly();
        auto out_handle = dtau_dt_->begin_access();

        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&zeroVector, &in_handle](std::size_t faultNo) {
            return zeroVector.get_block(in_handle, faultNo);
        });
        if (faultDim == 1){
            for (std::size_t faultNo = 0; faultNo < numFaultElements; ++faultNo) {
                adapter_->traction(faultNo, traction, scratch_);
                auto dtau_dt = dtau_dt_->get_block(out_handle, faultNo);
                for (int i = 0; i < nbf; ++i){
                    dtau_dt(i) = traction(i,1);
                }
            }
        } else if (faultDim == 2) {
            for (std::size_t faultNo = 0; faultNo < numFaultElements; ++faultNo) {
                adapter_->traction(faultNo, traction, scratch_);
                auto dtau_dt = dtau_dt_->get_block(out_handle, faultNo);
                int index = 0;
                for (int j = 0; j < tractionDim; j++){
                    for (int i = 0; i < nbf; ++i){
                        dtau_dt(index++) = traction(i,j);
                    }
                }
            }
        }

        std::cout<<std::endl;
        adapter_->end_traction();

        zeroVector.end_access_readonly(in_handle);
        dtau_dt_->end_access(out_handle);
    }


    /**
     * Initialize the Jacobian matrices and set up the constant part df/dS 
     */
    void prepareJacobian(){

        allocateJacobian(solverAsCfg_->formulation);
        if (solverAsCfg_->formulation != solverEqCfg_->formulation) allocateJacobian(solverEqCfg_->formulation);

        // -------------------- initialize constant df/dS --------------------- //
        initialize_dtau_dS();
    }

    /**
     * Calculate the LU decomposition without pivoting of df/dS
     */
    void calculateLU_dfDS(){
        size_t numFaultElements = this->numLocalElements();
        size_t nbf = lop_->space().numBasisFunctions();
        size_t n = nbf * numFaultElements;

        CHKERRTHROW(MatDuplicate(dtau_dS_, MAT_DO_NOT_COPY_VALUES, &dtau_dS_L_));
        CHKERRTHROW(MatDuplicate(dtau_dS_, MAT_COPY_VALUES, &dtau_dS_U_));
        CHKERRTHROW(MatZeroEntries(dtau_dS_L_));

        const double * A;
        double * L;
        double * U;
        CHKERRTHROW(MatDenseGetArrayRead(dtau_dS_, &A));
        CHKERRTHROW(MatDenseGetArray(dtau_dS_L_, &L));
        CHKERRTHROW(MatDenseGetArray(dtau_dS_U_, &U));

        int i,j,k;  // LU decomposition without pivoting
        for(j = 0; j < n; ++j){
            L[j + j*n] = 1;
            for (i = j + 1; i < n; ++i){
                L[i + j*n] = U[i + j*n] / U[j + j*n];
                for (k = j; k < n; k++){
                    U[i + k*n] -= L[i + j*n] * U[j + k*n];
                }
            }
        }

        CHKERRTHROW(MatDenseRestoreArrayRead(dtau_dS_, &A));
        CHKERRTHROW(MatDenseRestoreArray(dtau_dS_L_, &L));
        CHKERRTHROW(MatDenseRestoreArray(dtau_dS_U_, &U));
    }

    /**
     * calculate the analytic Jacobian matrix
     * @param Jac the Jacobian
     */
    void updateJacobianCompactODE(Mat& Jac){  
        size_t blockSize          = this->block_size();
        size_t numFaultElements   = this->numLocalElements();
        size_t nbf                = lop_->space().numBasisFunctions();
        size_t tractionDim        = adapter_->getNumberQuantities();
        size_t faultDim           = RateAndStateBase::TangentialComponents;
        size_t totalSize          = blockSize   * numFaultElements;
        size_t numFaultNodes      = nbf         * numFaultElements;
        size_t tractionSizeGlobal = tractionDim * numFaultNodes;

        using namespace Eigen;

        // fill vectors     
        VectorXd df_dV_vec   (numFaultNodes);
        VectorXd df_dpsi_vec (numFaultNodes);
        VectorXd dg_dV_vec   (numFaultNodes);
        VectorXd dg_dpsi_vec (numFaultNodes);

        VectorXd f_vec           (numFaultNodes);
        VectorXd ratio_tau_V_vec (numFaultNodes);
        VectorXd unit_y_vec      (numFaultNodes);
        VectorXd unit_z_vec      (numFaultNodes);

        auto accessRead = JacobianQuantities_->begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);
            for(int i = 0; i<nbf; i++){
                df_dV_vec(   noFault * nbf + i) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i) = derBlock(1 * nbf + i);
                dg_dV_vec(   noFault * nbf + i) = derBlock(2 * nbf + i);
                dg_dpsi_vec( noFault * nbf + i) = derBlock(3 * nbf + i);
            }
        }
        if (faultDim == 2){
            for (int noFault = 0; noFault < numFaultElements; noFault++){
                auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);
                for(int i = 0; i<nbf; i++){
                    f_vec(          noFault * nbf + i) = derBlock(4  * nbf + i);
                    ratio_tau_V_vec(noFault * nbf + i) = derBlock(5  * nbf + i);
                    unit_y_vec(     noFault * nbf + i) = derBlock(6  * nbf + i);
                    unit_z_vec(     noFault * nbf + i) = derBlock(7  * nbf + i);
                }
            }
        }

        JacobianQuantities_->end_access_readonly(accessRead);


        // fill the Jacobian
        int Sy_i;
        int Sz_i;
        int PSI_i;
        int Sy_j;
        int Sz_j;
        int n_i;
        int ny_j;
        int nz_j;
        int n_j;
        int sig_x_i;
        int tau_y_i;
        int tau_z_i;
        
        CHKERRTHROW(MatZeroEntries(Jac));

        const double* dtau_dS;
        double* F_x;
        CHKERRTHROW(MatDenseGetArrayRead(dtau_dS_, &dtau_dS));
        CHKERRTHROW(MatDenseGetArrayWrite(Jac, &F_x));
        if (faultDim == 1) {
            for (int noFault = 0; noFault < numFaultElements; noFault++){
                for (int i = 0; i < nbf; i++){                  // row iteration
                    Sy_i   = noFault * blockSize + i;
                    PSI_i  = noFault * blockSize + nbf + i;
                    n_i    = noFault * nbf + i;
                    
                    F_x[Sy_i  + PSI_i * totalSize] = -df_dpsi_vec(n_i) / df_dV_vec(n_i);  // dV/dpsi
                    F_x[PSI_i + PSI_i * totalSize] = dg_dpsi_vec(n_i) + dg_dV_vec(n_i) * F_x[Sy_i   + PSI_i * totalSize];  // dg/dpsi
                    
                    for (int noFault_j = 0; noFault_j < numFaultElements; noFault_j++){
                        for(int j = 0; j < nbf; j++) {          // column iteration

                            Sy_j = noFault_j * blockSize + j;
                            n_j = noFault_j * nbf + j;

                            // column major !                       
                            F_x[Sy_i   + Sy_j * totalSize  ] = -dtau_dS[n_i + n_j * numFaultNodes] / df_dV_vec(n_i);    // dV/dS
                            F_x[PSI_i + Sy_j * totalSize  ] = dg_dV_vec(n_i) * F_x[Sy_i   + Sy_j * totalSize  ];       // dg/dS = -b/L * dV/dS
                        }
                    }
                }
            }
        } else if (faultDim == 2){
            for (int noFault_i = 0; noFault_i < numFaultElements; noFault_i++){
                for (int i = 0; i < nbf; i++){                  // row iteration
                    Sy_i    = noFault_i * blockSize         + 0 * nbf + i;
                    Sz_i    = noFault_i * blockSize         + 1 * nbf + i;
                    PSI_i   = noFault_i * blockSize         + 2 * nbf + i;
                    n_i     = noFault_i * nbf                         + i;
                    sig_x_i = noFault_i * tractionDim * nbf + 0 * nbf + i;
                    tau_y_i = noFault_i * tractionDim * nbf + 1 * nbf + i;
                    tau_z_i = noFault_i * tractionDim * nbf + 2 * nbf + i;
                    
                    // sparse components
                    F_x[Sy_i  + PSI_i * totalSize] = -unit_y_vec(n_i) * df_dpsi_vec(n_i) / df_dV_vec(n_i);    // dVy/dpsi
                    F_x[Sz_i  + PSI_i * totalSize] = -unit_z_vec(n_i) * df_dpsi_vec(n_i) / df_dV_vec(n_i);    // dVz/dpsi
                    F_x[PSI_i + PSI_i * totalSize] =  dg_dpsi_vec(n_i) 
                                                     -dg_dV_vec(n_i) * df_dpsi_vec(n_i) / df_dV_vec(n_i);   // dg/dpsi

                    // dense components  
                    for (int noFault_j = 0; noFault_j < numFaultElements; noFault_j++){
                        for(int j = 0; j < nbf; j++) {          // column iteration
                            Sy_j    = noFault_j * blockSize      + 0 * nbf + j;
                            Sz_j    = noFault_j * blockSize      + 1 * nbf + j;
                            ny_j    = noFault_j * faultDim * nbf + 0 * nbf + j;
                            nz_j    = noFault_j * faultDim * nbf + 1 * nbf + j;

                            F_x[Sy_i   + Sy_j * totalSize  ] = 
                              + dtau_dS[tau_y_i + ny_j * tractionSizeGlobal] * ratio_tau_V_vec(n_i) 
                              + unit_y_vec(n_i) * 
                            (
                              - (   + dtau_dS[sig_x_i + ny_j * tractionSizeGlobal] * f_vec(n_i)
                                    + dtau_dS[tau_y_i + ny_j * tractionSizeGlobal] * unit_y_vec(n_i)
                                    + dtau_dS[tau_z_i + ny_j * tractionSizeGlobal] * unit_z_vec(n_i)
                                ) / df_dV_vec(n_i)
                                - ratio_tau_V_vec(n_i) * 
                                (   + dtau_dS[tau_y_i + ny_j * tractionSizeGlobal] * unit_y_vec(n_i)
                                    + dtau_dS[tau_z_i + ny_j * tractionSizeGlobal] * unit_z_vec(n_i)
                                )
                             );      // dVy/dSy

                            F_x[Sy_i   + Sz_j * totalSize  ] = 
                              + dtau_dS[tau_y_i + nz_j * tractionSizeGlobal] * ratio_tau_V_vec(n_i) 
                              + unit_y_vec(n_i) * 
                            (
                              - (   + dtau_dS[sig_x_i + nz_j * tractionSizeGlobal] * f_vec(n_i)
                                    + dtau_dS[tau_y_i + nz_j * tractionSizeGlobal] * unit_y_vec(n_i)
                                    + dtau_dS[tau_z_i + nz_j * tractionSizeGlobal] * unit_z_vec(n_i)
                                ) / df_dV_vec(n_i)
                                - ratio_tau_V_vec(n_i) * 
                                (   + dtau_dS[tau_y_i + nz_j * tractionSizeGlobal] * unit_y_vec(n_i)
                                    + dtau_dS[tau_z_i + nz_j * tractionSizeGlobal] * unit_z_vec(n_i)
                                )
                             );      // dVy/dSz


                            F_x[Sz_i   + Sy_j * totalSize  ] = 
                              + dtau_dS[tau_z_i + nz_j * tractionSizeGlobal] * ratio_tau_V_vec(n_i) 
                              + unit_z_vec(n_i) * 
                            (
                              - (   + dtau_dS[sig_x_i + ny_j * tractionSizeGlobal] * f_vec(n_i)
                                    + dtau_dS[tau_y_i + ny_j * tractionSizeGlobal] * unit_y_vec(n_i)
                                    + dtau_dS[tau_z_i + ny_j * tractionSizeGlobal] * unit_z_vec(n_i)
                                ) / df_dV_vec(n_i)
                                - ratio_tau_V_vec(n_i) * 
                                (   + dtau_dS[tau_y_i + ny_j * tractionSizeGlobal] * unit_y_vec(n_i)
                                    + dtau_dS[tau_z_i + ny_j * tractionSizeGlobal] * unit_z_vec(n_i)
                                )
                             );      // dVz/dSz

                            F_x[Sz_i   + Sz_j * totalSize  ] = 
                              + dtau_dS[tau_z_i + nz_j * tractionSizeGlobal] * ratio_tau_V_vec(n_i) 
                              + unit_z_vec(n_i) * 
                            (
                              - (   + dtau_dS[sig_x_i + nz_j * tractionSizeGlobal] * f_vec(n_i)
                                    + dtau_dS[tau_y_i + nz_j * tractionSizeGlobal] * unit_y_vec(n_i)
                                    + dtau_dS[tau_z_i + nz_j * tractionSizeGlobal] * unit_z_vec(n_i)
                                ) / df_dV_vec(n_i)
                                - ratio_tau_V_vec(n_i) * 
                                (   + dtau_dS[tau_y_i + nz_j * tractionSizeGlobal] * unit_y_vec(n_i)
                                    + dtau_dS[tau_z_i + nz_j * tractionSizeGlobal] * unit_z_vec(n_i)
                                )
                             );      // dVz/dSz


                            F_x[PSI_i + Sy_j * totalSize  ]  = dg_dV_vec(n_i) * 
                            (
                               + F_x[Sy_i   + Sy_j * totalSize  ] * unit_y_vec(n_i)
                               + F_x[Sz_i   + Sy_j * totalSize  ] * unit_z_vec(n_i)
                            );      // dg/dSy

                            F_x[PSI_i + Sz_j * totalSize  ]  = dg_dV_vec(n_i) * 
                            (
                               + F_x[Sy_i   + Sz_j * totalSize  ] * unit_y_vec(n_i)
                               + F_x[Sz_i   + Sz_j * totalSize  ] * unit_z_vec(n_i)
                            );      // dg/dSz
                        }
                    }
                }
            }
        }
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jac, &F_x));
        CHKERRTHROW(MatDenseRestoreArrayRead(dtau_dS_, &dtau_dS));
    }

    /**
     * calculate the analytic Jacobian matrix
     * @param Jac the Jacobian
     */
    void updateJacobianExtendedODE(Mat& Jac){  

        using namespace Eigen;

        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t nbf = lop_->space().numBasisFunctions();
        size_t totalSize = blockSize * numFaultElements;
        size_t numFaultNodes = nbf * numFaultElements;

        // fill vectors     
        VectorXd df_dV_vec(numFaultNodes);                             
        VectorXd df_dpsi_vec(numFaultNodes);
        VectorXd dg_dV_vec(numFaultNodes);                             
        VectorXd dg_dpsi_vec(numFaultNodes);
        VectorXd dxi_dV_vec(numFaultNodes);                             
        VectorXd dxi_dpsi_vec(numFaultNodes);
        VectorXd dzeta_dV_vec(numFaultNodes);                             
        VectorXd dzeta_dpsi_vec(numFaultNodes);
        VectorXd V_vec(numFaultNodes);
        VectorXd g_vec(numFaultNodes);
        VectorXd A_V_vec = VectorXd::Zero(numFaultNodes);

        auto accessRead = JacobianQuantities_->begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(      noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec(    noFault * nbf + i ) = derBlock(1 * nbf + i);
                dg_dV_vec(      noFault * nbf + i ) = derBlock(2 * nbf + i);
                dg_dpsi_vec(    noFault * nbf + i ) = derBlock(3 * nbf + i);
                dxi_dV_vec(     noFault * nbf + i ) = derBlock(4 * nbf + i);
                dxi_dpsi_vec(   noFault * nbf + i ) = derBlock(5 * nbf + i);
                dzeta_dV_vec(   noFault * nbf + i ) = derBlock(6 * nbf + i);
                dzeta_dpsi_vec( noFault * nbf + i ) = derBlock(7 * nbf + i);
                V_vec(          noFault * nbf + i ) = derBlock(8 * nbf + i);
                g_vec(          noFault * nbf + i ) = derBlock(9 * nbf + i);
            }
        }
        JacobianQuantities_->end_access_readonly(accessRead);

        // fill the Jacobian
        int S_i;
        int PSI_i;
        int V_i;
        int V_j;
        int n_i;
        int n_j;

        CHKERRTHROW(MatZeroEntries(Jac));

        const double* dtau_dS;
        double* J;

        CHKERRTHROW(MatDenseGetArrayRead(dtau_dS_, &dtau_dS));

        // Fill the vector A_V
        for (int j = 0; j < numFaultNodes; j++){
            for (int i = 0; i < numFaultNodes; i++){
                A_V_vec(i) += dtau_dS[i + j * numFaultNodes] * V_vec(j);
            }
        }

        auto dtaudt_handle = dtau_dt_->begin_access_readonly();
        CHKERRTHROW(MatDenseGetArrayWrite(Jac, &J));
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto C = dtau_dt_->get_block(dtaudt_handle, noFault);
            for (int i = 0; i < nbf; i++){                  // row iteration
                S_i = noFault * blockSize + i;
                PSI_i = noFault * blockSize + 
                        RateAndStateBase::TangentialComponents     * nbf + i;
                V_i = noFault * blockSize + 
                       (RateAndStateBase::TangentialComponents +1) * nbf + i;
                n_i = noFault * nbf + i;
                
                J[S_i   + S_i   * totalSize] = 1.0;
                J[PSI_i + PSI_i * totalSize] = dg_dpsi_vec(n_i);    // dg/dpsi
                J[PSI_i + V_i   * totalSize] = dg_dV_vec(n_i);      // dg/dV
                J[V_i   + PSI_i * totalSize] =                      // dh/dpsi
                    - dxi_dpsi_vec(n_i) * (C(i) + 
                                           A_V_vec(n_i) + 
                                           df_dpsi_vec(n_i) * g_vec(n_i) ) - 
                     (dzeta_dpsi_vec(n_i) * g_vec(n_i) + 
                      df_dpsi_vec(n_i) * dg_dpsi_vec(n_i)) / df_dV_vec(n_i);
                 J[V_i   + V_i   * totalSize] =                      // dh/dV
                    - dxi_dV_vec(n_i) * (C(i) + 
                                         A_V_vec(n_i) + 
                                         df_dpsi_vec(n_i) * g_vec(n_i) ) - 
                     (dzeta_dV_vec(n_i) * g_vec(n_i) + 
                      df_dpsi_vec(n_i) * dg_dV_vec(n_i)) / df_dV_vec(n_i) ;

                for (int noFault_j = 0; noFault_j < numFaultElements; noFault_j++){
                    for(int j = 0; j < nbf; j++) {          // column iteration

                        V_j = noFault * blockSize + 
                              (RateAndStateBase::TangentialComponents +1) * nbf + j;
                        n_j = noFault_j * nbf + j;

                        // dense components in dh/dV
                        J[V_i   + V_j * totalSize  ] -= dtau_dS[n_i + n_j * numFaultNodes] / df_dV_vec(n_i);
                    }
                }
            }
        }
        dtau_dt_->end_access_readonly(dtaudt_handle);
        CHKERRTHROW(MatDenseRestoreArrayRead(dtau_dS_, &dtau_dS));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jac, &J));
    }
 
    /**
     * Updates the two Jacobians F_x and F_\dot{x}
     * @param sigma shift of due to the numerical scheme
     * @param Jac the Jacobian
     */
    void updateJacobianCompactDAE(double sigma, Mat& Jac){  

        using namespace Eigen;

        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t totalSize = blockSize * numFaultElements;
        size_t nbf = lop_->space().numBasisFunctions();
        size_t numFaultNodes = nbf * numFaultElements;

        // fill vectors
        VectorXd df_dV_vec(numFaultNodes);                             
        VectorXd df_dpsi_vec(numFaultNodes);
        VectorXd dg_dV_vec(numFaultNodes);                             
        VectorXd dg_dpsi_vec(numFaultNodes);

        auto accessRead = JacobianQuantities_->begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
                dg_dV_vec(   noFault * nbf + i ) = derBlock(2 * nbf + i);
                dg_dpsi_vec( noFault * nbf + i ) = derBlock(3 * nbf + i);
            }
        }
        JacobianQuantities_->end_access_readonly(accessRead);

        // fill the Jacobian
        int S_i;
        int PSI_i;
        int S_j;
        int n_i;
        int n_j;

        CHKERRTHROW(MatZeroEntries(Jac));

        const double* dtau_dS;
        double* J;
        CHKERRTHROW(MatDenseGetArrayRead(dtau_dS_, &dtau_dS));
        CHKERRTHROW(MatDenseGetArrayWrite(Jac, &J));

        solverParameters_.ratio_addition = 0.0;
        double ratio;
        for (int i = 0; i < numFaultNodes; ++i) {
            ratio = sigma * df_dV_vec(i) / dtau_dS[i + i * numFaultNodes];
            solverParameters_.ratio_addition = std::max(solverParameters_.ratio_addition, std::abs(ratio));
        }

        for (int noFault = 0; noFault < numFaultElements; noFault++){
            for (int i = 0; i < nbf; i++){                  // row iteration
                S_i = noFault * blockSize + i;
                PSI_i = noFault * blockSize + 
                                       RateAndStateBase::TangentialComponents * nbf + i;
                n_i = noFault * nbf + i;

                // components F_xdot                
                J[S_i   +   S_i * totalSize] += sigma * df_dV_vec(n_i);
                J[PSI_i +   S_i * totalSize] += sigma * dg_dV_vec(n_i);
                J[PSI_i + PSI_i * totalSize] -= sigma;

                // components F_x
                J[S_i   + PSI_i * totalSize] += df_dpsi_vec(n_i);
                J[PSI_i + PSI_i * totalSize] += dg_dpsi_vec(n_i);

                //if (!solverGenCfg_->bdf_custom_LU_solver){
                    for (int noFault_j = 0; noFault_j < numFaultElements; noFault_j++){
                        for(int j = 0; j < nbf; j++) {          // column iteration

                            S_j = noFault_j * blockSize + j;
                            n_j = noFault_j * nbf + j;

                            // components F_x
                            J[S_i   + S_j * totalSize  ] += dtau_dS[n_i + n_j * numFaultNodes];
                        }
                    }
                //}
            }
        }

        CHKERRTHROW(MatDenseRestoreArrayRead(dtau_dS_, &dtau_dS));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jac, &J));
    }

    /**
     * Updates the two Jacobians F_x and F_\dot{x}
     * @param sigma shift of due to the numerical scheme
     * @param Jac the Jacobian to be updated
     */
    void updateJacobianExtendedDAE(double sigma, Mat& Jac ){  

        using namespace Eigen;

        size_t blockSize          = this->block_size();
        size_t numFaultElements   = this->numLocalElements();
        size_t nbf                = lop_->space().numBasisFunctions();
        size_t tractionDim        = adapter_->getNumberQuantities();
        size_t faultDim           = RateAndStateBase::TangentialComponents;
        size_t tractionBlockSize  = tractionDim * nbf;
        size_t totalSize          = blockSize         * numFaultElements;
        size_t numFaultNodes      = nbf               * numFaultElements;
        size_t tractionSizeGlobal = tractionBlockSize * numFaultElements;

        // fill vectors     
        VectorXd df_dV_vec   (numFaultNodes);                             
        VectorXd df_dpsi_vec (numFaultNodes);
        VectorXd dg_dV_vec   (numFaultNodes);                             
        VectorXd dg_dpsi_vec (numFaultNodes);

        VectorXd f_vec           (numFaultNodes);
        VectorXd ratio_tau_V_vec (numFaultNodes);
        VectorXd unit_y_vec      (numFaultNodes);
        VectorXd unit_z_vec      (numFaultNodes);

        auto accessRead = JacobianQuantities_->begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
                dg_dV_vec(   noFault * nbf + i ) = derBlock(2 * nbf + i);
                dg_dpsi_vec( noFault * nbf + i ) = derBlock(3 * nbf + i);
            }
        }
        if (faultDim == 2){
            for (int noFault = 0; noFault < numFaultElements; noFault++){
                auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);    
                for(int i = 0; i<nbf; i++){ 
                    f_vec(          noFault * nbf + i) = derBlock(4  * nbf + i);
                    ratio_tau_V_vec(noFault * nbf + i) = derBlock(5  * nbf + i);
                    unit_y_vec(     noFault * nbf + i) = derBlock(6  * nbf + i);
                    unit_z_vec(     noFault * nbf + i) = derBlock(7  * nbf + i);
                }
            }
        }

        JacobianQuantities_->end_access_readonly(accessRead);

        // fill the Jacobian
        int S_i;
        int Sy_i;
        int Sz_i;
        int PSI_i;
        int V_i;
        int sig_x_i;
        int tau_y_i;
        int tau_z_i;
        int n_i;
        int S_j;
        int Sy_j;
        int Sz_j;
        int ny_j;
        int nz_j;
        int n_j;

        CHKERRTHROW(MatZeroEntries(Jac));

        const double* dtau_dS;
        double* J;
        CHKERRTHROW(MatDenseGetArrayRead(dtau_dS_, &dtau_dS));
        CHKERRTHROW(MatDenseGetArrayWrite(Jac, &J));
        if (faultDim == 1) {
            for (int noFault_i = 0; noFault_i < numFaultElements; noFault_i++){
                for (int i = 0; i < nbf; i++){                  // row iteration
                    S_i   = noFault_i * blockSize + 0 * nbf + i;
                    PSI_i = noFault_i * blockSize + 1 * nbf + i;
                    V_i   = noFault_i * blockSize + 2 * nbf + i;
                    n_i   = noFault_i * nbf                 + i;
                    
                    // components F_xdot
                    J[S_i   +   S_i * totalSize] = -sigma;
                    J[PSI_i + PSI_i * totalSize] = -sigma;

                    // components F_x
                    J[S_i   + V_i   * totalSize] = 1;
                    J[PSI_i + PSI_i * totalSize] += dg_dpsi_vec(n_i);
                    J[PSI_i + V_i   * totalSize] = dg_dV_vec(n_i);
                    J[V_i   + PSI_i * totalSize] = df_dpsi_vec(n_i);
                    J[V_i   + V_i   * totalSize] = df_dV_vec(n_i);
                    
                    for (int noFault_j = 0; noFault_j < numFaultElements; noFault_j++){
                        for(int j = 0; j < nbf; j++) {          // column iteration

                            S_j = noFault_j * blockSize + j;
                            n_j = noFault_j * nbf       + j;

                            // components F_x     
                            J[V_i   + S_j * totalSize  ] += dtau_dS[n_i + n_j * numFaultNodes];
                        }
                    }
                }
            }
        } else if (faultDim == 2) {
            for (int noFault_i = 0; noFault_i < numFaultElements; noFault_i++){
                for (int i = 0; i < nbf; i++){                  // row iteration
                    Sy_i    = noFault_i * blockSize         + 0 * nbf + i;
                    Sz_i    = noFault_i * blockSize         + 1 * nbf + i;
                    PSI_i   = noFault_i * blockSize         + 2 * nbf + i;
                    V_i     = noFault_i * blockSize         + 3 * nbf + i;
                    n_i     = noFault_i * nbf                         + i;
                    sig_x_i = noFault_i * tractionBlockSize + 0 * nbf + i;
                    tau_y_i = noFault_i * tractionBlockSize + 1 * nbf + i;
                    tau_z_i = noFault_i * tractionBlockSize + 2 * nbf + i;
                    
                    // components F_xdot                
                    J[Sy_i  +  Sy_i * totalSize] = -sigma;
                    J[Sz_i  +  Sz_i * totalSize] = -sigma;
                    J[PSI_i + PSI_i * totalSize] = -sigma;

                    // components F_x
                    J[Sy_i  + V_i   * totalSize] = unit_y_vec(n_i);
                    J[Sz_i  + V_i   * totalSize] = unit_z_vec(n_i);
                    J[PSI_i + PSI_i * totalSize] += dg_dpsi_vec(n_i);
                    J[PSI_i + V_i   * totalSize] = dg_dV_vec(n_i);
                    J[V_i   + PSI_i * totalSize] = df_dpsi_vec(n_i);
                    J[V_i   + V_i   * totalSize] = df_dV_vec(n_i);
                    
                    for (int noFault_j = 0; noFault_j < numFaultElements; noFault_j++){
                        for(int j = 0; j < nbf; j++) {          // column iteration

                            Sy_j    = noFault_j * blockSize      + 0 * nbf + j;
                            Sz_j    = noFault_j * blockSize      + 1 * nbf + j;
                            ny_j    = noFault_j * faultDim * nbf + 0 * nbf + j;
                            nz_j    = noFault_j * faultDim * nbf + 1 * nbf + j;

                            // components F_x
                            J[Sy_i   + Sy_j * totalSize  ] +=  ratio_tau_V_vec(n_i) * // dSy/dSy
                              ( 
                                + dtau_dS[sig_x_i + ny_j * tractionSizeGlobal] * unit_y_vec(n_i) * f_vec(n_i)
                                + dtau_dS[tau_y_i + ny_j * tractionSizeGlobal]
                              );
                            J[Sy_i   + Sz_j * totalSize  ] +=  ratio_tau_V_vec(n_i) * // dSy/dSz
                              (  
                                + dtau_dS[sig_x_i + nz_j * tractionSizeGlobal] * unit_y_vec(n_i) * f_vec(n_i)
                                + dtau_dS[tau_y_i + nz_j * tractionSizeGlobal]
                              );  
                            J[Sz_i   + Sy_j * totalSize  ] +=  ratio_tau_V_vec(n_i) * // dSz/dSy
                              (  
                                + dtau_dS[sig_x_i + ny_j * tractionSizeGlobal] * unit_z_vec(n_i) * f_vec(n_i)
                                + dtau_dS[tau_z_i + ny_j * tractionSizeGlobal]
                              ); 
                            J[Sz_i   + Sz_j * totalSize  ] +=  ratio_tau_V_vec(n_i) * // dSz/dSz
                              (  
                                + dtau_dS[sig_x_i + nz_j * tractionSizeGlobal] * unit_z_vec(n_i) * f_vec(n_i)
                                + dtau_dS[tau_z_i + nz_j * tractionSizeGlobal]
                              ); 
                            J[V_i   + Sy_j * totalSize  ] =  // dF/dSy
                                + dtau_dS[sig_x_i + ny_j * tractionSizeGlobal] * f_vec(n_i)
                                + dtau_dS[tau_y_i + ny_j * tractionSizeGlobal] * unit_y_vec(n_i)
                                + dtau_dS[tau_z_i + ny_j * tractionSizeGlobal] * unit_z_vec(n_i);

                            J[V_i   + Sz_j * totalSize  ] =  // dF/dSz
                                + dtau_dS[sig_x_i + nz_j * tractionSizeGlobal] * f_vec(n_i)
                                + dtau_dS[tau_y_i + nz_j * tractionSizeGlobal] * unit_y_vec(n_i)
                                + dtau_dS[tau_z_i + nz_j * tractionSizeGlobal] * unit_z_vec(n_i);
                        }
                    }
                }
            }
        }

        CHKERRTHROW(MatDenseRestoreArrayRead(dtau_dS_, &dtau_dS));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jac, &J));
    }

    /**
     * Solve the linear system of the Newton iteration for the compact DAE formulation with a partial LU decomposition
     * @param x_vec searched unknown
     * @param b_vec rhs Vector
     * @param J_mat partial Jacobian with block diagonal entries
     * @param ksp linear solver from snes
     */
    void applyCustomLUSolver(Vec x_vec, Vec b_vec, Mat J_mat, KSP ksp){
        switch(solverParameters_.current_formulation) {
            case tndm::COMPACT_DAE  : ReducedKSPCompactDAE(x_vec, b_vec, J_mat, ksp); break;
            case tndm::EXTENDED_DAE : ReducedKSPExtendedDAE(x_vec, b_vec, J_mat, ksp); break;
            default : std::cerr << "Reduced Newton solver is only available for DAE formulations! " << std::endl;
        }
    }

    /**
     *  Getter functions
     */
    Mat& getJacobianCompactODE() { return Jacobian_compact_ode_; }

    Mat& getJacobianExtendedODE() { return Jacobian_extended_ode_; }

    Mat& getJacobianCompactDAE() { return Jacobian_compact_dae_; }

    Mat& getJacobianExtendedDAE() { return Jacobian_extended_dae_; }

    std::size_t block_size() const { return lop_->block_size(); }

    std::size_t block_size_formulation(tndm::Formulations formulation) const { return lop_->block_size_formulation(formulation); }
    
    MPI_Comm comm() const { return adapter_->topo().comm(); }
    
    BoundaryMap const& faultMap() const { return adapter_->faultMap(); }
    
    std::size_t numLocalElements() const { return adapter_->faultMap().size(); }

    SeasAdapter const& adapter() const { return *adapter_; }

    size_t rhs_count() {return evaluation_rhs_count; };

    double VMax() const { return VMax_; }

    double fMax() const { return (error_extended_ODE_ >= 0) ? error_extended_ODE_ : solverParameters_.FMax; }
    
    LocalOperator& lop() {return *lop_; }

    solverParametersASandEQ& getSolverParameters(){ return solverParameters_; }

    std::size_t getProblemDimension() const {return Dim;};

    auto& getGeneralSolverConfiguration() const { return solverGenCfg_; }

    auto& getEarthquakeSolverConfiguration() const { return solverEqCfg_; }

    auto& getAseismicSlipSolverConfiguration() const { return solverAsCfg_; }

    double getV0(){ return lop_->getV0(); }

    void setFormulation(tndm::Formulations formulation) { 
        lop_->setFormulation(formulation); 
        solverParameters_.current_formulation = formulation;
    }


    /**
     * Evalulate the allowed ratio between absolute error in the slip and in the state variable
     * Just for testing purposes, to know how to set tolerances
     */
    void evaluateErrorRatio() {
        // get one vector as state  
        PetscBlockVector oneVector(block_size(), numLocalElements(), comm());
        auto AccessHandle = oneVector.begin_access();
        for (int noFault = 0; noFault < numLocalElements(); noFault++){
            for (int i = 0; i < lop_->space().numBasisFunctions(); i++){
                auto block = oneVector.get_block(AccessHandle, noFault);
                block(i) = 1;
            }
        }
        oneVector.end_access(AccessHandle);        


        PetscBlockVector solutionVector(oneVector);
        adapter_->solveUnitVector(oneVector, solutionVector);

        scratch_.reset();
        auto in_handle = oneVector.begin_access_readonly();
       
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&oneVector, &in_handle](std::size_t faultNo) {
            return oneVector.get_block(in_handle, faultNo);
        });
        double tau_min = 10;
        double sn_max = 0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);
 
            for (std::size_t node = 0; node < lop_->space().numBasisFunctions(); ++node) {
                sn_max = std::max(sn_max, abs(traction(node, 0)));
                tau_min = std::min(tau_min, abs(traction(node, 1)));
                std::cout << "sn: " << traction(node, 0) << std::endl;
                std::cout << "tau: " << traction(node, 1) << std::endl;
            }
        }
        adapter_->end_traction();
        oneVector.end_access_readonly(in_handle);

        double psi_max = 0.85;
        double V_max = 5.0e9;
        double den = lop_->getLaw().evaluateErrorRatioDenominator(sn_max, psi_max, V_max);

        std::cout << "tau min: " << tau_min << " and denominator: " << den << std::endl; 

        double ratio = tau_min / den;
        std::cout << " allowed ratio is: " << ratio << std::endl;
    }

    /**
     * Only for the 2nd order ODE formulation:
     * set the components of the slip to 0 in the rhs of the Newton iteration
     * @param f rhs vector with 3 block elements
     */
    void updateRHSNewtonIteration(Vec& f){
        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t nbf = lop_->space().numBasisFunctions();
        size_t tc = RateAndStateBase::TangentialComponents;

        double *s;

        CHKERRTHROW(VecGetArray(f, &s));

        for (int noFault = 0; noFault < numFaultElements; ++noFault){
            for(int i = 0; i < nbf; ++i){
                for (int t = 0; t < tc; t++){       
                    s[noFault*blockSize + t*nbf + i] = 0.0;
                }
            }
        }

        CHKERRTHROW(VecRestoreArray(f, &s));
    }

    /**
     * Only for the 2nd order ODE formulation:
     * calculate the slip after the Newton iteration: V = shift*S + V0
     * @param state the solution vector to write to 
     * @param n amount of last steps to consider
     * @param vecs array of vectors with the solutions at the last step
     * @param alpha array of coefficients 
     * @param shift parameter from the timestep size and the of the BDF scheme
     */
    void setSlipAfterNewtonIteration(Vec& state, size_t n, Vec vecs[7], double alpha[7], double shift){
        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t nbf = lop_->space().numBasisFunctions();
        size_t tc = RateAndStateBase::TangentialComponents;

        double       *s;
        const double *v;

        CHKERRTHROW(VecGetArray(state, &s));

        for (int noFault = 0; noFault < numFaultElements; ++noFault){   // start with V
            for(int i = 0; i < nbf; ++i){
                for (int t = 0; t < tc; t++){
                    s[noFault*blockSize + t*nbf + i] = s[noFault*blockSize + (tc+t+1)*nbf + i];
                }
            }
        }
        for(int k = 1; k < n; k++){                                     // substract V0
            CHKERRTHROW(VecGetArrayRead(vecs[k], &v));  
            for (int noFault = 0; noFault < numFaultElements; ++noFault){
                for(int i = 0; i < nbf; ++i){
                    for (int t = 0; t < tc; t++){
                        s[noFault*blockSize + t*nbf + i] -= alpha[k] * v[noFault*blockSize + t*nbf + i];
                    }
                }
            }
            CHKERRTHROW(VecRestoreArrayRead(vecs[k], &v));
        }
        for (int noFault = 0; noFault < numFaultElements; ++noFault){   // divide by shift
            for(int i = 0; i < nbf; ++i){
                for (int t = 0; t < tc; t++){
                    s[noFault*blockSize + t*nbf + i] /= shift;
                }
            }
        }
        CHKERRTHROW(VecRestoreArray(state, &s));


    }

    private:

    /** allocate Jacobian matrices and set entries to 0
     * @param formulation Formulation whose Jacobian shall be initialized
     */
    void allocateJacobian(tndm::Formulations formulation){
        size_t numFaultElements = this->numLocalElements();
        size_t totalSize = numFaultElements * block_size_formulation(formulation);

        switch(formulation){
            case tndm::FIRST_ORDER_ODE: 
                CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_compact_ode_));
                CHKERRTHROW(MatZeroEntries(Jacobian_compact_ode_));
                break;
            case tndm::EXTENDED_DAE: 
                CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_extended_dae_));
                CHKERRTHROW(MatZeroEntries(Jacobian_extended_dae_));
                break;
            case tndm::COMPACT_DAE: 
                CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_compact_dae_));
                CHKERRTHROW(MatZeroEntries(Jacobian_compact_dae_));
                break;
            case tndm::SECOND_ORDER_ODE: 
                CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_extended_ode_));
                CHKERRTHROW(MatZeroEntries(Jacobian_extended_ode_));
                break;
                default: std::cerr << "Internal error: Unknown formulation." << std::endl;
        }
    }


    /**
     * Set up the constant part df/dS of the Jacobian matrix
     */
    void initialize_dtau_dS() {

        // ----------------- general parameters ----------------- //
        size_t blockSize            = this->block_size();
        size_t numFaultElements     = this->numLocalElements();
        size_t nbf                  = lop_->space().numBasisFunctions();
        size_t numTractionComps     = adapter_->getNumberQuantities();
        size_t faultDim             = RateAndStateBase::TangentialComponents;
        size_t faultSizeLocal       = faultDim         * nbf;
        size_t tractionSizeLocal    = numTractionComps * nbf;
        size_t quadratureSizeLocal  = adapter_->block_size_rhsDG();  
        size_t faultSizeGlobal      = faultSizeLocal      * numFaultElements;
        size_t quadratureSizeGlobal = quadratureSizeLocal * numFaultElements;
        size_t tractionSizeGlobal   = tractionSizeLocal   * numFaultElements;
        size_t totalSize            = blockSize           * numFaultElements;

        CHKERRTHROW(MatCreateSeqDense(comm(), tractionSizeGlobal, faultSizeGlobal, NULL, &dtau_dS_));

        PetscBlockVector tractionVector = PetscBlockVector(tractionSizeLocal, numFaultElements, this->comm());


        // Vec col_du_dS;
        Vec col_dtau_dS;
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            for (int i = 0; i < faultSizeLocal; i++){
                std::cout << std::setprecision(1) << std::fixed << "Initialize derivatives of the stress components...   [" << 100.*(noFault*faultSizeLocal + i)/faultSizeGlobal << "%]        \r"<<std::flush;
                // set up unit vector e
                PetscBlockVector unitVector = PetscBlockVector(blockSize, numFaultElements, this->comm());
                // PetscBlockVector unitVector(zeroVector);
                auto AccessHandle = unitVector.begin_access();
                auto block = unitVector.get_block(AccessHandle, noFault);
                block(i) = 1.0;
                unitVector.end_access(AccessHandle);        

                // solve system Au - e = 0
                PetscBlockVector solutionVector = PetscBlockVector(quadratureSizeLocal, numFaultElements, this->comm());
                solutionVector.set_zero();

                adapter_->solveUnitVector(unitVector, solutionVector);

                // copy traction to vector
                scratch_.reset();
                auto in_handle = unitVector.begin_access_readonly();
                auto out_handle = tractionVector.begin_access();

                auto traction = Managed<Matrix<double>>(adapter_->traction_info());
                adapter_->begin_traction([&unitVector, &in_handle](std::size_t faultNo) {
                    return unitVector.get_block(in_handle, faultNo);
                });
                for (std::size_t faultNo = 0; faultNo < numFaultElements; ++faultNo) {
                    adapter_->traction_onlySlip(faultNo, traction, scratch_);

                    auto result_block = tractionVector.get_block(out_handle, faultNo);
                    auto result_mat = reshape(result_block, nbf, numTractionComps);

                    if (faultDim == 1){
                        for(std::size_t node = 0; node < nbf; node++){
                            result_mat(node,0) = traction(node,1);
                        }
                    } else if (faultDim == 2){
                        for(std::size_t node = 0; node < nbf; node++){
                            for(std::size_t c = 0; c < numTractionComps; c++){
                                result_mat(node,c) = traction(node,c);
                            }
                        }
                    }
                }
                adapter_->end_traction();
                unitVector.end_access_readonly(in_handle);
                tractionVector.end_access(out_handle);

                CHKERRTHROW(MatDenseGetColumnVecWrite(    dtau_dS_, noFault * faultSizeLocal + i, &col_dtau_dS));
                CHKERRTHROW(VecCopy(tractionVector.vec(), col_dtau_dS));
                CHKERRTHROW(MatDenseRestoreColumnVecWrite(dtau_dS_, noFault * faultSizeLocal + i, &col_dtau_dS));
            }
        }
        std::cout << "Initialize derivatives of the stress components...    done     "<< std::setprecision(6) << std::defaultfloat << std::endl;

    }

    /**
     * Estimate the error in the DG scheme for an error in psi.
     * This function calculates the product Cx1 as described in the thesis to estimate the error in the slip rate
     */
    void calculateCdotONE(){
        size_t numFaultElements = this->numLocalElements();
        size_t nbf = lop_->space().numBasisFunctions();
        size_t faultSizeGlobal = nbf * numFaultElements;

        Vec oneVector, result;
        VecCreate(comm(), &oneVector);
        VecSetType(oneVector, VECSEQ);
        VecSetSizes(oneVector, PETSC_DECIDE, faultSizeGlobal);
        VecDuplicate(oneVector, &result);

        for(int i=0; i < faultSizeGlobal; i++) VecSetValue(oneVector, i, 1.0, INSERT_VALUES);


        double traction_min = std::numeric_limits<double>::infinity();
        double traction_max = 0.0;

        MatMult(dtau_dS_, oneVector, result);
        const double * r;
        VecGetArrayRead(result, &r);
        for (int i = 0; i < faultSizeGlobal; ++i){
            traction_min = std::min(traction_min, std::abs(r[i]));
            traction_max = std::max(traction_max, std::abs(r[i]));
        }
        VecRestoreArrayRead(result, &r);
        std::cout << "range of traction: " << traction_min << ", "<< traction_max<< std::endl;
    }


    /**
     * For the second order ODE, calculate the normal stress with the discrete Green's function
     * @param state [in   ] current solution vector
     * @param time  [in   ] current simulation time 
     */
    template <typename BlockVector> 
    void calculateSigma(BlockVector& state, double time){
        size_t blockSize          = this->block_size();
        size_t numFaultElements   = this->numLocalElements();
        size_t nbf                = lop_->space().numBasisFunctions();
        size_t tractionDim        = adapter_->getNumberQuantities();
        size_t faultDim           = RateAndStateBase::TangentialComponents;
        size_t totalSize          = blockSize   * numFaultElements;
        size_t numFaultNodes      = nbf         * numFaultElements;
        size_t tractionSizeGlobal = tractionDim * numFaultNodes;
        size_t faultSizeGlobal    = faultDim    * numFaultNodes;

        int sig_x_i;
        int tau_y_i;
        int tau_z_i;
        int n_i, ny_i, nz_i;

        sigmaHelperVector_->set_zero();  

        Vec slip_vec, tau_vec;
        CHKERRTHROW(VecCreateSeq(comm(), faultSizeGlobal, &slip_vec));
        CHKERRTHROW(VecCreateSeq(comm(), tractionSizeGlobal, &tau_vec));

        // fill vector S
        double * S_array;
        CHKERRTHROW(VecGetArrayWrite(slip_vec, &S_array));
        auto state_handle = state.begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto slip = state.get_block(state_handle, noFault);    
            for(int i = 0; i<nbf; i++){ 
                ny_i    = noFault * faultDim * nbf + 0 * nbf + i;
                nz_i    = noFault * faultDim * nbf + 1 * nbf + i;

                S_array[ny_i] = slip(0 * nbf + i);
                S_array[nz_i] = slip(1 * nbf + i);
            }
        }
        CHKERRTHROW(VecRestoreArrayWrite(slip_vec, &S_array));
        state.end_access_readonly(state_handle);

        // calculate product dtau/dS * S
        CHKERRTHROW(MatMult(dtau_dS_, slip_vec, tau_vec));

        auto sigma_handle = sigmaHelperVector_->begin_access();
        const double * tau_array;
        CHKERRTHROW(VecGetArrayRead(tau_vec, &tau_array));
        auto dtaudt_handle = dtau_dt_->begin_access_readonly();
        // Calculate sigma
        for (int noFault_i = 0; noFault_i < numFaultElements; noFault_i++){

            auto sigma = sigmaHelperVector_->get_block(sigma_handle, noFault_i);
            auto const_dtaudt = dtau_dt_->get_block(dtaudt_handle, noFault_i);

            for (int i = 0; i < nbf; i++){
                sig_x_i = noFault_i * tractionDim * nbf + 0 * nbf + i;

                sigma(i) = const_dtaudt(i) * time + tau_array[sig_x_i]; 
            }
        }
        sigmaHelperVector_->end_access(sigma_handle);
        dtau_dt_->end_access_readonly(dtaudt_handle);
        CHKERRTHROW(VecRestoreArrayRead(tau_vec, &tau_array));
        CHKERRTHROW(VecDestroy(&slip_vec));
        CHKERRTHROW(VecDestroy(&tau_vec));
    }
    /**
     * Calculate the rhs of the slip rate as dV/dt = J * d(S,psi)/dt + \pdv{V}{t}
     * @param state [in   ] current state of the system
     * @param rhs   [inout] current rhs of the system 
     * @param time  [in   ] current time of the system
     */
    template <typename BlockVector>
    void calculateSlipRateExtendedODE(BlockVector& state, BlockVector& rhs, double time){
        size_t blockSize          = this->block_size();
        size_t numFaultElements   = this->numLocalElements();
        size_t nbf                = lop_->space().numBasisFunctions();
        size_t tractionDim        = adapter_->getNumberQuantities();
        size_t faultDim           = RateAndStateBase::TangentialComponents;
        size_t totalSize          = blockSize   * numFaultElements;
        size_t numFaultNodes      = nbf         * numFaultElements;
        size_t tractionSizeGlobal = tractionDim * numFaultNodes;
        size_t faultSizeGlobal    = faultDim    * numFaultNodes;

        using namespace Eigen;

        int df_dV_index   = 0;
        int df_dpsi_index = 1;
        int V_index       = 8;
        int f_index       = 10;
        int unit_y_index  = 11;
        int unit_z_index  = 12;
        int VTau_index    = 13;

        // calculate the derivative of the slip rate
        int S_i;
        int PSI_i;
        int V_i;
        int Vy_i;
        int Vz_i;
        int S_j;
        int Sy_i;
        int Sz_i;
        int sig_x_i;
        int tau_y_i;
        int tau_z_i;
        int n_i, ny_i, nz_i;
        int ix,iy,iz,ipsi,iVy,iVz;

        int Sy_j;
        int Sz_j;
        int ny_j;
        int nz_j;
        int n_j;
        int jy,jz;

        auto out_handle    = rhs.begin_access();
        auto quant_handle = JacobianQuantities_->begin_access_readonly();
        auto dtaudt_handle = dtau_dt_->begin_access_readonly();
        const double* dtau_dS;
        CHKERRTHROW(MatDenseGetArrayRead(dtau_dS_, &dtau_dS));
        if (faultDim == 1){
            Vec V_vec, H_vec;
            CHKERRTHROW(VecCreateSeq(comm(), numFaultNodes, &V_vec));
            CHKERRTHROW(VecDuplicate(V_vec, &H_vec));

            // fill vector V
            auto accessRead = JacobianQuantities_->begin_access_readonly();
            double * V_array;
            CHKERRTHROW(VecGetArrayWrite(V_vec, &V_array));
            for (int noFault = 0; noFault < numFaultElements; noFault++){
                auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);    
                for(int i = 0; i<nbf; i++){ 
                    V_array[noFault * nbf + i] = derBlock(V_index * nbf + i);
                }
            }
            JacobianQuantities_->end_access_readonly(accessRead);
            CHKERRTHROW(VecRestoreArrayWrite(V_vec, &V_array));

            // calculate product dtau/dS * dS/dt 
            CHKERRTHROW(MatMult(dtau_dS_, V_vec, H_vec));

            const double * H_array;
            CHKERRTHROW(VecGetArrayRead(H_vec, &H_array));

            // write vector H = ( dtau/dS * dS/dt + dF/dpsi * dpsi/dt + pdv{V}{t} ) / (dF/dV)
            for (int noFault_i = 0; noFault_i < numFaultElements; noFault_i++){

                auto dxdt         = rhs.get_block(out_handle, noFault_i);
                auto const_dtaudt = dtau_dt_->get_block(dtaudt_handle, noFault_i);
                auto quant        = JacobianQuantities_->get_block(quant_handle, noFault_i);

                for (int i = 0; i < nbf; i++){                  // row iteration
                    S_i   = 0 * nbf + i;
                    PSI_i = 1 * nbf + i;
                    V_i   = 2 * nbf + i;

                    n_i = noFault_i * nbf + i;
    
                    dxdt(V_i) = - 
                            (
                              + const_dtaudt(i) + H_array[n_i]
                              + quant(df_dpsi_index * nbf + i) * dxdt(PSI_i)
                            ) / quant(df_dV_index * nbf + i); // = (dF/dV)^-1 * dF/dpsi * dpsi/dt
                }
            }
            CHKERRTHROW(VecRestoreArrayRead(H_vec, &H_array));
            CHKERRTHROW(VecDestroy(&V_vec));
            CHKERRTHROW(VecDestroy(&H_vec));

        } else if (faultDim == 2){

            /*********************** CALCULATE Dtau/Dt **********************/
            Vec V_vec, DtauDx_dot_dxdt_vec;
            CHKERRTHROW(VecCreateSeq(comm(), faultSizeGlobal, &V_vec));
            CHKERRTHROW(VecCreateSeq(comm(), tractionSizeGlobal, &DtauDx_dot_dxdt_vec));

            // fill vector V
            double * V_array;
            CHKERRTHROW(VecGetArrayWrite(V_vec, &V_array));
            auto accessRead = JacobianQuantities_->begin_access_readonly();
            for (int noFault = 0; noFault < numFaultElements; noFault++){
                auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);    
                for(int i = 0; i<nbf; i++){ 
                    ny_i    = noFault * faultDim * nbf + 0 * nbf + i;
                    nz_i    = noFault * faultDim * nbf + 1 * nbf + i;

                    V_array[ny_i] = derBlock(unit_y_index * nbf + i) * derBlock(V_index * nbf + i);
                    V_array[nz_i] = derBlock(unit_z_index * nbf + i) * derBlock(V_index * nbf + i);
                }
            }
            CHKERRTHROW(VecRestoreArrayWrite(V_vec, &V_array));
            JacobianQuantities_->end_access_readonly(accessRead);

            // calculate product dtau/dS * dS/dt 
            CHKERRTHROW(MatMult(dtau_dS_, V_vec, DtauDx_dot_dxdt_vec));



            /*********************** CALCULATE DV/Dt **********************/
            // fill the rhs vector components
            const double * DtauDx_dot_dxdt_array;
            CHKERRTHROW(VecGetArrayRead(DtauDx_dot_dxdt_vec, &DtauDx_dot_dxdt_array));

            double VtauRatio, dtauy_dt, dtauz_dt, dtau_dt, dV_dt, pdv_V_t, f, V, unit_y, unit_z, df_dV, df_dpsi;
            for (int noFault_i = 0; noFault_i < numFaultElements; noFault_i++){

                auto dxdt         = rhs.get_block(out_handle, noFault_i);
                auto const_dtaudt = dtau_dt_->get_block(dtaudt_handle, noFault_i);
                auto quant        = JacobianQuantities_->get_block(quant_handle, noFault_i);

                for (int i = 0; i < nbf; i++){
                    ix   = i + 0*nbf;    // for dtau/dt components
                    iy   = i + 1*nbf;    // for dtau/dt components
                    iz   = i + 2*nbf;    // for dtau/dt components

                    ipsi = i + 2*nbf;    // for dV/dt components
                    iVy  = i + 3*nbf;    // for dV/dt components    
                    iVz  = i + 4*nbf;    // for dV/dt components  

                    Sy_i    = noFault_i * blockSize         + 0 * nbf + i;
                    Sz_i    = noFault_i * blockSize         + 1 * nbf + i;
                    PSI_i   = noFault_i * blockSize         + 2 * nbf + i;
                    Vy_i    = noFault_i * blockSize         + 3 * nbf + i;
                    Vz_i    = noFault_i * blockSize         + 4 * nbf + i;
                    n_i     = noFault_i * nbf                         + i;
                    sig_x_i = noFault_i * tractionDim * nbf + 0 * nbf + i;
                    tau_y_i = noFault_i * tractionDim * nbf + 1 * nbf + i;
                    tau_z_i = noFault_i * tractionDim * nbf + 2 * nbf + i;

                    f         = quant(f_index       * nbf + i);
                    V         = quant(V_index       * nbf + i);
                    unit_y    = quant(unit_y_index  * nbf + i);
                    unit_z    = quant(unit_z_index  * nbf + i);
                    df_dpsi   = quant(df_dpsi_index * nbf + i);
                    df_dV     = quant(df_dV_index   * nbf + i);
                    VtauRatio = quant(VTau_index    * nbf + i);

                    dtauy_dt = const_dtaudt(iy) + DtauDx_dot_dxdt_array[tau_y_i];
                    dtauz_dt = const_dtaudt(iz) + DtauDx_dot_dxdt_array[tau_z_i];

                    dtau_dt  = unit_y * dtauy_dt + unit_z * dtauz_dt;

                    pdv_V_t  = 
                        -(
                            + const_dtaudt(ix) * f  
                            + const_dtaudt(iy) * unit_y
                            + const_dtaudt(iz) * unit_z
                         ) / df_dV;
                    dV_dt    = pdv_V_t
                        -(
                            + DtauDx_dot_dxdt_array[sig_x_i] * f
                            + DtauDx_dot_dxdt_array[tau_y_i] * unit_y
                            + DtauDx_dot_dxdt_array[tau_z_i] * unit_z
                            + df_dpsi * dxdt(ipsi)
                         ) / df_dV;

                    dxdt(iVy) = VtauRatio * dtauy_dt + unit_y * (dV_dt - VtauRatio * dtau_dt);
                    dxdt(iVz) = VtauRatio * dtauz_dt + unit_z * (dV_dt - VtauRatio * dtau_dt);

                }
            }
            CHKERRTHROW(VecRestoreArrayRead(DtauDx_dot_dxdt_vec, &DtauDx_dot_dxdt_array));
            CHKERRTHROW(VecDestroy(&DtauDx_dot_dxdt_vec));
            CHKERRTHROW(VecDestroy(&V_vec));
        }

        rhs.end_access(out_handle);
        dtau_dt_->end_access_readonly(dtaudt_handle);
        JacobianQuantities_->end_access_readonly(quant_handle);

        CHKERRTHROW(MatDenseRestoreArrayRead(dtau_dS_, &dtau_dS));
    }


    /**
     * For the system Jx = b:
     * Reduce to a n x n system which is solved with the KSP from the settings
     * @param x_vec searched unknown
     * @param b_vec rhs Vector
     * @param J_mat partial Jacobian with block diagonal entries
     * @param ksp the linear solver
     */
    void ReducedKSPCompactDAE(Vec x_vec, Vec b_vec, Mat J_mat, KSP ksp){
        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t totalSize = blockSize * numFaultElements;
        size_t nbf = lop_->space().numBasisFunctions();
        size_t numFaultNodes = nbf * numFaultElements;

        int b_i,b_j,S_i,S_j,psi_i,psi_j,n_i,n_j,i,j;

        Vec y_vec;
        Vec f_vec;
        Vec helper_vec;
        Mat K_mat;

        CHKERRTHROW(VecCreateSeq(comm(), numFaultNodes, &y_vec));
        CHKERRTHROW(VecDuplicate(y_vec, &f_vec));

        CHKERRTHROW(MatDuplicate(dtau_dS_, MAT_COPY_VALUES, &K_mat));

        const double * J;
        const double * b;
        const double * y;
        double * K;
        double * f;
        double * x;


        CHKERRTHROW(MatDenseGetArrayRead(J_mat, &J));
        CHKERRTHROW(VecGetArrayRead(b_vec, &b));

        // transform the system from from Jx = b to Ky = f

        CHKERRTHROW(MatDenseGetArray(K_mat, &K));
        CHKERRTHROW(VecGetArray(f_vec, &f));
        for (b_i = 0; b_i < numFaultElements; b_i++){
            for (i = 0; i < nbf; i++){
                S_i   = b_i * blockSize + i;
                psi_i = b_i * blockSize + i + nbf;
                n_i   = b_i * nbf + i;

                f[n_i] = b[S_i] - J[S_i + psi_i * totalSize] / J[psi_i + psi_i * totalSize] * b[psi_i];
                for (b_j = 0; b_j < numFaultElements; b_j++){
                    for(j=0; j < nbf; j++){
                        S_j   = b_j * blockSize + j;
                        n_j   = b_j * nbf + j;

                        K[n_i + n_j*numFaultNodes] = J[S_i + S_j*totalSize];
                    }
                }
                K[n_i + n_i*numFaultNodes] -= J[S_i + psi_i*totalSize] * J[psi_i + S_i*totalSize] / J[psi_i + psi_i*totalSize];
           }
        }
        CHKERRTHROW(MatDenseRestoreArray(K_mat, &K));
        CHKERRTHROW(VecRestoreArray(f_vec, &f));

        // solve the reduced Ky = f        
        CHKERRTHROW(KSPSetOperators(ksp, K_mat, K_mat));
        CHKERRTHROW(KSPSolve(ksp, f_vec, y_vec));

        // write values to the solution vector by forward substitution
        // from Ky = f to Jx = b
        CHKERRTHROW(VecGetArrayRead(y_vec, &y));
        CHKERRTHROW(VecGetArray(x_vec, &x));
        for (b_i = 0; b_i < numFaultElements; b_i++){
            for (i = 0; i < nbf; i++){
                S_i   = b_i * blockSize +       i;
                psi_i = b_i * blockSize + nbf + i;
                n_i   = b_i * nbf + i;

                x[S_i] = y[n_i];
                x[psi_i] = (b[psi_i] - J[psi_i + S_i*totalSize] * y[n_i]) / J[psi_i + psi_i*totalSize];
            }
        }
        CHKERRTHROW(MatDenseRestoreArrayRead(J_mat, &J));
        CHKERRTHROW(VecRestoreArrayRead(y_vec, &y));    

        CHKERRTHROW(VecRestoreArrayRead(b_vec, &b));
        CHKERRTHROW(VecRestoreArray(x_vec, &x));

        CHKERRTHROW(VecDestroy(&y_vec));
        CHKERRTHROW(VecDestroy(&f_vec));
        CHKERRTHROW(MatDestroy(&K_mat));
    }

    /**
     * For the system Jx = b:
     * Reduce to a n x n system which is solved with the KSP from the settings
     * @param x_vec searched unknown
     * @param b_vec rhs Vector
     * @param J_mat partial Jacobian with block diagonal entries
     * @param ksp the linear solver
     */
    void ReducedKSPExtendedDAE(Vec x_vec, Vec b_vec, Mat J_mat, KSP ksp){
        size_t blockSize        = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t tractionDim      = adapter_->getNumberQuantities();
        size_t nbf              = lop_->space().numBasisFunctions();
        size_t faultDim         = RateAndStateBase::TangentialComponents;
        size_t reducedBlockSize = tractionDim       * nbf;
        size_t totalSize        = blockSize         * numFaultElements;
        size_t reducedSize      = reducedBlockSize  * numFaultElements;

        int b_i,b_j,S_i,Sy_i,Sz_i,S_j,psi_i,psi_j,V_i,n_i,n1_i,n2_i,n3_i,n_j,n1_j,n2_j,n3_j,i,j;

        Vec y_vec;
        Vec f_vec;
        Vec helper_vec;
        Mat K_mat;

        CHKERRTHROW(VecCreateSeq(comm(), reducedSize, &y_vec));
        CHKERRTHROW(VecDuplicate(y_vec, &f_vec));

        CHKERRTHROW(MatCreateSeqDense(comm(), reducedSize, reducedSize, NULL, &K_mat));        
        CHKERRTHROW(MatZeroEntries(K_mat));

        const double * J;
        const double * b;
        const double * y;
        double * K;
        double * f;
        double * x;


        CHKERRTHROW(MatDenseGetArrayRead(J_mat, &J));
        CHKERRTHROW(VecGetArrayRead(b_vec, &b));

        // transform the system from from Jx = b to Ky = f

        CHKERRTHROW(MatDenseGetArray(K_mat, &K));
        CHKERRTHROW(VecGetArray(f_vec, &f));
        if (faultDim == 1) {
            for (b_i = 0; b_i < numFaultElements; b_i++){
                for (i = 0; i < reducedBlockSize; i++){
                    S_i   = b_i * blockSize                      + i;
                    psi_i = b_i * blockSize + (faultDim+0) * nbf + i;
                    V_i   = b_i * blockSize + (faultDim+1) * nbf + i;
                    n_i   = b_i * reducedBlockSize               + i;

                    // f = b3 - J33 / J13 * b1 - J32 / J22 * (b2 - J23 / J13 * b1)
                    f[n_i] = b[V_i] 
                        - J[V_i + V_i * totalSize] / J[S_i + V_i * totalSize] * b[S_i]
                        - J[V_i + psi_i * totalSize] / J[psi_i + psi_i * totalSize] 
                            * (b[psi_i] - J[psi_i + V_i * totalSize] / J[S_i + V_i * totalSize] * b[S_i]);

                    // K11 = B - J33 * J11 / J13 + J32 * J23 * J11 / (J22 * J13)
                    for (b_j = 0; b_j < numFaultElements; b_j++){
                        for(j=0; j < reducedBlockSize; j++){
                            S_j   = b_j * blockSize        + j;
                            n_j   = b_j * reducedBlockSize + j;

                            K[n_i + n_j*reducedSize] = J[V_i + S_j*totalSize];
                        }
                    }
                    K[n_i + n_i*reducedSize] 
                        += -J[V_i + V_i * totalSize]     * J[S_i + S_i * totalSize]   / J[S_i + V_i * totalSize]
                        +  J[V_i + psi_i * totalSize]   * J[psi_i + V_i * totalSize] * J[S_i + S_i * totalSize] 
                        / (J[psi_i + psi_i * totalSize] * J[S_i + V_i * totalSize]);
                }
            }
        } else if (faultDim == 2) {
            for (b_i = 0; b_i < numFaultElements; b_i++){
                for (i = 0; i < nbf; i++){
                    Sy_i  = b_i * blockSize        + 0 * nbf + i;
                    Sz_i  = b_i * blockSize        + 1 * nbf + i;
                    psi_i = b_i * blockSize        + 2 * nbf + i;
                    V_i   = b_i * blockSize        + 3 * nbf + i;
                    n1_i  = b_i * reducedBlockSize + 0 * nbf + i;
                    n2_i  = b_i * reducedBlockSize + 1 * nbf + i;
                    n3_i  = b_i * reducedBlockSize + 2 * nbf + i;

                    // fz = bz1 -Jz13 / Jy13 * by1
                    f[n3_i] = b[Sz_i] 
                        -J[Sz_i + V_i * totalSize] / J[Sz_i + V_i * totalSize] * b[Sy_i];

                    // K21 = -Jz13 / Jy13 * Jy11
                    K[n3_i + n2_i*reducedSize] = 
                        -J[Sz_i + V_i * totalSize] / J[Sz_i + V_i * totalSize] * J[Sy_i + Sy_i * totalSize];

                    // K22 = Jz11
                    K[n3_i + n3_i*reducedSize] = J[Sz_i + Sz_i * totalSize];
                }
            }
        }

        CHKERRTHROW(MatDenseRestoreArray(K_mat, &K));
        CHKERRTHROW(VecRestoreArray(f_vec, &f));

        // solve the reduced Ky = f        
        CHKERRTHROW(KSPSetOperators(ksp, K_mat, K_mat));
        CHKERRTHROW(KSPSolve(ksp, f_vec, y_vec));

        // write values to the solution vector by forward substitution
        // from Ky = f to Jx = b
        CHKERRTHROW(VecGetArrayRead(y_vec, &y));
        CHKERRTHROW(VecGetArray(x_vec, &x));
        for (b_i = 0; b_i < numFaultElements; b_i++){
            for (i = 0; i < nbf; i++){
                S_i   = b_i * blockSize                      + i;
                psi_i = b_i * blockSize + (faultDim+0) * nbf + i;
                V_i   = b_i * blockSize + (faultDim+1) * nbf + i;
                n_i   = b_i * reducedBlockSize               + i;

                // x1
                x[S_i] = y[n_i];
                // x2 = (b2 - J23 * x3) / J22
                x[psi_i] = (b[psi_i] - J[psi_i + V_i * totalSize] * (b[S_i] - J[S_i + S_i * totalSize] * y[n_i]) / J[S_i + V_i * totalSize]) / J[psi_i + psi_i * totalSize];
                // x3 = (b1 - J11 * x1) / J13
                x[V_i] = (b[S_i] - J[S_i + S_i * totalSize] * y[n_i]) / J[S_i + V_i * totalSize];
            }
        }
        if (faultDim == 2){
            for (b_i = 0; b_i < numFaultElements; b_i++){
                for (i = 0; i < nbf; i++){
                    Sz_i  = b_i * blockSize        + 1 * nbf + i;
                    n3_i  = b_i * reducedBlockSize + 1 * nbf + i;

                    x[Sz_i] = y[n3_i];
                }
            }
        }

        CHKERRTHROW(MatDenseRestoreArrayRead(J_mat, &J));
        CHKERRTHROW(VecRestoreArrayRead(y_vec, &y));    

        CHKERRTHROW(VecRestoreArrayRead(b_vec, &b));
        CHKERRTHROW(VecRestoreArray(x_vec, &x));

        CHKERRTHROW(VecDestroy(&y_vec));
        CHKERRTHROW(VecDestroy(&f_vec));
        CHKERRTHROW(MatDestroy(&K_mat));
    }


    /**
     * Implements the Dirac delta function
     * @param i row index
     * @param j column index
     * @return 1 if on diagonal, 0 else
     */   
    double delta(int i, int j){
        return (i == j) ? 1.0 : 0.0;
    }


    /**
     * Approximates several Jacobian matrices in the system and compares them to the analytic expression
     * @param state current state of the system
     * @param rhs current rhs of the system
     * @param time current time of the system
     * @param full print full matrices with relative differences or only the max value
     */
     template <typename BlockVector>
     void testJacobianMatricesCompactODE(BlockVector& state, BlockVector& rhs, double time, bool full){

        using namespace Eigen;

        updateJacobianCompactODE(Jacobian_compact_ode_);

        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t tractionDim = adapter_->getNumberQuantities();
        size_t faultDim = RateAndStateBase::TangentialComponents;
        size_t totalSize = blockSize * numFaultElements;
        size_t nbf = lop_->space().numBasisFunctions();
        size_t numFaultNodes = nbf * numFaultElements;
        size_t tractionSizeGlobal = tractionDim * numFaultNodes;

        int n_i;
        int n_j;

        // fill vectors     
        VectorXd df_dV_vec(numFaultNodes);                             
        VectorXd df_dpsi_vec(numFaultNodes);
        VectorXd dg_dV_vec(numFaultNodes);                             
        VectorXd dg_dpsi_vec(numFaultNodes);

        auto accessRead = JacobianQuantities_->begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
                dg_dV_vec(   noFault * nbf + i ) = derBlock(2 * nbf + i);
                dg_dpsi_vec( noFault * nbf + i ) = derBlock(3 * nbf + i);
            }
        }

        JacobianQuantities_->end_access_readonly(accessRead);

        double max_rel_diff = 0;

        const double* F_x;
        CHKERRTHROW(MatDenseGetArrayRead(Jacobian_compact_ode_, &F_x));

        double J_approx;
        double J;


        std::ofstream JacobianFile;
        JacobianFile.open("Jacobian_matrices_t"+std::to_string(time));
        // --------------APPROXIMATE J ------------------- //
//       if (full) std::cout << "relative difference to the approximated J is: "<<std::endl<<"[ "; 
//        if (full) JacobianFile << "approximated J: \n [ "; 
//         for (int faultNo_j = 0; faultNo_j < 3; ++faultNo_j){
//             for (int j = 0; j < blockSize; j++){
//                 n_j = faultNo_j * blockSize + j;
//                 PetscBlockVector x_left(blockSize, numFaultElements, comm());
//                 PetscBlockVector x_right(blockSize, numFaultElements, comm());
//                 PetscBlockVector f_left(blockSize, numFaultElements, comm());
//                 PetscBlockVector f_right(blockSize, numFaultElements, comm());
//
//                 CHKERRTHROW(VecCopy(state.vec(), x_left.vec()));
//                 CHKERRTHROW(VecCopy(state.vec(), x_right.vec()));
//                
//                 auto x_r = x_right.begin_access();
//                 auto x_l = x_left.begin_access();
//                 const auto x = state.begin_access_readonly();
//
//                 auto x_l_block = x_left.get_block(x_l, faultNo_j);
//                 auto x_r_block = x_right.get_block(x_r, faultNo_j);
//                 auto x_block = rhs.get_block(x, faultNo_j);
//
//                 double h = 1e-8 + 1e-10 * abs(x_block(j));
//                 x_l_block(j) -= h;
//                 x_r_block(j) += h;
//
//                 x_left.end_access(x_l);
//                 x_right.end_access(x_r);
//                 state.end_access_readonly(x);
//
//                 rhsCompactODE(time, x_left, f_left, false);
//                 rhsCompactODE(time, x_right, f_right, false);
//
//                 const auto f_r = f_right.begin_access_readonly();
//                 const auto f_l = f_left.begin_access_readonly();
//
//                 for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
//                     auto f_l_block = f_left.get_block(f_l, faultNo_i);
//                     auto f_r_block = f_right.get_block(f_r, faultNo_i);
//                     for (int i = 0; i < blockSize; i++){                        
//                         n_i = faultNo_i * blockSize + i;
//
//                         J_approx = (f_r_block(i) - f_l_block(i)) / (2.0 * h);            
//                         J = F_x[n_i + n_j * totalSize];
//                         if (full) JacobianFile << J_approx << " ";
// //                        if (full) std::cout << (J_approx - J) / J_approx << " ";
//                         if (!full) max_rel_diff = std::max(max_rel_diff, 
//                                                   std::abs((J_approx - J) / J_approx));
//                     }
//                 }
//
//                 if (full) ((j+1 == blockSize) && (faultNo_j + 1 == 3)) ? JacobianFile << "]" : JacobianFile << "; "; 
//                 if (full) JacobianFile << "\n";
//                 f_left.end_access_readonly(f_l);
//                 f_right.end_access_readonly(f_r);
//             }            
//         }
        JacobianFile << "analytic Jacobian is: \n[ ";
        for (int faultNo_j = 0; faultNo_j < 3; ++faultNo_j){
            for (int j = 0; j < blockSize; j++){
                n_j = faultNo_j * blockSize + j;
                for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
                    for (int i = 0; i < blockSize; i++){                        
                        n_i = faultNo_i * blockSize + i;
                        J = F_x[n_i + n_j * totalSize];
                        if (full) JacobianFile << J << " ";
//                        if (full) std::cout << (J_approx - J) / J_approx << " ";
                        if (!full) max_rel_diff = std::max(max_rel_diff,
                                                  std::abs((J_approx - J) / J_approx));
                    }
                }

                if (full) ((j+1 == blockSize) && (faultNo_j + 1 == 3)) ? JacobianFile << "]" : JacobianFile << "; "; 
                if (full) JacobianFile << "\n";
            }
        }

        if (full) JacobianFile << "\n\n\n";
        if (!full) std::cout << "maximal relative difference between J and its approximate is " << max_rel_diff << "\n";

        CHKERRTHROW(MatDenseRestoreArrayRead(Jacobian_compact_ode_, &F_x));
        JacobianFile << std::flush;

        // --------------APPROXIMATE DF/DS------------------- //
        double dfdS_approx;
        double dfdS;
        const double* dtau_dS;
        max_rel_diff = 0; 
        CHKERRTHROW(MatDenseGetArrayRead(dtau_dS_, &dtau_dS));
        if (faultDim == 1){
            if (full) std::cout << "relative difference to the approximated df/dS is: "<<std::endl<<"[ ";        
            for (int faultNo_j = 0; faultNo_j < numFaultElements; ++faultNo_j){
                for (int j = 0; j < nbf; j++){
                    n_j = faultNo_j * nbf + j;

                    PetscBlockVector x_left(blockSize, numFaultElements, comm());
                    PetscBlockVector x_right(blockSize, numFaultElements, comm());
                    PetscBlockVector f_left(blockSize, numFaultElements, comm());
                    PetscBlockVector f_right(blockSize, numFaultElements, comm());

                    CHKERRTHROW(VecCopy(state.vec(), x_left.vec()));
                    CHKERRTHROW(VecCopy(state.vec(), x_right.vec()));
                    
                    auto x_r = x_right.begin_access();
                    auto x_l = x_left.begin_access();
                    const auto x = state.begin_access_readonly();

                    auto x_l_block = x_left.get_block(x_l, faultNo_j);
                    auto x_r_block = x_right.get_block(x_r, faultNo_j);
                    auto x_block = rhs.get_block(x, faultNo_j);

                    double h = 1e-4 + 1e-10 * abs(x_block(j));
                    x_l_block(j) -= h;
                    x_r_block(j) += h;

                    x_left.end_access(x_l);
                    x_right.end_access(x_r);
                    state.end_access_readonly(x);

                    // get the friction law to the left
                    adapter_->solve(time, x_left);

                    scratch_.reset();
                    auto in_handle = x_left.begin_access_readonly();
                    auto in_rhs_handle = rhs.begin_access();
                    auto traction = Managed<Matrix<double>>(adapter_->traction_info());
                    adapter_->begin_traction([&x_left, &in_handle](std::size_t faultNo) {
                        return x_left.get_block(in_handle, faultNo);
                    });
                    auto out_handle = f_left.begin_access();
                
                    for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
                        adapter_->traction(faultNo, traction, scratch_);

                        auto state_block = x_left.get_block(in_handle, faultNo);
                        auto state_der_block = rhs.get_block(in_rhs_handle, faultNo);
                        auto result_block = f_left.get_block(out_handle, faultNo);

                        lop_->applyFrictionLaw(faultNo, time, traction, state_block, state_der_block, result_block, scratch_);
                    }
                    adapter_->end_traction();
                    x_left.end_access_readonly(in_handle);
                    rhs.end_access(in_rhs_handle);
                    f_left.end_access(out_handle);

                    // get the friction law to the right
                    adapter_->solve(time, x_right);

                    in_handle = x_right.begin_access_readonly();
                    in_rhs_handle = rhs.begin_access();
                    traction = Managed<Matrix<double>>(adapter_->traction_info());
                    adapter_->begin_traction([&x_right, &in_handle](std::size_t faultNo) {
                        return x_right.get_block(in_handle, faultNo);
                    });
                    out_handle = f_right.begin_access();

                    for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
                        adapter_->traction(faultNo, traction, scratch_);

                        auto state_block = x_right.get_block(in_handle, faultNo);
                        auto state_der_block = rhs.get_block(in_rhs_handle, faultNo);
                        auto result_block = f_right.get_block(out_handle, faultNo);

                        lop_->applyFrictionLaw(faultNo, time, traction, state_block, state_der_block, result_block, scratch_);
                    }
                    adapter_->end_traction();
                    x_right.end_access_readonly(in_handle);
                    rhs.end_access(in_rhs_handle);
                    f_right.end_access(out_handle);


                    const auto f_r = f_right.begin_access_readonly();
                    const auto f_l = f_left.begin_access_readonly();

                    for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
                        auto f_l_block = f_left.get_block(f_l, faultNo_i);
                        auto f_r_block = f_right.get_block(f_r, faultNo_i);

                        for (int i = 0; i < nbf; i++){                        
                            n_i = faultNo_i * nbf + i;
                            dfdS_approx = (f_r_block(i) - f_l_block(i)) / (2.0 * h);                        
                            dfdS = dtau_dS[n_i + n_j * numFaultNodes];
                            if (full)  std::cout << (dfdS_approx - dfdS) / dfdS_approx<< " ";                            
                            if (!full) max_rel_diff = std::max(max_rel_diff, 
                                                        std::abs((dfdS_approx - dfdS) / dfdS_approx));
                        }
                    }
                    if (full) ((j+1 == nbf) && (faultNo_j + 1 == numFaultElements)) ? std::cout << "]" : std::cout << "; "; 
                    if (full) std::cout << std::endl;
                    f_left.end_access_readonly(f_l);
                    f_right.end_access_readonly(f_r);
                }
            }            
            if (full) std::cout << std::endl << std::endl << std::endl;
            if (!full) std::cout << "maximal relative difference between df/dS and its approximate is " << max_rel_diff << std::endl;
        } else {
//             if (full) JacobianFile << "approximated df/dS is: "<<std::endl<<"[ ";        
//             for (int faultNo_j = 0; faultNo_j < numFaultNodes; ++faultNo_j){
//                 for (int j = 0; j < faultDim * nbf; j++){
//                     n_j = faultNo_j * faultDim * nbf + j;
//
//                     PetscBlockVector x_left(blockSize, numFaultElements, comm());
//                     PetscBlockVector x_right(blockSize, numFaultElements, comm());
//                     PetscBlockVector f_left(tractionDim*nbf, numFaultElements, comm());
//                     PetscBlockVector f_right(tractionDim*nbf, numFaultElements, comm());
//
//                     CHKERRTHROW(VecCopy(state.vec(), x_left.vec()));
//                     CHKERRTHROW(VecCopy(state.vec(), x_right.vec()));
//                 
//                     auto x_r = x_right.begin_access();
//                     auto x_l = x_left.begin_access();
//                     const auto x = state.begin_access_readonly();
//
//                     auto x_l_block = x_left.get_block(x_l, faultNo_j);
//                     auto x_r_block = x_right.get_block(x_r, faultNo_j);
//                     auto x_block = rhs.get_block(x, faultNo_j);
//
//                     double h = 1e-4 + 1e-10 * abs(x_block(j));
//                     x_l_block(j) -= h;
//                     x_r_block(j) += h;
//
//                     x_left.end_access(x_l);
//                     x_right.end_access(x_r);
//                     state.end_access_readonly(x);
//
//                     // get the friction law to the left
//                     adapter_->solve(time, x_left);
//
//                     scratch_.reset();
//                     auto in_handle = x_left.begin_access_readonly();
//                     auto in_rhs_handle = rhs.begin_access();
//                     auto traction = Managed<Matrix<double>>(adapter_->traction_info());
//                     adapter_->begin_traction([&x_left, &in_handle](std::size_t faultNo) {
//                         return x_left.get_block(in_handle, faultNo);
//                     });
//                     auto out_handle = f_left.begin_access();
//             
//                     for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
//                         adapter_->traction(faultNo, traction, scratch_);
//
//                         auto result_block = f_left.get_block(out_handle, faultNo);
//
//                         lop_->getTractionComponents(faultNo, time, traction, result_block, scratch_);
//                     }
//                     adapter_->end_traction();
//                     x_left.end_access_readonly(in_handle);
//                     rhs.end_access(in_rhs_handle);
//                     f_left.end_access(out_handle);
//
//                     // get the friction law to the right
//                     adapter_->solve(time, x_right);
//
//                     in_handle = x_right.begin_access_readonly();
//                     in_rhs_handle = rhs.begin_access();
//                     traction = Managed<Matrix<double>>(adapter_->traction_info());
//                     adapter_->begin_traction([&x_right, &in_handle](std::size_t faultNo) {
//                         return x_right.get_block(in_handle, faultNo);
//                     });
//                     out_handle = f_right.begin_access();
//
//                     for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
//                         adapter_->traction(faultNo, traction, scratch_);
//
//                         auto result_block = f_right.get_block(out_handle, faultNo);
//
//                         lop_->getTractionComponents(faultNo, time, traction, result_block, scratch_);
//                     }
//                     adapter_->end_traction();
//                     x_right.end_access_readonly(in_handle);
//                     rhs.end_access(in_rhs_handle);
//                     f_right.end_access(out_handle);
//
//
//                     const auto f_r = f_right.begin_access_readonly();
//                     const auto f_l = f_left.begin_access_readonly();
//
//                     for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
//                         auto f_l_block = f_left.get_block(f_l, faultNo_i);
//                         auto f_r_block = f_right.get_block(f_r, faultNo_i);
//
//                         for (int i = 0; i < tractionDim*nbf; i++){                        
//                             n_i = faultNo_i * tractionDim*nbf  + i;
//                             dfdS_approx = (f_r_block(i) - f_l_block(i)) / (2.0 * h);                        
//                             dfdS = dtau_dS[n_i + n_j * tractionSizeGlobal];
//                             JacobianFile << dfdS_approx << ", ";
// //                            if (full)  JacobianFile << (dfdS_approx - dfdS) / dfdS_approx<< " ";                            
//                             if (!full) max_rel_diff = std::max(max_rel_diff, 
//                                                         std::abs((dfdS_approx - dfdS) / dfdS_approx));
//                         }
//                     }
//                            if (full) ((j+1 == faultDim*nbf) && (faultNo_j + 1 == numFaultNodes)) ? JacobianFile << "]" : JacobianFile << "; "; 
//                     if (full) JacobianFile << std::endl;
//                     f_left.end_access_readonly(f_l);
//                     f_right.end_access_readonly(f_r);
//                 }
//             }            
//             if (full) JacobianFile << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
//             if (!full) std::cout << "maximal relative difference between df/dS and its approximate is " << max_rel_diff << std::endl;
//
//             if (full) JacobianFile << "analytic df/dS is: \n[ ";        
//             for (int faultNo_j = 0; faultNo_j < numFaultNodes; ++faultNo_j){
//                 for (int j = 0; j < faultDim*nbf; j++){
//                     n_j = faultNo_j * faultDim*nbf + j;
//                     for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
//                         for (int i = 0; i < tractionDim*nbf; i++){                        
//                             n_i = faultNo_i * tractionDim*nbf  + i;
//                             dfdS = dtau_dS[n_i + n_j * tractionSizeGlobal];
//                             JacobianFile << dfdS << ", ";
// //                            if (full)  JacobianFile << (dfdS_approx - dfdS) / dfdS_approx<< " ";                            
//                             if (!full) max_rel_diff = std::max(max_rel_diff, 
//                                                         std::abs((dfdS_approx - dfdS) / dfdS_approx));
//                         }
//                     }
//                             if (full) ((j+1 == faultDim*nbf) && (faultNo_j + 1 == numFaultNodes)) ? JacobianFile << "]" : JacobianFile << "; "; 
//                             if (full) JacobianFile << std::endl;
//                 }
//             }            
        }
        CHKERRTHROW(MatDenseRestoreArrayRead(dtau_dS_, &dtau_dS));



        // // --------------APPROXIMATE DF/DV------------------- //

        // double dfdV_approx;
        // double dfdV;
        // max_rel_diff = 0; 
        // if (full) std::cout << "relative difference to the approximated df/dV is: "<<std::endl<<"[ ";        
        // for (int faultNo_j = 0; faultNo_j < numFaultElements; ++faultNo_j){
        //     for (int j = 0; j < nbf; j++){
        //         n_j = faultNo_j * nbf + j;

        //         PetscBlockVector f_left(blockSize, numFaultElements, comm());
        //         PetscBlockVector f_right(blockSize, numFaultElements, comm());

        //         double h;

        //         // get the friction law
        //         scratch_.reset();
        //         auto in_handle = state.begin_access_readonly();
        //         auto in_rhs_handle = rhs.begin_access();
        //         auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        //         adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
        //             return state.get_block(in_handle, faultNo);
        //         });
        //         auto out_left_handle = f_left.begin_access();
        //         auto out_right_handle = f_right.begin_access();

        //         for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
        //             adapter_->traction(faultNo, traction, scratch_);

        //             auto state_block = state.get_block(in_handle, faultNo);
        //             auto state_der_block = rhs.get_block(in_rhs_handle, faultNo);
        //             auto result_left_block = f_left.get_block(out_left_handle, faultNo);
        //             auto result_right_block = f_right.get_block(out_right_handle, faultNo);

        //             h = 1e-2 * abs(state_der_block(j));
        //             if (faultNo == faultNo_j) state_der_block(j) -= h;
        //             lop_->applyFrictionLaw(faultNo, time, traction, state_block, state_der_block, result_left_block, scratch_);
        //             if (faultNo == faultNo_j) state_der_block(j) += h;
        //             if (faultNo == faultNo_j) state_der_block(j) += h;
        //             lop_->applyFrictionLaw(faultNo, time, traction, state_block, state_der_block, result_right_block, scratch_);
        //             if (faultNo == faultNo_j) state_der_block(j) -= h;
        //         }
        //         adapter_->end_traction();
        //         state.end_access_readonly(in_handle);
        //         rhs.end_access(in_rhs_handle);
        //         f_left.end_access(out_left_handle);
        //         f_right.end_access(out_right_handle);


        //         const auto f_r = f_right.begin_access_readonly();
        //         const auto f_l = f_left.begin_access_readonly();

        //         for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
        //             auto f_l_block = f_left.get_block(f_l, faultNo_i);
        //             auto f_r_block = f_right.get_block(f_r, faultNo_i);

        //             for (int i = 0; i < nbf; i++){     
        //                 n_i = faultNo_i * nbf + i;                                           
        //                 dfdV_approx = (f_r_block(i) - f_l_block(i)) / (2.0 * h);
        //                 dfdV = df_dV * delta(n_i, n_j);
        //                 if (full) std::cout << (dfdV_approx - dfdV) / dfdV_approx <<" ";                           
        //                 if (!full) max_rel_diff = std::max(max_rel_diff, 
        //                                           std::abs((dfdV_approx - dfdV) / dfdV_approx)); 
        //             }
        //         }
        //         if (full) ((j+1 == nbf) && (faultNo_j + 1 == numFaultElements)) ? std::cout << "]" : std::cout << "; "; 
        //         if (full) std::cout << std::endl;
        //         f_left.end_access_readonly(f_l);
        //         f_right.end_access_readonly(f_r);
        //     }            
        // }
        // if (full) std::cout << std::endl << std::endl << std::endl;
        // if (!full) std::cout << "maximal relative difference between df/dV and its approximate is " << max_rel_diff << std::endl;

        // // ------------------APPROXIMATE DV/Dt ---------------------------- //
        // PetscBlockVector x_left(blockSize, numFaultElements, comm());
        // PetscBlockVector x_right(blockSize, numFaultElements, comm());
        // PetscBlockVector f_left(blockSize, numFaultElements, comm());
        // PetscBlockVector f_right(blockSize, numFaultElements, comm());

        // CHKERRTHROW(VecCopy(state.vec(), x_left.vec()));
        // CHKERRTHROW(VecCopy(state.vec(), x_right.vec()));

        // double h = 1e-5;

        // CHKERRTHROW(VecAXPY(x_right.vec(), h, rhs.vec()));
        // CHKERRTHROW(VecAXPY(x_left.vec(), -h, rhs.vec()));

        // rhsCompactODE(time-h, x_left, f_left, false);
        // rhsCompactODE(time+h, x_right, f_right, false);


        // // calculate the derivative of the slip rate
        // int S_i;
        // int PSI_i;
        // int S_j;

        // auto out_handle = rhs.begin_access();

        // auto left_handle = f_left.begin_access_readonly();
        // auto right_handle = f_right.begin_access_readonly();

        // double dVdt;
        // double dVdt_approx;

        // CHKERRTHROW(MatDenseGetArrayRead(dtau_dS_, &dtau_dS));
        // max_rel_diff = 0; 
        // if (full) std::cout << "relative difference to the approximated dV/dt is: " << std::endl;
        // for (int noFault = 0; noFault < numFaultElements; noFault++){
        //     auto dPSIdt = rhs.get_block(out_handle, noFault);

        //     auto lb = f_left.get_block(left_handle, noFault);
        //     auto rb = f_right.get_block(right_handle, noFault);

        //    for (int i = 0; i < nbf; i++){                  // row iteration
        //         S_i   = i;
        //         PSI_i =  RateAndStateBase::TangentialComponents * nbf + i;

        //         n_i = noFault * nbf + i;
 
        //         dVdt = -df_dpsi_vec(n_i) * dPSIdt(PSI_i);  // = -dF/dpsi * dpsi/dt
                
        //         for (int noFault_j = 0; noFault_j < numFaultElements; noFault_j++){ 
        //             auto dSdt = rhs.get_block(out_handle, noFault_j);
              
        //             for(int j = 0; j < nbf; j++) {          // column iteration

        //                 n_j = noFault_j * nbf + j;
        //                 S_j = j;

        //                 dVdt -= dtau_dS[n_i + n_j * numFaultNodes] * dSdt(S_j);    // = -dF/dS * dS/dt
        //             }
        //         }
        //         dVdt /= df_dV_vec(n_i); // = (dF/dV)^-1 * dF/dt

        //         dVdt_approx = (rb(i) - lb(i)) / (2.0 * h);
        //         if (full) std::cout << (dVdt - dVdt_approx) / dVdt_approx << std::endl;
        //         max_rel_diff = std::max(max_rel_diff,std::abs((dVdt - dVdt_approx) / dVdt_approx));
        //     }
        // }
        // if (full) std::cout << std::endl << std::endl << std::endl;
        // if (!full) std::cout << "maximal relative difference between dV/dt and its approximate is " << max_rel_diff << std::endl;

        // rhs.end_access_readonly(out_handle);
        // f_left.end_access_readonly(left_handle);
        // f_right.end_access_readonly(right_handle);
        // CHKERRTHROW(MatDenseRestoreArrayRead(dtau_dS_, &dtau_dS));

    }

    /**
     * Approximates several Jacobian matrices in the system and compares them to the analytic expression
     * @param state current state of the system
     * @param state_der current state derivative of the system
     * @param rhs current rhs of the system
     * @param time current time of the system
     * @param full print full matrices with relative differences or only the max value
     */
     template <typename BlockVector>
     void testJacobianMatricesExtendedDAE(BlockVector& state, BlockVector& state_der, BlockVector& rhs, double time, bool full){

        using namespace Eigen;

        double sigma = 1e1;

        updateJacobianExtendedDAE(sigma, Jacobian_extended_dae_);

        size_t blockSize          = this->block_size();
        size_t numFaultElements   = this->numLocalElements();
        size_t tractionDim        = adapter_->getNumberQuantities();
        size_t faultDim           = RateAndStateBase::TangentialComponents;
        size_t totalSize          = blockSize * numFaultElements;
        size_t nbf                = lop_->space().numBasisFunctions();
        size_t numFaultNodes      = nbf * numFaultElements;
        size_t tractionSizeGlobal = tractionDim * numFaultNodes;

        int n_i;
        int n_j;

        // fill vectors     
        VectorXd df_dV_vec(numFaultNodes);                             
        VectorXd df_dpsi_vec(numFaultNodes);
        VectorXd dg_dV_vec(numFaultNodes);                             
        VectorXd dg_dpsi_vec(numFaultNodes);

        auto accessRead = JacobianQuantities_->begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
                dg_dV_vec(   noFault * nbf + i ) = derBlock(2 * nbf + i);
                dg_dpsi_vec( noFault * nbf + i ) = derBlock(3 * nbf + i);
            }
        }

        JacobianQuantities_->end_access_readonly(accessRead);

        double max_rel_diff = 0;

        const double* F_x;
        CHKERRTHROW(MatDenseGetArrayRead(Jacobian_extended_dae_, &F_x));

        double J_approx;
        double J;


        std::ofstream JacobianFile;
        JacobianFile.open("Jacobian_matrices_t"+std::to_string(time));
        // --------------APPROXIMATE J ------------------- //
       if (full) JacobianFile << "approximated J: \n [ "; 
        for (int faultNo_j = 0; faultNo_j < 3; ++faultNo_j){
            for (int j = 0; j < blockSize; j++){
                n_j = faultNo_j * blockSize + j;
                PetscBlockVector x_old(blockSize, numFaultElements, comm());
                PetscBlockVector xd_old(blockSize, numFaultElements, comm());
                PetscBlockVector x_left(blockSize, numFaultElements, comm());
                PetscBlockVector x_right(blockSize, numFaultElements, comm());
                PetscBlockVector xd_left(blockSize, numFaultElements, comm());
                PetscBlockVector xd_right(blockSize, numFaultElements, comm());
                PetscBlockVector f_left(blockSize, numFaultElements, comm());
                PetscBlockVector f_right(blockSize, numFaultElements, comm());
                PetscBlockVector fd_left(blockSize, numFaultElements, comm());
                PetscBlockVector fd_right(blockSize, numFaultElements, comm());

                CHKERRTHROW(VecCopy(state.vec(), x_old.vec()));
                CHKERRTHROW(VecCopy(state.vec(), x_left.vec()));
                CHKERRTHROW(VecCopy(state.vec(), x_right.vec()));

                CHKERRTHROW(VecCopy(state_der.vec(), xd_old.vec()));
                CHKERRTHROW(VecCopy(state_der.vec(), xd_left.vec()));
                CHKERRTHROW(VecCopy(state_der.vec(), xd_right.vec()));
               
                auto x_r = x_right.begin_access();
                auto x_l = x_left.begin_access();
                const auto x = state.begin_access_readonly();

                auto xd_r = xd_right.begin_access();
                auto xd_l = xd_left.begin_access();
                const auto xd = state_der.begin_access_readonly();

                auto x_l_block = x_left.get_block(x_l, faultNo_j);
                auto x_r_block = x_right.get_block(x_r, faultNo_j);
                auto x_block = state.get_block(x, faultNo_j);

                auto xd_l_block = xd_left.get_block(xd_l, faultNo_j);
                auto xd_r_block = xd_right.get_block(xd_r, faultNo_j);
                auto xd_block = state_der.get_block(xd, faultNo_j);

                double h = 1e-8 + 1e-10 * abs(x_block(j));
                x_l_block(j) -= h;
                x_r_block(j) += h;

                double hd = 1e-8 + 1e-10 * abs(xd_block(j));
                xd_l_block(j) -= hd;
                xd_r_block(j) += hd;

                x_left.end_access(x_l);
                x_right.end_access(x_r);
                state.end_access_readonly(x);

                xd_left.end_access(xd_l);
                xd_right.end_access(xd_r);
                state_der.end_access_readonly(xd);

                lhsExtendedDAE(time, x_left, xd_old, f_left, false);
                lhsExtendedDAE(time, x_right, xd_old, f_right, false);

                lhsExtendedDAE(time, x_old, xd_left, fd_left, false);
                lhsExtendedDAE(time, x_old, xd_right, fd_right, false);

                const auto f_r = f_right.begin_access_readonly();
                const auto f_l = f_left.begin_access_readonly();

                const auto fd_r = fd_right.begin_access_readonly();
                const auto fd_l = fd_left.begin_access_readonly();

                for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
                    auto f_l_block = f_left.get_block(f_l, faultNo_i);
                    auto f_r_block = f_right.get_block(f_r, faultNo_i);
                    auto fd_l_block = fd_left.get_block(fd_l, faultNo_i);
                    auto fd_r_block = fd_right.get_block(fd_r, faultNo_i);
                    for (int i = 0; i < blockSize; i++){                        
                        n_i = faultNo_i * blockSize + i;

                        J_approx = (f_r_block(i)  - f_l_block(i))  / (2.0 * h)
                                 + (fd_r_block(i) - fd_l_block(i)) / (2.0 * hd) * sigma;
                        J = F_x[n_i + n_j * totalSize];
                        if (full) JacobianFile << J_approx << " ";
                        if (!full) max_rel_diff = std::max(max_rel_diff, 
                                                  std::abs((J_approx - J) / J_approx));
                    }
                }

                if (full) ((j+1 == blockSize) && (faultNo_j + 1 == 3)) ? JacobianFile << "]" : JacobianFile << "; "; 
                if (full) JacobianFile << "\n";
                f_left.end_access_readonly(f_l);
                f_right.end_access_readonly(f_r);
                fd_left.end_access_readonly(fd_l);
                fd_right.end_access_readonly(fd_r);
            }            
        }
        if (full){
            JacobianFile << "analytic Jacobian is: \n[ ";
            for (int faultNo_j = 0; faultNo_j < 3; ++faultNo_j){
                for (int j = 0; j < blockSize; j++){
                    n_j = faultNo_j * blockSize + j;
                    for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
                        for (int i = 0; i < blockSize; i++){                        
                            n_i = faultNo_i * blockSize + i;
                            J = F_x[n_i + n_j * totalSize];
                            if (full) JacobianFile << J << " ";
                        }
                    }

                    ((j+1 == blockSize) && (faultNo_j + 1 == 3)) ? JacobianFile << "]" : JacobianFile << "; "; 
                    JacobianFile << "\n";
                }
            }
            JacobianFile << "\n\n\n";
        }

        CHKERRTHROW(MatDenseRestoreArrayRead(Jacobian_extended_dae_, &F_x));
        JacobianFile << std::flush;
     }

    /**
     * Approximates several Jacobian matrices in the system and compares them to the analytic expression
     * @param state current state of the system
     * @param rhs current rhs of the system
     * @param time current time of the system
     * @param full print full matrices with relative differences or only the max value
     */
     template <typename BlockVector>
     void testJacobianMatricesExtendedODE(BlockVector& state, BlockVector& rhs, double time, bool full){

        using namespace Eigen;

        updateJacobianExtendedODE(Jacobian_extended_ode_);

        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t nbf = lop_->space().numBasisFunctions();
        size_t totalSize = (blockSize - nbf) * numFaultElements;
        size_t numFaultNodes = nbf * numFaultElements;

        int n_i;
        int n_j;

        double max_rel_diff = 0;

        const double* F_x;
        CHKERRTHROW(MatDenseGetArrayRead(Jacobian_extended_ode_, &F_x));

        double J_approx;
        double J;

        // --------------APPROXIMATE J ------------------- //
       if (full) std::cout << "relative difference to the approximated J is: "<<std::endl<<"[ "; 
        for (int faultNo_j = 0; faultNo_j < numFaultElements; ++faultNo_j){
            for (int j = 0; j < blockSize; j++){
                n_j = faultNo_j * blockSize + j;
                PetscBlockVector x_left(blockSize, numFaultElements, comm());
                PetscBlockVector x_right(blockSize, numFaultElements, comm());
                PetscBlockVector f_left(blockSize, numFaultElements, comm());
                PetscBlockVector f_right(blockSize, numFaultElements, comm());

                CHKERRTHROW(VecCopy(state.vec(), x_left.vec()));
                CHKERRTHROW(VecCopy(state.vec(), x_right.vec()));
                
                auto x_r = x_right.begin_access();
                auto x_l = x_left.begin_access();
                const auto x = state.begin_access_readonly();

                auto x_l_block = x_left.get_block(x_l, faultNo_j);
                auto x_r_block = x_right.get_block(x_r, faultNo_j);
                auto x_block = rhs.get_block(x, faultNo_j);

                double h = 1e-5 * abs(x_block(j));
                x_l_block(j) -= h;
                x_r_block(j) += h;

                x_left.end_access(x_l);
                x_right.end_access(x_r);
                state.end_access_readonly(x);

                rhsExtendedODE(time, x_left, f_left, false);
                rhsExtendedODE(time, x_right, f_right, false);

                const auto f_r = f_right.begin_access_readonly();
                const auto f_l = f_left.begin_access_readonly();

                for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
                    auto f_l_block = f_left.get_block(f_l, faultNo_i);
                    auto f_r_block = f_right.get_block(f_r, faultNo_i);
                    for (int i = 0; i < blockSize; i++){                        
                        n_i = faultNo_i * blockSize + i;

                        J_approx = (f_r_block(i) - f_l_block(i)) / (2.0 * h);            
                        J = F_x[n_i + n_j * totalSize];
//                        if (full) std::cout << J_approx<< " ";
                        if (full) std::cout << (J_approx?(J_approx - J) / J_approx:0) << " ";
                        if (!full) max_rel_diff = std::max(max_rel_diff, 
                                                  std::abs((J_approx?(J_approx - J) / J_approx:0)));
                        if (abs(J_approx?(J_approx - J) / J_approx:0)>1) std::cout << "J: " << J << ", J_a: " << J_approx << " at " << n_i << ", " << n_j << std::endl;
                    }
                }

                if (full) ((j+1 == blockSize) && (faultNo_j+1 == numFaultElements)) ? std::cout << "]" : std::cout << "; "; 
                if (full) std::cout << std::endl;
                f_left.end_access_readonly(f_l);
                f_right.end_access_readonly(f_r);
            }            
        }
        if (full) std::cout << std::endl << std::endl << std::endl;
        if (!full) std::cout << "maximal relative difference between J and its approximate is " << max_rel_diff << std::endl;

        CHKERRTHROW(MatDenseRestoreArrayRead(Jacobian_extended_ode_, &F_x));
    }

    std::unique_ptr<LocalOperator> lop_;    // on fault: rate and state instance (handles ageing law and slip_rate)
    std::unique_ptr<SeasAdapter> adapter_;  // on domain: DG solver (handles traction and mechanical solver)
    Scratch<double> scratch_;               // some aligned scratch memory
    double VMax_ = 0.0;                     // metrics: maximal velocity among all fault elements
    double error_extended_ODE_ = -1.0;      // metrics: evaluation of the maximum absolute value of the friction law 
    size_t evaluation_rhs_count = 0;        // metrics: counts the number of calls of the rhs function in one time step

    PetscBlockVector* JacobianQuantities_;  // contains the partial derivatives [df/dV, df/dpsi, dg/dV, dg/dpsi]
    PetscBlockVector* dtau_dt_;             // partial derivative dV/dt (constant, needed to construct  the real Jacobian)     
    PetscBlockVector* sigmaHelperVector_;   // Contains the value of the normal stress calculated with the discrete Green's function
    Mat dtau_dS_;                           // Jacobian dtau/dS (constant, needed to construct  the real Jacobian)
    Mat dtau_dS_L_;                         // L in the LU of df/dS (constant, needed to construct  the real Jacobian)     
    Mat dtau_dS_U_;                         // U in the LU of df/dS (constant, needed to construct  the real Jacobian)     
    Mat Jacobian_compact_ode_;              // Jacobian matrix of the compact ODE system
    Mat Jacobian_extended_ode_;             // Jacobian matrix of the extended ODE system
    Mat Jacobian_compact_dae_;              // Jacobian matrix sigma*F_xdot + F_x     
    Mat Jacobian_extended_dae_;             // Jacobian matrix sigma*F_xdot + F_x 
    Mat Jacobian_approx_;                   // approximated Jacobian matrix    

    solverParametersASandEQ solverParameters_;
    const std::optional<tndm::SolverConfigGeneral>& solverGenCfg_;    // general solver configuration   
    const std::optional<tndm::SolverConfigSpecific>& solverEqCfg_;    // solver configuration in earthquake   
    const std::optional<tndm::SolverConfigSpecific>& solverAsCfg_;    // solver configuration in aseismic slip

    Vec x_prev_;                            // solution at the previous timestep (unused)
    Vec rhs_prev_;                          // rhs evaluation at the previous timestep (unused)

};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
