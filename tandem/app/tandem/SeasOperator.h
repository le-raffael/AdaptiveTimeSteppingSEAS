#ifndef SEASOPERATOR_20201001_H
#define SEASOPERATOR_20201001_H

#include "form/BoundaryMap.h"
#include "geometry/Curvilinear.h"

#include "tandem/RateAndStateBase.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <petscts.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petscsnes.h>
#include <petsc/private/snesimpl.h>   
#include <petsc/private/kspimpl.h>
//#include <../src/snes/impls/ls/lsimpl.h>

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

        double time_eq = 0.;              // stores the time at the beginning of the earthquake 

        double FMax = 0.0;                // Maximum residual - only for output
        double ratio_addition = 0.0;      // ratio (sigma * df/dS) / (df/dV) - only for output

        std::optional<tndm::SolverConfigSpecific> current_solver_cfg;

        std::shared_ptr<PetscBlockVector> state_compact;        
        std::shared_ptr<PetscBlockVector> state_extended;        

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
        solverGenCfg_(cfg), solverEqCfg_(cfg->solver_earthquake), solverAsCfg_(cfg->solver_aseismicslip) {
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
     * Destructor
     */
    ~SeasOperator() {
        MatDestroy(&Jacobian_compact_ode_);
    }

    /**
     * Initialize the system 
     *  - apply initial conditions on the local operator
     *  - solve the system once
     * @param vector solution vector to be initialized [S,psi]
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
            lop_->initCompact(faultNo, traction, B, scratch);
        }
        adapter_->end_traction();
        vector.end_access(access_handle);

        // Initialize the Jacobian matrix
        JacobianQuantities_ = new PetscBlockVector(4 * lop_->space().numBasisFunctions(), numLocalElements(), comm());
    }

    /**
     * transform a compact to an extended formulation
     *  - calculate slip rate by solving the system
     * @param time current simulation time
     * @param vector_small solution vector to be written from [S,psi]
     * @param vector_big solution vector to be written to [S,psi,V]
     */
    template <class BlockVector> void makeSystemBig(double time, BlockVector& vector_small, BlockVector& vector_big) {
        adapter_->solve(time, vector_small);

        auto scratch = make_scratch();
        auto small_handle = vector_small.begin_access();
        auto big_handle = vector_big.begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&vector_small, &small_handle](std::size_t faultNo) {
            return vector_small.get_block(const_cast<typename BlockVector::const_handle>(small_handle),
                                    faultNo);
        });
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto S = vector_small.get_block(small_handle, faultNo);
            auto B = vector_big.get_block(big_handle, faultNo);
            lop_->makeBig(faultNo, traction, S, B, scratch);            
        }
        adapter_->end_traction();
        vector_small.end_access(small_handle);
        vector_big.end_access(big_handle);
    }

    /**
     * transform an extended to a compact formulation
     *  - calculate slip rate by solving the system
     * @param vector_small solution vector to be written from [S,psi]
     * @param vector_big solution vector to be written to [S,psi,V]
     */
    template <class BlockVector> void makeSystemSmall(BlockVector& vector_small, BlockVector& vector_big) {
        auto small_handle = vector_small.begin_access();
        auto big_handle = vector_big.begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto S = vector_small.get_block(small_handle, faultNo);
            auto B = vector_big.get_block(big_handle, faultNo);
            lop_->makeSmall(faultNo, S, B);            
        }
        vector_small.end_access(small_handle);
        vector_big.end_access(big_handle);
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
        adapter_->solve(time, state);

        auto scratch = make_scratch();
        auto in_handle = state.begin_access_readonly();
        auto out_handle = result.begin_access();

        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        auto outJac_handle = JacobianQuantities_->begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto state_block = state.get_block(in_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities_->get_block(outJac_handle, faultNo);

            double VMax = lop_->rhsCompactODE(faultNo, time, traction, state_block, result_block, scratch);

            lop_->getJacobianQuantitiesCompact(faultNo, time, traction, state_block, result_block, JacobianQuantities_block, scratch);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities_->end_access(outJac_handle);

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
        if (error_extended_ODE_ >= 0) adapter_->solve(time, state);

        auto scratch = make_scratch();
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
        auto outJac_handle = JacobianQuantities_->begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            if (error_extended_ODE_ >= 0) adapter_->traction(faultNo, traction, scratch);

            auto state_block = state.get_block(in_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities_->get_block(outJac_handle, faultNo);

            double VMax = lop_->rhsExtendedODE(faultNo, time, state_block, result_block, scratch, solverParameters_.checkAS);

            lop_->getJacobianQuantitiesExtended(faultNo, time, traction, state_block, JacobianQuantities_block, scratch);

            if (error_extended_ODE_ >= 0) error_extended_ODE_ = std::max(error_extended_ODE_, 
                lop_->applyMaxFrictionLaw(faultNo, time, traction, state_block, scratch));

            VMax_ = std::max(VMax_, VMax);
        }
        if (error_extended_ODE_ >= 0) adapter_->end_traction();
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities_->end_access(outJac_handle);

        // calculate the misssing entry dV/dt
        calculateSlipRateExtendedODE(result, time);

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

        auto scratch = make_scratch();
        auto in_handle = state.begin_access_readonly();
        auto in_der_handle = state_der.begin_access_readonly();
        auto out_handle = result.begin_access();
        
        auto outJac_handle = JacobianQuantities_->begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        double FMax = 0.0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto state_block = state.get_block(in_handle, faultNo);
            auto state_der_block = state_der.get_block(in_der_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities_->get_block(outJac_handle, faultNo);

            double VMax = lop_->lhsCompactDAE(faultNo, time, traction, state_block, state_der_block, result_block, scratch);

            lop_->getJacobianQuantitiesCompact(faultNo, time, traction, state_block, state_der_block, JacobianQuantities_block, scratch);

            VMax_ = std::max(VMax_, VMax);
            // std::cout << "u     at fault " << faultNo <<": ";
            // for (int i = 0; i < block_size(); ++i){
            //     std::cout << state_block(i) << " ";
            // }
            // std::cout<<std::endl;
            // std::cout << "du/dt at fault " << faultNo <<": ";
            // for (int i = 0; i < block_size(); ++i){
            //     std::cout << state_der_block(i) << " ";
            // }
            // std::cout<<std::endl;
            // std::cout << "F     at fault " << faultNo <<": ";
            // for (int i = 0; i < block_size(); ++i){
            //     std::cout << result_block(i) << " ";
            // }
            // std::cout<<std::endl;
            for (int i = 0; i < block_size(); ++i) {
                FMax = std::max(FMax, std::abs(result_block(i)));
            }
        }
        solverParameters_.FMax = FMax;
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
     */
    template <typename BlockVector> void lhsExtendedDAE(double time, BlockVector& state, BlockVector& state_der, BlockVector& result) {
        time += solverParameters_.time_eq;
        adapter_->solve(time, state);

        auto scratch = make_scratch();
        auto in_handle = state.begin_access_readonly();
        auto in_der_handle = state_der.begin_access_readonly();
        auto out_handle = result.begin_access();
        
        auto outJac_handle = JacobianQuantities_->begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;        
        double FMax = 0.0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto state_block = state.get_block(in_handle, faultNo);
            auto state_der_block = state_der.get_block(in_der_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities_->get_block(outJac_handle, faultNo);

            double VMax = lop_->lhsExtendedDAE(faultNo, time, traction, state_block, state_der_block, result_block, scratch);

            lop_->getJacobianQuantitiesExtended(faultNo, time, traction, state_block, JacobianQuantities_block, scratch);

            VMax_ = std::max(VMax_, VMax);
            for (int i = 0; i < block_size(); ++i) {     
                FMax = std::max(FMax, std::abs(result_block(i)));
            }
        }
        solverParameters_.FMax = FMax;                  
        adapter_->end_traction();
        state_der.end_access_readonly(in_der_handle);
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities_->end_access(outJac_handle);

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

    SeasAdapter const& adapter() const { return *adapter_; }

    size_t rhs_count() {return evaluation_rhs_count; };

    double VMax() const { return VMax_; }

    double fMax() const { return (error_extended_ODE_ >= 0) ? error_extended_ODE_ : solverParameters_.FMax; }
    
    LocalOperator& lop() {return *lop_; }

    /**
     * Set up the constant part dV/dt for the extended ODE formulation
     */
    void initialize_secondOrderDerivative() {
        // ----------------- general parameters ----------------- //
        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t nbf = lop_->space().numBasisFunctions();
        size_t Nbf = adapter_->block_size_rhsDG();
        size_t totalnbf = nbf * numFaultElements;
        size_t totalNbf = Nbf * numFaultElements;
        size_t totalSize = blockSize * numFaultElements;

        // ----------------- Initiallize vector ---------------//
        dV_dt_ = new PetscBlockVector(nbf, numFaultElements, comm());

        // ----------------- Calculate dV/dt ----------------- //
                                            
        // vector to initialize other vectors to 0
        PetscBlockVector zeroVector = PetscBlockVector(blockSize, numFaultElements, this->comm());
            zeroVector.set_zero();

        // evaluate tau for the zero vector at time t=1
        double time = 1.0; 
        adapter_->solve(time, zeroVector);

        auto scratch = make_scratch();
        auto in_handle = zeroVector.begin_access_readonly();
        auto out_handle = dV_dt_->begin_access();

        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&zeroVector, &in_handle](std::size_t faultNo) {
            return zeroVector.get_block(in_handle, faultNo);
        });
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);
            auto res = dV_dt_->get_block(out_handle, faultNo);
            for (int i = 0; i < nbf; ++i){
                res(i) = traction(i,1);
            }
        }
        std::cout<<std::endl;
        adapter_->end_traction();

        zeroVector.end_access_readonly(in_handle);
        dV_dt_->end_access(out_handle);
    }


    /**
     * Initialize the Jacobian matrices and set up the constant part df/dS 
     * @param bs_compact block size in the compact formulation
     * @param bs_extended block size in the extended formulation
     */
    void initialize_Jacobian(std::size_t bs_compact, std::size_t bs_extended){
        // ----------------- general compact parameters ----------------- //
        size_t numFaultElements = this->numLocalElements();
        size_t totalSize = bs_compact * numFaultElements;

        // ----------------- initialize Jacobian matices to 0 ----------------- //
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_compact_ode_));
        CHKERRTHROW(MatZeroEntries(Jacobian_compact_ode_));
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_compact_dae_));
        CHKERRTHROW(MatZeroEntries(Jacobian_compact_dae_));

        // ----------------- general extended parameters ---------------------- //
       totalSize = bs_extended * numFaultElements;

        // ----------------- initialize Jacobian matices to 0 ----------------- //
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_extended_ode_));
        CHKERRTHROW(MatZeroEntries(Jacobian_extended_ode_));
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_extended_dae_));
        CHKERRTHROW(MatZeroEntries(Jacobian_extended_dae_));

        // -------------------- initialize constant df/dS --------------------- //
        initialize_df_dS();
        
    }
    /**
     * calculate the analytic Jacobian matrix
     * @param Jac the Jacobian
     */
    void updateJacobianCompactODE(Mat& Jac){  
        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t totalSize = blockSize * numFaultElements;
        size_t nbf = lop_->space().numBasisFunctions();
        size_t numFaultNodes = nbf * numFaultElements;

        using namespace Eigen;

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

        const double* df_dS;
        double* F_x;
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseGetArrayWrite(Jac, &F_x));
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            for (int i = 0; i < nbf; i++){                  // row iteration
                S_i = noFault * blockSize + i;
                PSI_i = noFault * blockSize + 
                                       RateAndStateBase::TangentialComponents * nbf + i;
                n_i = noFault * nbf + i;

                F_x[S_i   + PSI_i * totalSize] = -df_dpsi_vec(n_i) / df_dV_vec(n_i);  // dV/dpsi
                F_x[PSI_i + PSI_i * totalSize] = dg_dpsi_vec(n_i) + dg_dV_vec(n_i) * F_x[S_i   + PSI_i * totalSize];
                
                for (int noFault2 = 0; noFault2 < numFaultElements; noFault2++){
                    for(int j = 0; j < nbf; j++) {          // column iteration

                        S_j = noFault2 * blockSize + j;
                        n_j = noFault2 * nbf + j;

                        // column major !                       
                        F_x[S_i   + S_j * totalSize  ] = -df_dS[n_i + n_j * numFaultNodes] / df_dV_vec(n_i);    // dV/dS
                        F_x[PSI_i + S_j * totalSize  ] = dg_dV_vec(n_i) * F_x[S_i   + S_j * totalSize  ];       // dg/dS = -b/L * dV/dS
                    }
                }
            }
        }

        CHKERRTHROW(MatDenseRestoreArrayWrite(Jac, &F_x));
        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));

    }

    /**
     * calculate the analytic Jacobian matrix
     * @param Jac the Jacobian
     */
    void updateJacobianExtendedODE(Mat& Jac){  

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

        const double* df_dS;
        double* J;
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseGetArrayWrite(Jac, &J));

        solverParameters_.ratio_addition = 0.0;
        double ratio;
        for (int i = 0; i < numFaultNodes; ++i) {
            ratio = sigma * df_dV_vec(i) / df_dS[i + i * numFaultNodes];
            std::cout << sigma << ": dg/dpsi: " << dg_dpsi_vec(i) << ", df/dS: "<< df_dS[i + i * numFaultNodes] << std::endl;
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
                                
                for (int noFault2 = 0; noFault2 < numFaultElements; noFault2++){
                    for(int j = 0; j < nbf; j++) {          // column iteration

                        S_j = noFault2 * blockSize + j;
                        n_j = noFault2 * nbf + j;

                        // components F_x
                        J[S_i   + S_j * totalSize  ] += df_dS[n_i + n_j * numFaultNodes];
                    }
                }
            }
        }

        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jac, &J));
    }

    /**
     * Updates the two Jacobians F_x and F_\dot{x}
     * @param sigma shift of due to the numerical scheme
     * @param Jac the Jacobian to be updated
     */
    void updateJacobianExtendedDAE(double sigma, Mat& Jac ){  

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
        int V_i;
        int S_j;
        int n_i;
        int n_j;

        CHKERRTHROW(MatZeroEntries(Jac));

        const double* df_dS;
        double* J;
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseGetArrayWrite(Jac, &J));
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            for (int i = 0; i < nbf; i++){                  // row iteration
                S_i = noFault * blockSize + i;
                PSI_i = noFault * blockSize + 
                                       RateAndStateBase::TangentialComponents     * nbf + i;
                V_i = noFault * blockSize + 
                                      (RateAndStateBase::TangentialComponents +1) * nbf + i;
                n_i = noFault * nbf + i;
                
                // components F_xdot                
                J[S_i   +   S_i * totalSize] = -sigma;
                J[PSI_i + PSI_i * totalSize] = -sigma;

                // components F_x                
                J[S_i   + V_i   * totalSize] = 1;
                J[PSI_i + PSI_i * totalSize] += dg_dpsi_vec(n_i);
                J[PSI_i + V_i   * totalSize] = dg_dV_vec(n_i);
                J[V_i   + PSI_i * totalSize] = df_dpsi_vec(n_i);
                J[V_i   + V_i   * totalSize] = df_dV_vec(n_i);
                
                for (int noFault2 = 0; noFault2 < numFaultElements; noFault2++){
                    for(int j = 0; j < nbf; j++) {          // column iteration

                        S_j = noFault2 * blockSize + j;
                        n_j = noFault2 * nbf + j;

                        // components F_x     
                        J[V_i   + S_j * totalSize  ] += df_dS[n_i + n_j * numFaultNodes];
                    }
                }
            }
        }
        // std::cout << "df/dV: "   << std::endl << df_dV_vec   << std::endl
        //           << "df/dpsi: " << std::endl << df_dpsi_vec << std::endl   
        //           << "dg/dV: "   << std::endl << dg_dV_vec   << std::endl   
        //           << "dg/dpsi: " << std::endl << dg_dpsi_vec << std::endl;

        // std::cout << "Jacobian for sigma="<<sigma<< " is: " << std::endl<< "[ ";
        // for (int i = 0; i < totalSize; ++i){
        //     for (int j = 0; j < totalSize; ++j){
        //         std::cout << J[i + j * totalSize] << " ";
        //     }            
        //     (i+1 == totalSize) ? std::cout << "]" : std::cout << "; "; 
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl<<std::endl; 

        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jac, &J));
    }

    Mat& getJacobianCompactODE() { return Jacobian_compact_ode_; }

    Mat& getJacobianExtendedODE() { return Jacobian_extended_ode_; }

    Mat& getJacobianCompactDAE() { return Jacobian_compact_dae_; }

    Mat& getJacobianExtendedDAE() { return Jacobian_extended_dae_; }

    solverParametersASandEQ& getSolverParameters(){ return solverParameters_; }

    auto& getGeneralSolverConfiguration() const { return solverGenCfg_; }

    auto& getEarthquakeSolverConfiguration() const { return solverEqCfg_; }

    auto& getAseismicSlipSolverConfiguration() const { return solverAsCfg_; }

    double getV0(){ return lop_->getV0(); }

    void setExtendedFormulation(bool is) { lop_->setExtendedFormulation(is); }

    void setTSInstance(TS ts) { ts_ = ts; }

    void resetDivergenceTest() { solverParameters_.FMax = -1.0; }

    /**
     * Allocate memory in scratch
     */
    auto make_scratch() const {
        return LinearAllocator<double>(scratch_mem_.get(), scratch_mem_.get() + scratch_size_);
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

        auto scratch = make_scratch();
        auto in_handle = oneVector.begin_access_readonly();
       
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&oneVector, &in_handle](std::size_t faultNo) {
            return oneVector.get_block(in_handle, faultNo);
        });
        double tau_min = 10;
        double sn_max = 0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);
 
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

    private:
    /**
     * Set up the constant part df/dS of the Jacobian matrix
     */
    void initialize_df_dS() {
        // ----------------- general parameters ----------------- //
        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t nbf = lop_->space().numBasisFunctions();
        size_t Nbf = adapter_->block_size_rhsDG();
        size_t totalnbf = nbf * numFaultElements;
        size_t totalNbf = Nbf * numFaultElements;
        size_t totalSize = blockSize * numFaultElements;

        // ----------------- set up the Jacobian du_dS ----------------- //
        Mat du_dS;
        CHKERRTHROW(MatCreateSeqDense(comm(), totalNbf, totalnbf, NULL, &du_dS));
        CHKERRTHROW(MatZeroEntries(du_dS));
                                            
        // vector to initialize other vectors to 0
        PetscBlockVector zeroVector = PetscBlockVector(blockSize, numFaultElements, this->comm());
        zeroVector.set_zero();

        Vec col_du_dS;
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            for (int i = 0; i < nbf; i++){
                // set up unit vector e
                PetscBlockVector unitVector(zeroVector);
                auto AccessHandle = unitVector.begin_access();
                auto block = unitVector.get_block(AccessHandle, noFault);
                block(i) = 1.0;
                unitVector.end_access(AccessHandle);        

                // solve system Au - e = 0
                PetscBlockVector solutionVector = PetscBlockVector(Nbf, numFaultElements, this->comm());
                solutionVector.set_zero();

                adapter_->solveUnitVector(unitVector, solutionVector);

                // copy u = A^{-1} * e to the columns of du/dS
                CHKERRTHROW(MatDenseGetColumnVecWrite(du_dS, noFault * nbf + i, &col_du_dS));  
                CHKERRTHROW(VecCopy(solutionVector.vec(), col_du_dS));
                CHKERRTHROW(MatDenseRestoreColumnVecWrite(du_dS, noFault * nbf + i, &col_du_dS));  
           }
        }

        // ----------------- calculate dtau/dU ----------------- //
        Mat dtau_du_big;
        CHKERRTHROW(MatCreateSeqDense(comm(), totalnbf, totalNbf, NULL, &dtau_du_big));
        CHKERRTHROW(MatZeroEntries(dtau_du_big));

        // initialize element tensor for the kernel
        auto scratch = this->make_scratch();
        TensorBase<Matrix<double>> tensorBase = adapter_->getBaseDtauDu();
        auto dtau_du = Managed<Matrix<double>>(tensorBase);
        assert(dtau_du.shape()[0] == nbf);
        assert(dtau_du.shape()[1] == Nbf);

        double* mat;
        CHKERRTHROW(MatDenseGetArrayWrite(dtau_du_big, &mat));
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            this->adapter().dtau_du(noFault, dtau_du, scratch);     // call kernel for one element

            for(int i = 0; i<nbf; i++){                             // write result matrix from kernel
                for(int j = 0; j<Nbf; j++){
                    // column major in mat!
                    mat[noFault * nbf + i + (noFault * Nbf + j) * totalnbf] = dtau_du(i, j);
                }
            }

        }
        CHKERRTHROW(MatDenseRestoreArrayWrite(dtau_du_big, &mat));


        // ----------------- calculate dtau/dS ----------------- //
        Mat dtau_dS_big;
        CHKERRTHROW(MatCreateSeqDense(comm(), totalnbf, totalnbf, NULL, &dtau_dS_big));
        CHKERRTHROW(MatZeroEntries(dtau_dS_big));

        // initialize element tensor for the kernel
        tensorBase = adapter_->getBaseDtauDS();
        auto dtau_dS = Managed<Matrix<double>>(tensorBase);
        assert(dtau_dS.shape()[0] == nbf);
        assert(dtau_dS.shape()[1] == nbf);

        CHKERRTHROW(MatDenseGetArrayWrite(dtau_dS_big, &mat));
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            this->adapter().dtau_dS(noFault, dtau_dS, scratch);     // call kernel for one element

            for(int i = 0; i < nbf; i++){                             // write result matrix from kernel
                for(int j = 0; j < nbf; j++){
                    // column major in mat!
                    mat[noFault * nbf + i + (noFault * nbf + j) * totalnbf] = dtau_dS(i, j);
                }
            }

        }
        CHKERRTHROW(MatDenseRestoreArrayWrite(dtau_dS_big, &mat));


        // ----------------- df/dS = dtau/dS = dtau/dU * dU/dS + pdv{tau}{S}----------------- //
        CHKERRTHROW(MatMatMult(dtau_du_big, du_dS, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &df_dS_));
        CHKERRTHROW(MatAXPY(df_dS_,1.0,dtau_dS_big,DIFFERENT_NONZERO_PATTERN));

        // get rid of the local matrices
        CHKERRTHROW(MatDestroy(&du_dS));
        CHKERRTHROW(MatDestroy(&dtau_du_big));
        CHKERRTHROW(MatDestroy(&dtau_dS_big));
    }

    /**
     * Calculate the rhs of the slip rate as dV/dt = J * d(S,psi)/dt
     * @param rhs current rhs of the system
     * @param time current time of the system
     */
    template <typename BlockVector>
    void calculateSlipRateExtendedODE(BlockVector& rhs, double time){
        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t totalSize = blockSize * numFaultElements;
        size_t nbf = lop_->space().numBasisFunctions();
        size_t numFaultNodes = nbf * numFaultElements;

        using namespace Eigen;

        // fill vectors     
        VectorXd df_dV_vec(numFaultNodes);                             
        VectorXd df_dpsi_vec(numFaultNodes);

        auto accessRead = JacobianQuantities_->begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = JacobianQuantities_->get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
            }
        }

        JacobianQuantities_->end_access_readonly(accessRead);


        // calculate the derivative of the slip rate
        int S_i;
        int PSI_i;
        int V_i;
        int S_j;

        int n_i;
        int n_j;

        auto out_handle = rhs.begin_access();
        auto dVdt_handle = dV_dt_->begin_access_readonly();

        const double* df_dS;
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto dxdt       = rhs.get_block(out_handle, noFault);
            auto const_dVdt = dV_dt_->get_block(dVdt_handle, noFault);

           for (int i = 0; i < nbf; i++){                  // row iteration
                S_i   = i;
                PSI_i =  RateAndStateBase::TangentialComponents      * nbf + i;
                V_i   = (RateAndStateBase::TangentialComponents + 1) * nbf + i;

                n_i = noFault * nbf + i;
 
                dxdt(V_i) = const_dVdt(i);
                dxdt(V_i) += df_dpsi_vec(n_i) * dxdt(PSI_i);  // = -dF/dpsi * dpsi/dt
                
                for (int noFault2 = 0; noFault2 < numFaultElements; noFault2++){ 
                    auto dSdt = rhs.get_block(out_handle, noFault2);
              
                    for(int j = 0; j < nbf; j++) {          // column iteration

                        n_j = noFault2 * nbf + j;
                        S_j = j;

                        dxdt(V_i) += df_dS[n_i + n_j * numFaultNodes] * dSdt(S_j);    // = -dF/dS * dS/dt
                    }
                }
                dxdt(V_i) /= -df_dV_vec(n_i); // = (dF/dV)^-1 * dF/dt
            }
        }

        rhs.end_access(out_handle);
        dV_dt_->end_access_readonly(dVdt_handle);
        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));

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
     void testJacobianMatrices(BlockVector& state, BlockVector& rhs, double time, bool full){

        using namespace Eigen;

        updateJacobianCompactODE();

        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t totalSize = blockSize * numFaultElements;
        size_t nbf = lop_->space().numBasisFunctions();
        size_t numFaultNodes = nbf * numFaultElements;

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

                double h = 1e-8 + 1e-10 * abs(x_block(j));
                x_l_block(j) -= h;
                x_r_block(j) += h;

                x_left.end_access(x_l);
                x_right.end_access(x_r);
                state.end_access_readonly(x);

                rhsCompactODE(time, x_left, f_left, false);
                rhsCompactODE(time, x_right, f_right, false);

                const auto f_r = f_right.begin_access_readonly();
                const auto f_l = f_left.begin_access_readonly();

                for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
                    auto f_l_block = f_left.get_block(f_l, faultNo_i);
                    auto f_r_block = f_right.get_block(f_r, faultNo_i);
                    for (int i = 0; i < blockSize; i++){                        
                        n_i = faultNo_i * blockSize + i;

                        J_approx = (f_r_block(i) - f_l_block(i)) / (2.0 * h);            
                        J = F_x[n_i + n_j * totalSize];
                        if (full) std::cout << (J_approx - J) / J_approx << " ";                            
                        if (!full) max_rel_diff = std::max(max_rel_diff, 
                                                  std::abs((J_approx - J) / J_approx));
                    }
                }

                if (full) ((j+1 == nbf) && (faultNo_j + 1 == numFaultElements)) ? std::cout << "]" : std::cout << "; "; 
                if (full) std::cout << std::endl;
                f_left.end_access_readonly(f_l);
                f_right.end_access_readonly(f_r);
            }            
        }
        if (full) std::cout << std::endl << std::endl << std::endl;
        if (!full) std::cout << "maximal relative difference between J and its approximate is " << max_rel_diff << std::endl;

        CHKERRTHROW(MatDenseRestoreArrayRead(Jacobian_compact_ode_, &F_x));


        // --------------APPROXIMATE DF/DS------------------- //

        double dfdS_approx;
        double dfdS;

        const double* df_dS;
        max_rel_diff = 0; 
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
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

                auto scratch = make_scratch();
                auto in_handle = x_left.begin_access_readonly();
                auto in_rhs_handle = rhs.begin_access();
                auto traction = Managed<Matrix<double>>(adapter_->traction_info());
                adapter_->begin_traction([&x_left, &in_handle](std::size_t faultNo) {
                    return x_left.get_block(in_handle, faultNo);
                });
                auto out_handle = f_left.begin_access();

                for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
                    adapter_->traction(faultNo, traction, scratch);

                    auto state_block = x_left.get_block(in_handle, faultNo);
                    auto state_der_block = rhs.get_block(in_rhs_handle, faultNo);
                    auto result_block = f_left.get_block(out_handle, faultNo);

                    lop_->applyFrictionLaw(faultNo, time, traction, state_block, state_der_block, result_block, scratch);
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
                    adapter_->traction(faultNo, traction, scratch);

                    auto state_block = x_right.get_block(in_handle, faultNo);
                    auto state_der_block = rhs.get_block(in_rhs_handle, faultNo);
                    auto result_block = f_right.get_block(out_handle, faultNo);

                    lop_->applyFrictionLaw(faultNo, time, traction, state_block, state_der_block, result_block, scratch);
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
                        dfdS = df_dS[n_i + n_j * numFaultNodes];
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
        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));
        if (full) std::cout << std::endl << std::endl << std::endl;
        if (!full) std::cout << "maximal relative difference between df/dS and its approximate is " << max_rel_diff << std::endl;



        // --------------APPROXIMATE DF/DV------------------- //

        double dfdV_approx;
        double dfdV;
        max_rel_diff = 0; 
        if (full) std::cout << "relative difference to the approximated df/dV is: "<<std::endl<<"[ ";        
        for (int faultNo_j = 0; faultNo_j < numFaultElements; ++faultNo_j){
            for (int j = 0; j < nbf; j++){
                n_j = faultNo_j * nbf + j;

                PetscBlockVector f_left(blockSize, numFaultElements, comm());
                PetscBlockVector f_right(blockSize, numFaultElements, comm());

                double h;

                // get the friction law
                auto scratch = make_scratch();
                auto in_handle = state.begin_access_readonly();
                auto in_rhs_handle = rhs.begin_access();
                auto traction = Managed<Matrix<double>>(adapter_->traction_info());
                adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
                    return state.get_block(in_handle, faultNo);
                });
                auto out_left_handle = f_left.begin_access();
                auto out_right_handle = f_right.begin_access();

                for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
                    adapter_->traction(faultNo, traction, scratch);

                    auto state_block = state.get_block(in_handle, faultNo);
                    auto state_der_block = rhs.get_block(in_rhs_handle, faultNo);
                    auto result_left_block = f_left.get_block(out_left_handle, faultNo);
                    auto result_right_block = f_right.get_block(out_right_handle, faultNo);

                    h = 1e-2 * abs(state_der_block(j));
                    if (faultNo == faultNo_j) state_der_block(j) -= h;
                    lop_->applyFrictionLaw(faultNo, time, traction, state_block, state_der_block, result_left_block, scratch);
                    if (faultNo == faultNo_j) state_der_block(j) += h;
                    if (faultNo == faultNo_j) state_der_block(j) += h;
                    lop_->applyFrictionLaw(faultNo, time, traction, state_block, state_der_block, result_right_block, scratch);
                    if (faultNo == faultNo_j) state_der_block(j) -= h;
                }
                adapter_->end_traction();
                state.end_access_readonly(in_handle);
                rhs.end_access(in_rhs_handle);
                f_left.end_access(out_left_handle);
                f_right.end_access(out_right_handle);


                const auto f_r = f_right.begin_access_readonly();
                const auto f_l = f_left.begin_access_readonly();

                for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
                    auto f_l_block = f_left.get_block(f_l, faultNo_i);
                    auto f_r_block = f_right.get_block(f_r, faultNo_i);

                    for (int i = 0; i < nbf; i++){     
                        n_i = faultNo_i * nbf + i;                                           
                        dfdV_approx = (f_r_block(i) - f_l_block(i)) / (2.0 * h);
                        dfdV = df_dV_vec(n_i) * delta(n_i, n_j);
                        if (full) std::cout << (dfdV_approx - dfdV) / dfdV_approx <<" ";                           
                        if (!full) max_rel_diff = std::max(max_rel_diff, 
                                                  std::abs((dfdV_approx - dfdV) / dfdV_approx)); 
                    }
                }
                if (full) ((j+1 == nbf) && (faultNo_j + 1 == numFaultElements)) ? std::cout << "]" : std::cout << "; "; 
                if (full) std::cout << std::endl;
                f_left.end_access_readonly(f_l);
                f_right.end_access_readonly(f_r);
            }            
        }
        if (full) std::cout << std::endl << std::endl << std::endl;
        if (!full) std::cout << "maximal relative difference between df/dV and its approximate is " << max_rel_diff << std::endl;

        // ------------------APPROXIMATE DV/Dt ---------------------------- //
        PetscBlockVector x_left(blockSize, numFaultElements, comm());
        PetscBlockVector x_right(blockSize, numFaultElements, comm());
        PetscBlockVector f_left(blockSize, numFaultElements, comm());
        PetscBlockVector f_right(blockSize, numFaultElements, comm());

        CHKERRTHROW(VecCopy(state.vec(), x_left.vec()));
        CHKERRTHROW(VecCopy(state.vec(), x_right.vec()));

        double h = 1e-5;

        CHKERRTHROW(VecAXPY(x_right.vec(), h, rhs.vec()));
        CHKERRTHROW(VecAXPY(x_left.vec(), -h, rhs.vec()));

        rhsCompactODE(time-h, x_left, f_left, false);
        rhsCompactODE(time+h, x_right, f_right, false);


        // calculate the derivative of the slip rate
        int S_i;
        int PSI_i;
        int S_j;

        auto out_handle = rhs.begin_access();

        auto left_handle = f_left.begin_access_readonly();
        auto right_handle = f_right.begin_access_readonly();

        double dVdt;
        double dVdt_approx;

        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
        max_rel_diff = 0; 
        if (full) std::cout << "relative difference to the approximated dV/dt is: " << std::endl;
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto dPSIdt = rhs.get_block(out_handle, noFault);

            auto lb = f_left.get_block(left_handle, noFault);
            auto rb = f_right.get_block(right_handle, noFault);

           for (int i = 0; i < nbf; i++){                  // row iteration
                S_i   = i;
                PSI_i =  RateAndStateBase::TangentialComponents * nbf + i;

                n_i = noFault * nbf + i;
 
                dVdt = -df_dpsi_vec(n_i) * dPSIdt(PSI_i);  // = -dF/dpsi * dpsi/dt
                
                for (int noFault2 = 0; noFault2 < numFaultElements; noFault2++){ 
                    auto dSdt = rhs.get_block(out_handle, noFault2);
              
                    for(int j = 0; j < nbf; j++) {          // column iteration

                        n_j = noFault2 * nbf + j;
                        S_j = j;

                        dVdt -= df_dS[n_i + n_j * numFaultNodes] * dSdt(S_j);    // = -dF/dS * dS/dt
                    }
                }
                dVdt /= df_dV_vec(n_i); // = (dF/dV)^-1 * dF/dt

                dVdt_approx = (rb(i) - lb(i)) / (2.0 * h);
                if (full) std::cout << (dVdt - dVdt_approx) / dVdt_approx << std::endl;
                max_rel_diff = std::max(max_rel_diff,std::abs((dVdt - dVdt_approx) / dVdt_approx));
            }
        }
        if (full) std::cout << std::endl << std::endl << std::endl;
        if (!full) std::cout << "maximal relative difference between dV/dt and its approximate is " << max_rel_diff << std::endl;

        rhs.end_access_readonly(out_handle);
        f_left.end_access_readonly(left_handle);
        f_right.end_access_readonly(right_handle);
        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));

    }


    std::unique_ptr<LocalOperator> lop_;    // on fault: rate and state instance (handles ageing law and slip_rate)
    std::unique_ptr<SeasAdapter> adapter_;  // on domain: DG solver (handles traction and mechanical solver)
    std::unique_ptr<double[]> scratch_mem_; // memory allocated, not sure for what
    std::size_t scratch_size_;              // size of this memory
    double VMax_ = 0.0;                     // metrics: maximal velocity among all fault elements
    double error_extended_ODE_ = -1.0;      // metrics: evaluation of the maximum absolute value of the friction law 
    size_t evaluation_rhs_count = 0;        // metrics: counts the number of calls of the rhs function in one time step

    PetscBlockVector* JacobianQuantities_;  // contains the partial derivatives [df/dV, df/dpsi, dg/dV, dg/dpsi]
    PetscBlockVector* dV_dt_;               // partial derivative dV/dt (constant, needed to construct  the real Jacobian)     
    Mat df_dS_;                             // Jacobian df/dS (constant, needed to construct  the real Jacobian)     
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

    TS ts_;                                 // TS instance

};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
