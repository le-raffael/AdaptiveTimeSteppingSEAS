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

#include <Eigen/Dense>

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

    struct solverParametersASandEQ {
        double S_rtol; 
        double S_atol; 

        double psi_rtol; 
        double psi_atol_eq; 
        double psi_atol_as;

        std::string type_eq;
        std::string type_as;

        std::string rk_type_eq;
        std::string rk_type_as;

        int bdf_order_eq;
        int bdf_order_as;

        std::string ksp_type;
        std::string pc_type;
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

#include <Eigen/Dense>

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

    struct solverParametersASandEQ {
        double S_rtol; 
        double S_atol; 

        double psi_rtol; 
        double psi_atol_eq; 
        double psi_atol_as;

        std::string type_eq;
        std::string type_as;

        std::string rk_type_eq;
        std::string rk_type_as;

        int bdf_order_eq;
        int bdf_order_as;

        std::string ksp_type;
        std::string pc_type;

        bool bdf_custom_error_evaluation;
        bool bdf_custom_Newton_iteration;
        int (*customErrorFct)(TS,NormType,PetscInt*,PetscReal*);
        int (*customNewtonFct)(SNES,Vec);

        bool checkAS;           // true if it is a period of aseismic slip, false if it is an eartquake    
        bool JacobianNeeded;    // true if an implicit method is used either in the aseismic slip or in the earthquake phase
    }; 


    /**
     * COnstructor function, prepare all operators
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
     * Destructor
     */
    ~SeasOperator() {
        MatDestroy(&Jacobian_ode_);
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

        // Initialize the Jacobian matrix
        this->initializeJacobian();
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
    template <typename BlockVector> void rhsODE(double time, BlockVector& state, BlockVector& result, bool PetscCall) {
        adapter_->solve(time, state);

        // auto x = state.begin_access_readonly();
        // for (int i = 0; i < 30; ++i){
        //     std::cout << x[i] << ", ";
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;
        // state.end_access_readonly(x);

        auto scratch = make_scratch();
        auto in_handle = state.begin_access_readonly();
        auto out_handle = result.begin_access();

        PetscBlockVector JacobianQuantities(4 * lop_->space().numBasisFunctions(), numLocalElements(), comm());

        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        auto outJac_handle = JacobianQuantities.begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto state_block = state.get_block(in_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities.get_block(outJac_handle, faultNo);

            double VMax = lop_->rhsODE(faultNo, time, traction, state_block, result_block, scratch);

            lop_->getJacobianQuantities(faultNo, time, traction, state_block, result_block, JacobianQuantities_block, scratch);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities.end_access(outJac_handle);

        //update the Jacobian
        if (PetscCall) updateJacobianODE(JacobianQuantities, state, result, time);

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
        adapter_->solve(time, state);

        auto scratch = make_scratch();
        auto in_handle = state.begin_access_readonly();
        auto in_der_handle = state_der.begin_access();
        auto out_handle = result.begin_access();

        PetscBlockVector JacobianQuantities(4 * lop_->space().numBasisFunctions(), numLocalElements(), comm());
        
        auto outJac_handle = JacobianQuantities.begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto state_block = state.get_block(in_handle, faultNo);
            auto state_der_block = state_der.get_block(in_der_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities.get_block(outJac_handle, faultNo);

            double VMax = lop_->lhsCompactDAE(faultNo, time, traction, state_block, state_der_block, result_block, scratch);

            lop_->getJacobianQuantities(faultNo, time, traction, state_block, state_der_block, JacobianQuantities_block, scratch);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state_der.end_access(in_der_handle);
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities.end_access(outJac_handle);

        //update the Jacobian
        updateJacobianCompactDAE(JacobianQuantities);

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
    
    LocalOperator& lop() {return *lop_; }

    Mat& getJacobianODE() { return Jacobian_ode_; }

    Mat& getJacobianCompactDAE(double sigma ) {
        CHKERRTHROW(MatAXPY(Jacobian_compact_dae_x_, sigma, Jacobian_compact_dae_xdot_, DIFFERENT_NONZERO_PATTERN));
        return Jacobian_compact_dae_x_;
    }

    solverParametersASandEQ& getSolverParameters(){ return solverParameters_; }

    double getV0(){ return lop_->getV0(); }

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
    void initializeJacobian(){
        // ----------------- general parameters ----------------- //
        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t nbf = lop_->space().numBasisFunctions();
        size_t Nbf = adapter_->block_size_rhsDG();
        size_t totalnbf = nbf * numFaultElements;
        size_t totalNbf = Nbf * numFaultElements;
        size_t totalSize = blockSize * numFaultElements;

        // ----------------- initialize Jacobian matix to 1 for S and 0 for psi ----------------- //
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_ode_));
        CHKERRTHROW(MatZeroEntries(Jacobian_ode_));
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_compact_dae_x_));
        CHKERRTHROW(MatZeroEntries(Jacobian_compact_dae_x_));
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_compact_dae_xdot_));
        CHKERRTHROW(MatZeroEntries(Jacobian_compact_dae_xdot_));



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
                block(i) = 1;
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

        // ----------------- df/dS = dtau/dS = dtau/dU * dU/dS ----------------- //
        CHKERRTHROW(MatMatMult(dtau_du_big, du_dS, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &df_dS_));

        // get rid of the local matrices
        CHKERRTHROW(MatDestroy(&du_dS));
        CHKERRTHROW(MatDestroy(&dtau_du_big));
    }


    /**
     * approximate the Jacobian by second order finite differences
     * @param state current state of the system
     * @param rhs current rhs of the system
     * @param time current time of the system
     */
    template <typename BlockVector>
    void updateJacobianODE(PetscBlockVector& derivatives, BlockVector& state, BlockVector& rhs, double time){  
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
        VectorXd f_vec(numFaultNodes);

        auto accessRead = derivatives.begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = derivatives.get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
                dg_dV_vec(   noFault * nbf + i ) = derBlock(2 * nbf + i);
                dg_dpsi_vec( noFault * nbf + i ) = derBlock(3 * nbf + i);
            }
        }

        derivatives.end_access_readonly(accessRead);


        // fill the Jacobian
        int S_i;
        int PSI_i;
        int S_j;
        int n_i;
        int n_j;

        const double* df_dS;
        double* F_x;
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseGetArrayWrite(Jacobian_ode_, &F_x));
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

        Mat J_approx;
        CHKERRTHROW(MatDuplicate(Jacobian_ode_, MAT_DO_NOT_COPY_VALUES, &J_approx));

        double* J;
        CHKERRTHROW(MatDenseGetArrayWrite(J_approx, &J));

        // --------------APPROXIMATE J ------------------- //

//        std::cout << "difference between the two matrices is: "<<std::endl<<"[ "; 
        for (int faultNo_j = 0; faultNo_j < numFaultElements; ++faultNo_j){
            for (int j = 0; j < blockSize; j++){
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

                rhsODE(time, x_left, f_left, false);
                rhsODE(time, x_right, f_right, false);

                const auto f_r = f_right.begin_access_readonly();
                const auto f_l = f_left.begin_access_readonly();

                for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
                    auto f_l_block = f_left.get_block(f_l, faultNo_i);
                    auto f_r_block = f_right.get_block(f_r, faultNo_i);

                    for (int i = 0; i < blockSize; i++){    
                        n_i = faultNo_i * blockSize + i;
                                            
                        J[n_i + n_j * totalSize] = 
                            (f_r_block(i) - f_l_block(i)) / (2.0 * h);                        
                        // std::cout << (J[n_i + n_j * totalSize]
                        //             - F_x[n_i + n_j * totalSize])
                        //             / J[n_i + n_j * totalSize]<< ", ";                            
                    }
                }
                std::cout << std::endl;
                f_left.end_access_readonly(f_l);
                f_right.end_access_readonly(f_r);
            }            
        }
        std::cout << std::endl;
        CHKERRTHROW(MatDenseRestoreArrayWrite(J_approx, &J));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jacobian_ode_, &F_x));

        // --------------APPROXIMATE DF/DS------------------- //

        Mat df_dS_approx;
        double* dfdSapprox;
        CHKERRTHROW(MatDuplicate(df_dS_, MAT_DO_NOT_COPY_VALUES, &df_dS_approx));
        CHKERRTHROW(MatDenseGetArrayWrite(df_dS_approx, &dfdSapprox));
        std::cout << "relative difference to the approximated df/dS is: "<<std::endl<<"[ ";        
        for (int faultNo_j = 0; faultNo_j < numFaultElements; ++faultNo_j){
            for (int j = 0; j < nbf; j++){
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
                        dfdSapprox[n_i + n_j * numFaultElements*nbf] = 
                            (f_r_block(i) - f_l_block(i)) / (2.0 * h);                        
                        std::cout << (dfdSapprox[n_i + n_j * numFaultElements*nbf]
                                     -df_dS[n_i + n_j * numFaultElements*nbf])
                                      / dfdSapprox[n_i + n_j * numFaultElements*nbf]
                                    << ", ";                            
                    }
                }
                std::cout << std::endl;
                f_left.end_access_readonly(f_l);
                f_right.end_access_readonly(f_r);
            }            
        }

        std::cout << "analytic df/dS is: "<<std::endl<<"[ ";        
        for (int faultNo_j = 0; faultNo_j < numFaultElements; ++faultNo_j){
            for (int j = 0; j < nbf; j++){
                for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
                    for (int i = 0; i < nbf; i++){      
                        n_i = faultNo_i * nbf + i;                  
                        std::cout << df_dS[n_i + n_j * numFaultElements*nbf]
                                    << ", ";                            
                    }
                }
                std::cout << std::endl;
            }            
        }

        // --------------APPROXIMATE DF/DV------------------- //

        Mat df_dV_approx;
        double* dfdVapprox;
        CHKERRTHROW(MatDuplicate(df_dS_, MAT_DO_NOT_COPY_VALUES, &df_dV_approx));
        CHKERRTHROW(MatDenseGetArrayWrite(df_dV_approx, &dfdVapprox));
        std::cout << "relative difference to the approximated df/dV is: "<<std::endl<<"[ ";        
        for (int faultNo_j = 0; faultNo_j < numFaultElements; ++faultNo_j){
            for (int j = 0; j < nbf; j++){
                PetscBlockVector f_left(blockSize, numFaultElements, comm());
                PetscBlockVector f_right(blockSize, numFaultElements, comm());

                double h = 1e-5 * abs(x_block(j));

                // get the friction law
                in_handle = state.begin_access_readonly();
                in_rhs_handle = rhs.begin_access();
                traction = Managed<Matrix<double>>(adapter_->traction_info());
                adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
                    return x_right.get_block(in_handle, faultNo);
                });
                auto out_left_handle = f_left.begin_access();
                auto out_right_handle = f_right.begin_access();

                for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
                    adapter_->traction(faultNo, traction, scratch);

                    auto state_block = x_right.get_block(in_handle, faultNo);
                    auto state_der_block = rhs.get_block(in_rhs_handle, faultNo);
                    auto result_left_block = f_left.get_block(out_left_handle, faultNo);
                    auto result_right_block = f_right.get_block(out_right_handle, faultNo);

                    result_left_block(j) -= h;
                    result_right_block(j) += h;

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
                        dfdSapprox[n_i + n_j * numFaultElements*nbf] = 
                            (f_r_block(i) - f_l_block(i)) / (2.0 * h);                        
                        std::cout << (dfdSapprox[n_i + n_j * numFaultElements*nbf]
                                     -df_dS[n_i + n_j * numFaultElements*nbf])
                                      / dfdSapprox[n_i + n_j * numFaultElements*nbf]
                                    << ", ";                            
                    }
                }
                std::cout << std::endl;
                f_left.end_access_readonly(f_l);
                f_right.end_access_readonly(f_r);
            }            
        }


        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));
    }

    /**
     * Updates the two Jacobians F_x and F_\dot{x}
     * @param derivatives contains df/dV, df/dpsi and dg/dpsi in a vector
     */
    template <typename BlockVector>
    void updateJacobianCompactDAE(BlockVector& derivatives){  

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

        auto accessRead = derivatives.begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = derivatives.get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
                dg_dV_vec(   noFault * nbf + i ) = derBlock(2 * nbf + i);
                dg_dpsi_vec( noFault * nbf + i ) = derBlock(3 * nbf + i);
            }
        }

        derivatives.end_access_readonly(accessRead);

        // fill the Jacobian
        int S_i;
        int PSI_i;
        int S_j;
        int n_i;
        int n_j;

        const double* df_dS;
        double* F_x;
        double* F_xdot;
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseGetArrayWrite(Jacobian_compact_dae_x_, &F_x));
        CHKERRTHROW(MatDenseGetArrayWrite(Jacobian_compact_dae_xdot_, &F_xdot));
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            for (int i = 0; i < nbf; i++){                  // row iteration
                S_i = noFault * blockSize + i;
                PSI_i = noFault * blockSize + 
                                       RateAndStateBase::TangentialComponents * nbf + i;
                n_i = noFault * nbf + i;
                
                F_xdot[S_i   +   S_i * totalSize] = df_dV_vec(n_i);
                F_xdot[PSI_i +   S_i * totalSize] = -dg_dV_vec(n_i);
                F_xdot[PSI_i + PSI_i * totalSize] = 1;

                F_x[S_i   + PSI_i * totalSize] = df_dpsi_vec(n_i);
                F_x[PSI_i + PSI_i * totalSize] = dg_dpsi_vec(n_i);
                

                for (int noFault2 = 0; noFault2 < numFaultElements; noFault2++){
                    for(int j = 0; j < nbf; j++) {          // column iteration

                        S_j = noFault2 * blockSize + j;
                        n_j = noFault2 * nbf + j;

                        // column major !                       
                        F_x[S_i   + S_j * totalSize  ] = df_dS[n_i + n_j * numFaultNodes];
                    }
                }
            }
        }
        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jacobian_compact_dae_x_, &F_x));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jacobian_compact_dae_xdot_, &F_xdot));
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



    std::unique_ptr<LocalOperator> lop_;    // on fault: rate and state instance (handles ageing law and slip_rate)
    std::unique_ptr<SeasAdapter> adapter_;  // on domain: DG solver (handles traction and mechanical solver)
    std::unique_ptr<double[]> scratch_mem_; // memory allocated, not sure for what
    std::size_t scratch_size_;              // size of this memory
    double VMax_ = 0.0;                     // metrics: maximal velocity among all fault elements
    size_t evaluation_rhs_count = 0;        // metrics: counts the number of calls of the rhs function in one time step
    Mat df_dS_;                             // Jacobian df/dS (constant, needed to construct  the real Jacobian)     
    Mat Jacobian_ode_;                      // Jacobian matrix     
    Mat Jacobian_compact_dae_x_;            // Jacobian matrix F_x     
    Mat Jacobian_compact_dae_xdot_;         // Jacobian matrix F_\dot{x}    
    Mat Jacobian_approx_;                   // approximated Jacobian matrix     
    solverParametersASandEQ solverParameters_;
    Vec x_prev_;                            // solution at the previous timestep
    Vec rhs_prev_;                          // rhs evaluation at the previous timestep
};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H

        bool bdf_custom_error_evaluation;
        bool bdf_custom_Newton_iteration;
        int (*customErrorFct)(TS,NormType,PetscInt*,PetscReal*);
        int (*customNewtonFct)(SNES,Vec);

        bool checkAS;           // true if it is a period of aseismic slip, false if it is an eartquake    
        bool JacobianNeeded;    // true if an implicit method is used either in the aseismic slip or in the earthquake phase
    }; 


    /**
     * COnstructor function, prepare all operators
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
     * Destructor
     */
    ~SeasOperator() {
        MatDestroy(&Jacobian_ode_);
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

        // Initialize the Jacobian matrix
        this->initializeJacobian();
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
    template <typename BlockVector> void rhsODE(double time, BlockVector& state, BlockVector& result, bool PetscCall) {
        adapter_->solve(time, state);

        // auto x = state.begin_access_readonly();
        // for (int i = 0; i < 30; ++i){
        //     std::cout << x[i] << ", ";
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;
        // state.end_access_readonly(x);

        auto scratch = make_scratch();
        auto in_handle = state.begin_access_readonly();
        auto out_handle = result.begin_access();

        PetscBlockVector JacobianQuantities(4 * lop_->space().numBasisFunctions(), numLocalElements(), comm());

        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        auto outJac_handle = JacobianQuantities.begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto state_block = state.get_block(in_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities.get_block(outJac_handle, faultNo);

            double VMax = lop_->rhsODE(faultNo, time, traction, state_block, result_block, scratch);

            lop_->getJacobianQuantities(faultNo, time, traction, state_block, result_block, JacobianQuantities_block, scratch);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities.end_access(outJac_handle);

        //update the Jacobian
        if (PetscCall) updateJacobianODE(JacobianQuantities, state, result, time);

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
        adapter_->solve(time, state);

        auto scratch = make_scratch();
        auto in_handle = state.begin_access_readonly();
        auto in_der_handle = state_der.begin_access();
        auto out_handle = result.begin_access();

        PetscBlockVector JacobianQuantities(4 * lop_->space().numBasisFunctions(), numLocalElements(), comm());
        
        auto outJac_handle = JacobianQuantities.begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto state_block = state.get_block(in_handle, faultNo);
            auto state_der_block = state_der.get_block(in_der_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities.get_block(outJac_handle, faultNo);

            double VMax = lop_->lhsCompactDAE(faultNo, time, traction, state_block, state_der_block, result_block, scratch);

            lop_->getJacobianQuantities(faultNo, time, traction, state_block, state_der_block, JacobianQuantities_block, scratch);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state_der.end_access(in_der_handle);
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities.end_access(outJac_handle);

        //update the Jacobian
        updateJacobianCompactDAE(JacobianQuantities);

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
    
    LocalOperator& lop() {return *lop_; }

    Mat& getJacobianODE() { return Jacobian_ode_; }

    Mat& getJacobianCompactDAE(double sigma ) {
        CHKERRTHROW(MatAXPY(Jacobian_compact_dae_x_, sigma, Jacobian_compact_dae_xdot_, DIFFERENT_NONZERO_PATTERN));
        return Jacobian_compact_dae_x_;
    }

    solverParametersASandEQ& getSolverParameters(){ return solverParameters_; }

    double getV0(){ return lop_->getV0(); }

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
    void initializeJacobian(){
        // ----------------- general parameters ----------------- //
        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t nbf = lop_->space().numBasisFunctions();
        size_t Nbf = adapter_->block_size_rhsDG();
        size_t totalnbf = nbf * numFaultElements;
        size_t totalNbf = Nbf * numFaultElements;
        size_t totalSize = blockSize * numFaultElements;

        // ----------------- initialize Jacobian matix to 1 for S and 0 for psi ----------------- //
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_ode_));
        CHKERRTHROW(MatZeroEntries(Jacobian_ode_));
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_compact_dae_x_));
        CHKERRTHROW(MatZeroEntries(Jacobian_compact_dae_x_));
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_compact_dae_xdot_));
        CHKERRTHROW(MatZeroEntries(Jacobian_compact_dae_xdot_));



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
                block(i) = 1;
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

        // ----------------- df/dS = dtau/dS = dtau/dU * dU/dS ----------------- //
        CHKERRTHROW(MatMatMult(dtau_du_big, du_dS, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &df_dS_));

        // get rid of the local matrices
        CHKERRTHROW(MatDestroy(&du_dS));
        CHKERRTHROW(MatDestroy(&dtau_du_big));
    }


    /**
     * approximate the Jacobian by second order finite differences
     * @param derivatives block vector of some partial derivatives in the system [df/dV,df/dpsi,dg/dV,dg/dpsi,f(S,psi,V)]
     * @param state current state of the system
     * @param rhs current rhs of the system
     * @param time current time of the system
     */
    template <typename BlockVector>
    void updateJacobianODE(PetscBlockVector& derivatives, BlockVector& state, BlockVector& rhs, double time){  
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
        VectorXd f_vec(numFaultNodes);

        auto accessRead = derivatives.begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = derivatives.get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
                dg_dV_vec(   noFault * nbf + i ) = derBlock(2 * nbf + i);
                dg_dpsi_vec( noFault * nbf + i ) = derBlock(3 * nbf + i);
            }
        }

        derivatives.end_access_readonly(accessRead);


        // fill the Jacobian
        int S_i;
        int PSI_i;
        int S_j;
        int n_i;
        int n_j;

        const double* df_dS;
        double* F_x;
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseGetArrayWrite(Jacobian_ode_, &F_x));
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

        CHKERRTHROW(MatDenseRestoreArrayWrite(Jacobian_ode_, &F_x));
        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));

        testJacobianMatrices(derivatives, state, rhs, time);

    }

    /**
     * Updates the two Jacobians F_x and F_\dot{x}
     * @param derivatives contains df/dV, df/dpsi and dg/dpsi in a vector
     */
    template <typename BlockVector>
    void updateJacobianCompactDAE(BlockVector& derivatives){  

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
        VectorXd f_vec(numFaultNodes);

        auto accessRead = derivatives.begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = derivatives.get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
                dg_dV_vec(   noFault * nbf + i ) = derBlock(2 * nbf + i);
                dg_dpsi_vec( noFault * nbf + i ) = derBlock(3 * nbf + i);
            }
        }

        derivatives.end_access_readonly(accessRead);

        // fill the Jacobian
        int S_i;
        int PSI_i;
        int S_j;
        int n_i;
        int n_j;

        const double* df_dS;
        double* F_x;
        double* F_xdot;
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseGetArrayWrite(Jacobian_compact_dae_x_, &F_x));
        CHKERRTHROW(MatDenseGetArrayWrite(Jacobian_compact_dae_xdot_, &F_xdot));
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            for (int i = 0; i < nbf; i++){                  // row iteration
                S_i = noFault * blockSize + i;
                PSI_i = noFault * blockSize + 
                                       RateAndStateBase::TangentialComponents * nbf + i;
                n_i = noFault * nbf + i;
                
                F_xdot[S_i   +   S_i * totalSize] = df_dV_vec(n_i);
                F_xdot[PSI_i +   S_i * totalSize] = -dg_dV_vec(n_i);
                F_xdot[PSI_i + PSI_i * totalSize] = 1;

                F_x[S_i   + PSI_i * totalSize] = df_dpsi_vec(n_i);
                F_x[PSI_i + PSI_i * totalSize] = dg_dpsi_vec(n_i);
                

                for (int noFault2 = 0; noFault2 < numFaultElements; noFault2++){
                    for(int j = 0; j < nbf; j++) {          // column iteration

                        S_j = noFault2 * blockSize + j;
                        n_j = noFault2 * nbf + j;

                        // column major !                       
                        F_x[S_i   + S_j * totalSize  ] = df_dS[n_i + n_j * numFaultNodes];
                    }
                }
            }
        }
        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jacobian_compact_dae_x_, &F_x));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jacobian_compact_dae_xdot_, &F_xdot));
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
     * @param derivatives block vector of some partial derivatives in the system [df/dV,df/dpsi,dg/dV,dg/dpsi,f(S,psi,V)]
     * @param state current state of the system
     * @param rhs current rhs of the system
     * @param time current time of the system
     */
     template <typename BlockVector>
     void testJacobianMatrices(PetscBlockVector& derivatives, BlockVector& state, BlockVector& rhs, double time){
        
        using namespace Eigen;

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
        VectorXd f_vec(numFaultNodes);

        auto accessRead = derivatives.begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = derivatives.get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
                dg_dV_vec(   noFault * nbf + i ) = derBlock(2 * nbf + i);
                dg_dpsi_vec( noFault * nbf + i ) = derBlock(3 * nbf + i);
            }
        }

        derivatives.end_access_readonly(accessRead);

        const double* F_x;
        CHKERRTHROW(MatDenseGetArrayRead(Jacobian_ode_, &F_x));

        Mat J_approx;
        CHKERRTHROW(MatDuplicate(Jacobian_ode_, MAT_DO_NOT_COPY_VALUES, &J_approx));

        double* J;
        CHKERRTHROW(MatDenseGetArrayWrite(J_approx, &J));

        // --------------APPROXIMATE J ------------------- //
        std::cout << "difference between the two matrices is: "<<std::endl<<"[ "; 
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

                rhsODE(time, x_left, f_left, false);
                rhsODE(time, x_right, f_right, false);

                const auto f_r = f_right.begin_access_readonly();
                const auto f_l = f_left.begin_access_readonly();

                for (int faultNo_i = 0; faultNo_i < numFaultElements; ++faultNo_i){
                    auto f_l_block = f_left.get_block(f_l, faultNo_i);
                    auto f_r_block = f_right.get_block(f_r, faultNo_i);
                    for (int i = 0; i < blockSize; i++){                        
                        n_i = faultNo_i * blockSize + i;

                        J[n_i + n_j * totalSize] = 
                            (f_r_block(i) - f_l_block(i)) / (2.0 * h);            
                        std::cout << (J[n_i + n_j * totalSize]
                                    - F_x[n_i + n_j * totalSize])
                                    / J[n_i + n_j * totalSize]<< " ";                            
                    }
                }

                ((j+1 == nbf) && (faultNo_j + 1 == numFaultElements)) ? std::cout << "]" : std::cout << "; "; 
                std::cout << std::endl;
                f_left.end_access_readonly(f_l);
                f_right.end_access_readonly(f_r);
            }            
        }
        std::cout << std::endl << std::endl << std::endl;
        CHKERRTHROW(MatDenseRestoreArrayWrite(J_approx, &J));
        CHKERRTHROW(MatDenseRestoreArrayRead(Jacobian_ode_, &F_x));


        // --------------APPROXIMATE DF/DS------------------- //

        Mat df_dS_approx;
        double* dfdSapprox;
        CHKERRTHROW(MatDuplicate(df_dS_, MAT_DO_NOT_COPY_VALUES, &df_dS_approx));
        CHKERRTHROW(MatDenseGetArrayWrite(df_dS_approx, &dfdSapprox));

        const double* df_dS;
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
       std::cout << "relative difference to the approximated df/dS is: "<<std::endl<<"[ ";        
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

                double h = 1e-8 + 1e-10 * abs(x_block(j));
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
                        dfdSapprox[n_i + n_j * numFaultNodes] = 
                            (f_r_block(i) - f_l_block(i)) / (2.0 * h);                        
                        std::cout << (dfdSapprox[n_i + n_j * numFaultNodes]
                                     -df_dS[n_i + n_j * numFaultNodes])
                                      / dfdSapprox[n_i + n_j * numFaultNodes]
                                    << " ";                            
                    }
                }
                ((j+1 == nbf) && (faultNo_j + 1 == numFaultElements)) ? std::cout << "]" : std::cout << "; "; 
                std::cout << std::endl;
                f_left.end_access_readonly(f_l);
                f_right.end_access_readonly(f_r);
            }            
        }
        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));
        std::cout << std::endl << std::endl << std::endl;



        // --------------APPROXIMATE DF/DV------------------- //

        Mat df_dV_approx;
        double* dfdVapprox;
        CHKERRTHROW(MatDuplicate(df_dS_, MAT_DO_NOT_COPY_VALUES, &df_dV_approx));
        CHKERRTHROW(MatDenseGetArrayWrite(df_dV_approx, &dfdVapprox));
        std::cout << "relative difference to the approximated df/dV is: "<<std::endl<<"[ ";        
        for (int faultNo_j = 0; faultNo_j < numFaultElements; ++faultNo_j){
            for (int j = 0; j < nbf; j++){
                n_j = faultNo_j * nbf + j;

                PetscBlockVector f_left(blockSize, numFaultElements, comm());
                PetscBlockVector f_right(blockSize, numFaultElements, comm());

                double h = 1e-10;

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

                    if (faultNo == faultNo_j) state_der_block(j) -= h;
                    lop_->applyFrictionLaw(faultNo, time, traction, state_block, state_der_block, result_left_block, scratch);
                    if (faultNo == faultNo_j) result_right_block(j) += h;
                    if (faultNo == faultNo_j) state_der_block(j) += h;
                    lop_->applyFrictionLaw(faultNo, time, traction, state_block, state_der_block, result_right_block, scratch);
                    if (faultNo == faultNo_j) result_right_block(j) -= h;
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
                        dfdVapprox[n_i + n_j * numFaultNodes] = 
                            (f_r_block(i) - f_l_block(i)) / (h);
                       std::cout << (dfdVapprox[n_i + n_j * numFaultNodes]
                                    - df_dV_vec(n_i) * delta(n_i, n_j))
                                     / dfdVapprox[n_i + n_j * numFaultNodes]
                                    <<" ";                            
                    }
                }
                ((j+1 == nbf) && (faultNo_j + 1 == numFaultElements)) ? std::cout << "]" : std::cout << "; "; 
                std::cout << std::endl;
                f_left.end_access_readonly(f_l);
                f_right.end_access_readonly(f_r);
            }            
        }
        std::cout << std::endl << std::endl << std::endl;
        std::cout << std::endl << std::endl << std::endl;
    }


    std::unique_ptr<LocalOperator> lop_;    // on fault: rate and state instance (handles ageing law and slip_rate)
    std::unique_ptr<SeasAdapter> adapter_;  // on domain: DG solver (handles traction and mechanical solver)
    std::unique_ptr<double[]> scratch_mem_; // memory allocated, not sure for what
    std::size_t scratch_size_;              // size of this memory
    double VMax_ = 0.0;                     // metrics: maximal velocity among all fault elements
    size_t evaluation_rhs_count = 0;        // metrics: counts the number of calls of the rhs function in one time step
    Mat df_dS_;                             // Jacobian df/dS (constant, needed to construct  the real Jacobian)     
    Mat Jacobian_ode_;                      // Jacobian matrix     
    Mat Jacobian_compact_dae_x_;            // Jacobian matrix F_x     
    Mat Jacobian_compact_dae_xdot_;         // Jacobian matrix F_\dot{x}    
    Mat Jacobian_approx_;                   // approximated Jacobian matrix     
    solverParametersASandEQ solverParameters_;
    Vec x_prev_;                            // solution at the previous timestep
    Vec rhs_prev_;                          // rhs evaluation at the previous timestep
};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
