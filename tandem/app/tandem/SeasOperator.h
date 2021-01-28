#ifndef SEASOPERATOR_20201001_H
#define SEASOPERATOR_20201001_H

#include "form/BoundaryMap.h"
#include "geometry/Curvilinear.h"

#include "tandem/RateAndStateBase.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

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

    struct tolerancesSEAS {
        double S_rtol; 
        double S_atol; 
        double psi_rtol; 
        double psi_atol_eq; 
        double psi_atol_as;
        bool checkAS;           // true if it is a period of aseismic slip, false if it is an eartquake    
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
        MatDestroy(&Jacobian_);
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

        // Initialize previous x and rhs
        CHKERRTHROW(VecDuplicate(vector.vec(), &x_prev_));
        CHKERRTHROW(VecDuplicate(vector.vec(), &rhs_prev_));
        CHKERRTHROW(VecCopy(vector.vec(), x_prev_));


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
     */
    template <typename BlockVector> void rhs(double time, BlockVector& state, BlockVector& result) {
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

        PetscBlockVector JacobianQuantities(5 * lop_->space().numBasisFunctions(), numLocalElements(), comm());
        
        auto outJac_handle = JacobianQuantities.begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch);

            auto state_block = state.get_block(in_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            auto JacobianQuantities_block = JacobianQuantities.get_block(outJac_handle, faultNo);
            
            double VMax = lop_->rhs(faultNo, time, traction, state_block, result_block, scratch);

            lop_->getJacobianQuantities(faultNo, time, traction, state_block, result_block, JacobianQuantities_block, scratch);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
        JacobianQuantities.end_access(outJac_handle);

        //update the Jacobian
        updateJacobian(JacobianQuantities);

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

    Mat& getJacobian() { return Jacobian_; }

    tolerancesSEAS& getTolerances(){ return tolerances_; }

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


    /**
     * Calculate an approximation of the Jacobian based on finite differences
     * @param time current simulation time
     * @param x_new new solution
     */
    void calculateApproximateJacobian(double time, Vec x_new){
 
        PetscBlockVector x_new_pbv(block_size(), numLocalElements(), comm());
        PetscBlockVector rhs_new_pbv(block_size(), numLocalElements(), comm());

        double* x_n;
        CHKERRTHROW(VecGetArray(x_new, &x_n));
        auto x_n_pbv = x_new_pbv.begin_access();
        for (int i = 0; i < block_size() * numLocalElements(); ++i){
            x_n_pbv[i] = x_n[i];
        }
        CHKERRTHROW(VecRestoreArray(x_new, &x_n));
        x_new_pbv.end_access(x_n_pbv);

        rhs(time, x_new_pbv, rhs_new_pbv);

        double* x;
        double* f_n;
        double* f;
        double* J_a;
        const double* J;

        int m;
        int n;

        CHKERRTHROW(MatGetSize(Jacobian_approx_, &m, &n));

        CHKERRTHROW(VecGetArray(x_new, &x_n));
        CHKERRTHROW(VecGetArray(x_prev_, &x));
        CHKERRTHROW(VecGetArray(rhs_new_pbv.vec(), &f_n));
        CHKERRTHROW(VecGetArray(rhs_prev_, &f));

        CHKERRTHROW(MatDenseGetArrayWrite(Jacobian_approx_, &J_a));
        CHKERRTHROW(MatDenseGetArrayRead(Jacobian_, &J));

        std::cout << "approximate Jacobian is: "<<std::endl << "[ ";  
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                J_a[i + j*m] = (f_n[i] - f[i]) / (x_n[j] - x[j]);
                std::cout << J_a[i + j*m] << " ";
            }
            (i < m-1) ? std::cout <<  ";" << std::endl : std::cout <<  "]" << std::endl;
        }
        std::cout << std::endl << std::endl;

        std::cout << "analytic Jacobian is: "<<std::endl << "[ ";  
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                std::cout << J[i + j*m] << " ";
            }
            (i < m-1) ? std::cout <<  ";" << std::endl : std::cout <<  "]" << std::endl;
        }
        std::cout << std::endl << std::endl;


        CHKERRTHROW(VecRestoreArray(x_new, &x_n));
        CHKERRTHROW(VecRestoreArray(x_prev_, &x));
        CHKERRTHROW(VecRestoreArray(rhs_new_pbv.vec(), &f_n));
        CHKERRTHROW(VecRestoreArray(rhs_prev_, &f));
        CHKERRTHROW(MatDenseRestoreArrayWrite(Jacobian_approx_, &J_a));
        CHKERRTHROW(MatDenseRestoreArrayRead(Jacobian_, &J));

        setPreviousSolution(x_new, rhs_new_pbv.vec());         
    }

    void setPreviousSolution(Vec x, Vec rhs) {
        CHKERRTHROW(VecCopy(x, x_prev_)); 
        CHKERRTHROW(VecCopy(rhs, rhs_prev_)); 
    };


    private:

    /**
     * Set up the constant parts of the Jacobian matrix
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
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_));
        CHKERRTHROW(MatZeroEntries(Jacobian_));
        CHKERRTHROW(MatCreateSeqDense(comm(), totalSize, totalSize, NULL, &Jacobian_approx_));
        CHKERRTHROW(MatZeroEntries(Jacobian_approx_));



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
     * Update the Jacobian to the current state
     * @param state current solution vector
     * @param derivatives contains df/dV, df/dpsi and dg/dpsi in a vector
     */
    template <typename BlockVector>
    void updateJacobian(BlockVector& derivatives){  

        using namespace Eigen;

        size_t blockSize = this->block_size();
        size_t numFaultElements = this->numLocalElements();
        size_t totalSize = blockSize * numFaultElements;
        size_t nbf = lop_->space().numBasisFunctions();
        size_t numFaultNodes = nbf * numFaultElements;

        // fill vectors     
        VectorXd df_dV_vec(numFaultNodes);                             
        VectorXd df_dpsi_vec(numFaultNodes);
        VectorXd psi_vec(numFaultNodes);
        VectorXd V_vec(numFaultNodes);
        VectorXd g_vec(numFaultNodes);

        auto accessRead = derivatives.begin_access_readonly();
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            auto derBlock = derivatives.get_block(accessRead, noFault);    
            for(int i = 0; i<nbf; i++){ 
                df_dV_vec(   noFault * nbf + i ) = derBlock(0 * nbf + i);
                df_dpsi_vec( noFault * nbf + i ) = derBlock(1 * nbf + i);
                psi_vec(     noFault * nbf + i ) = derBlock(2 * nbf + i);
                V_vec(       noFault * nbf + i ) = derBlock(3 * nbf + i);
                g_vec(       noFault * nbf + i ) = derBlock(4 * nbf + i);                
            }
        }

        derivatives.end_access_readonly(accessRead);

        // fill helper matrices (all are transposed)
        MatrixXd dS_dS_T(numFaultNodes, numFaultNodes);
        MatrixXd dS_dpsi_T(numFaultNodes, numFaultNodes);
        MatrixXd dpsi_dS_T(numFaultNodes, numFaultNodes);
        MatrixXd dpsi_dpsi_T(numFaultNodes, numFaultNodes);
        MatrixXd A_T(numFaultNodes, numFaultNodes);
        MatrixXd B_T(numFaultNodes, numFaultNodes);
        MatrixXd C_T(numFaultNodes, numFaultNodes);  
        MatrixXd D_T(numFaultNodes, numFaultNodes);  
        MatrixXd I_CD_T(numFaultNodes, numFaultNodes); // = (I - CD)^T
        MatrixXd I_DC_T(numFaultNodes, numFaultNodes); // = (I - DC)^T

        for (int i = 0; i < numFaultNodes; i++){
            for (int j = 0; j < numFaultNodes; j++){
                dS_dS_T(i,j) = V_vec(j) / V_vec(i);
                dS_dpsi_T(i,j) = V_vec(j) / g_vec(i);                
                dpsi_dS_T(i,j) = g_vec(j) / V_vec(i);
                dpsi_dpsi_T(i,j) = g_vec(j) / g_vec(i);                
            }
        }        

        FullPivLU<MatrixXd> dS_dS_T_lu(dS_dS_T);
        FullPivLU<MatrixXd> dpsi_dpsi_T_lu(dpsi_dpsi_T);
        for (int i = 0; i < numFaultNodes; ++i){    
            C_T.col(i) = dS_dS_T_lu.solve(dpsi_dS_T.col(i));        // C^T = dS/dS^-T . dpsi/dS^T
            D_T.col(i) = dpsi_dpsi_T_lu.solve(dS_dpsi_T.col(i));    // D^T = dpsi/dpsi^-T . dS/dpsi^T
        }
        std::cout << "dS/dS^T = " << std::endl << dS_dS_T << std::endl << std::endl;
        std::cout << "dpsi/dpsi^T = " << std::endl << dpsi_dpsi_T << std::endl << std::endl;
        std::cout << "dS/dS^T * C^T = dpsi/dS^T" << std::endl << dS_dS_T * C_T << std::endl << std::endl;
        std::cout << "dS/dpsi^T = " << std::endl << dS_dpsi_T << std::endl << std::endl;
        std::cout << "dpsi/dS^T = " << std::endl << dpsi_dS_T << std::endl << std::endl;
        std::cout << "C^T = " << std::endl << C_T << std::endl << std::endl;
        std::cout << "D^T = " << std::endl << D_T << std::endl << std::endl;

        const double* df_dS;
        CHKERRTHROW(MatDenseGetArrayRead(df_dS_, &df_dS));
        for (int i = 0; i < numFaultNodes; i++){
            for (int j = 0; j < numFaultNodes; j++){
                // A^T = (df/dS^T + C^T . df/dpsi^T) . df/dV^-T
                A_T(i,j) = (df_dS[j + i * numFaultNodes] + C_T(i,j) * df_dpsi_vec(i)) / df_dV_vec(j);

                // B^T = (df/dpsi^T + D^T . df/dS^T) . df/dV^-T
                B_T(i,j) = df_dpsi_vec(i) * delta(i,j);
                for  (int k = 0; k < numFaultNodes; k++){
                    B_T(i,j) += D_T(i,k) * df_dS[j + k * numFaultNodes];
                }
                B_T(i,j) /= df_dV_vec(j);
            }
        }
        CHKERRTHROW(MatDenseRestoreArrayRead(df_dS_, &df_dS));

        // (I - CD)^T = I - D^T . C^T
        I_CD_T = MatrixXd::Identity(numFaultNodes, numFaultNodes) - D_T * C_T;
        // (I - DC)^T = I - C^T . D^T
        I_DC_T = MatrixXd::Identity(numFaultNodes, numFaultNodes) - C_T * D_T;

        MatrixXd CB_A = C_T * B_T - A_T; // = C^T * B^T - A^T
        MatrixXd DA_B = D_T * A_T - B_T; // = D^T * A^T - B^T
       
        FullPivLU<MatrixXd> I_CD_T_lu(I_CD_T);
        FullPivLU<MatrixXd> I_DC_T_lu(I_DC_T);

        MatrixXd dV_dS(numFaultNodes, numFaultNodes);   // = (I - DC)^-T . (C^T . B^T - A^T)
        MatrixXd dV_dpsi(numFaultNodes, numFaultNodes); // = (I - CD)^-T . (D^T . A^T - B^T)

        for (int i = 0; i < numFaultNodes; ++i){    // fill all rows of the matrices
            dV_dS.row(i) = I_DC_T_lu.solve(CB_A.col(i));
            dV_dpsi.row(i) = I_CD_T_lu.solve(DA_B.col(i));
        }

        // fill the Jacobian
        int V_i;
        int PSI_i;
        int V_j;
        int PSI_j;
        int n_i;
        int n_j;


        double dg_dS;
        double dg_dpsi;

        double* J;
        CHKERRTHROW(MatDenseGetArrayWrite(Jacobian_, &J));
        for (int noFault = 0; noFault < numFaultElements; noFault++){
            for (int i = 0; i < nbf; i++){                  // row iteration
                V_i = noFault * blockSize + i;
                PSI_i = noFault * blockSize + 
                                       RateAndStateBase::TangentialComponents * nbf + i;
                n_i = noFault * nbf + i;
                
                

                for (int noFault2 = 0; noFault2 < numFaultElements; noFault2++){
                    for(int j = 0; j < nbf; j++) {          // column iteration

                        V_j = noFault2 * blockSize + j;
                        PSI_j = noFault2 * blockSize + 
                                                RateAndStateBase::TangentialComponents * nbf + j;
                        n_j = noFault2 * nbf + j;

                        // column major in df_dS and J!                        
                        lop_->getAgeingDerivatives(delta(n_i, n_j), psi_vec(n_i), 
                                                  dV_dS(n_i,n_j), dV_dpsi(n_i,n_j), 
                                                  dg_dS, dg_dpsi);

                        J[V_i   + V_j * totalSize  ] = dV_dS(n_i, n_j);
                        J[V_i   + PSI_j * totalSize] = dV_dpsi(n_i, n_j);
                        J[PSI_i + V_j * totalSize  ] = dg_dS;
                        J[PSI_i + PSI_j * totalSize] = dg_dpsi;
                    }
                }
            }
        }

        // int m;
        // int n;
        // CHKERRTHROW(MatGetSize(Jacobian_, &m, &n));

        // std::cout << "Jacobian is: "<<std::endl << "[ ";  
        // for (int i = 0; i < m; i++){
        //     for (int j = 0; j < n; j++){
        //         std::cout << J[i + j*m] << " ";
        //     }
        //     (i < m-1) ? std::cout <<  ";" << std::endl : std::cout <<  "]" << std::endl;
        // }
        // std::cout << std::endl << std::endl;


        CHKERRTHROW(MatDenseRestoreArrayWrite(Jacobian_, &J));
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
    Mat Jacobian_;                          // Jacobian matrix     
    Mat Jacobian_approx_;                   // approximated Jacobian matrix     
    tolerancesSEAS tolerances_;
    Vec x_prev_;                            // solution at the previous timestep
    Vec rhs_prev_;                          // rhs evaluation at the previous timestep
};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
