#include "testJacobianScript.h"

#include "kernels/poisson_adapter/tensor.h"

namespace tndm::detail {

using namespace Eigen;

size_t numFaultElements;
size_t nbf;
size_t blockSize;
size_t totalSize;

template <SeasType type> struct adapter;
template <> struct adapter<SeasType::Poisson> {
    using type = SeasPoissonAdapter;
    static auto make(Config const& cfg, SeasScenario<Poisson> const& scenario,
                     std::shared_ptr<Curvilinear<DomainDimension>> cl,
                     std::shared_ptr<DGOperatorTopo> topo,
                     std::unique_ptr<RefElement<DomainDimension - 1u>> space) {
        auto lop = std::make_unique<Poisson>(cl, scenario.mu());
        return std::make_unique<SeasPoissonAdapter>(std::move(cl), std::move(topo),
                                                    std::move(space), std::move(lop), cfg.seas.up,
                                                    cfg.seas.ref_normal);
    }
};
template <> struct adapter<SeasType::Elasticity> {
    using type = SeasElasticityAdapter;
    static auto make(Config const& cfg, SeasScenario<Elasticity> const& scenario,
                     std::shared_ptr<Curvilinear<DomainDimension>> cl,
                     std::shared_ptr<DGOperatorTopo> topo,
                     std::unique_ptr<RefElement<DomainDimension - 1u>> space) {
        auto lop = std::make_unique<Elasticity>(cl, scenario.lam(), scenario.mu());
        return std::make_unique<SeasElasticityAdapter>(std::move(cl), std::move(topo),
                                                       std::move(space), std::move(lop),
                                                       cfg.seas.up, cfg.seas.ref_normal);
    }
};


template <SeasType type>
void solve_Jacobian(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    using adapter_t = typename adapter<type>::type;
    using adapter_lop_t = typename adapter<type>::type::local_operator_t;
    using fault_op_t = RateAndState<DieterichRuinaAgeing>;
    using seas_op_t = SeasOperator<fault_op_t, adapter_t>;
    using seas_writer_t = SeasWriter<DomainDimension, seas_op_t>;

    auto scenario = SeasScenario<adapter_lop_t>(cfg.seas);
    auto friction_scenario = DieterichRuinaAgeingScenario(cfg.friction);

    auto cl = std::make_shared<Curvilinear<DomainDimension>>(mesh, scenario.transform(),
                                                             PolynomialDegree);

    auto fop = std::make_unique<fault_op_t>(cl);
    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);
    auto adapt = adapter<type>::make(cfg, scenario, cl, topo, fop->space().clone());

    KSP& ksp = adapt->getKSP();
    
    auto seasop = std::make_shared<seas_op_t>(std::move(fop), std::move(adapt));
    seasop->lop().set_constant_params(friction_scenario.constant_params());
    seasop->lop().set_params(friction_scenario.param_fun());
    if (friction_scenario.source_fun()) {
        seasop->lop().set_source_fun(*friction_scenario.source_fun());
    }
    if (scenario.boundary()) {
        seasop->set_boundary(*scenario.boundary());
    } 

    if (cfg.solver){
        // set ksp type
        CHKERRTHROW(KSPSetType(ksp, cfg.solver->ksp_type.c_str()));

        // set preconditioner options
        PC pc;
        CHKERRTHROW(KSPGetPC(ksp, &pc));
        CHKERRTHROW(PCSetType(pc, cfg.solver->pc_type.c_str()));
        CHKERRTHROW(PCFactorSetMatSolverType(pc, cfg.solver->pc_factor_mat_solver_type.c_str()));
    }


    /************ BEGIN of new code ***************/
    blockSize = seasop->block_size();
    nbf = seasop->lop().space().numBasisFunctions();
    numFaultElements = seasop->numLocalElements();
    totalSize = blockSize * numFaultElements;

    std::cout << "Test the Jacobian for a symmetric domain with " <<numFaultElements 
              << " fault elements" << std::endl;

    double dt = 1e7;    
    double nextTime = dt;


    // initialize full solution vector
    PetscBlockVector xB = PetscBlockVector(blockSize, numFaultElements, seasop->comm());
    seasop->initial_condition(xB);
    VectorXd x_init(totalSize);
    writeBlockToEigen(xB, x_init);

    // get handle for the right hand side
    auto rhs = [seasop, nextTime](VectorXd& x) ->  VectorXd {
        PetscBlockVector xBlock(blockSize, numFaultElements, MPI_COMM_WORLD);
        PetscBlockVector fBlock(blockSize, numFaultElements, MPI_COMM_WORLD);
        writeEigenToBlock(x, xBlock);
        seasop->rhs(nextTime, xBlock, fBlock);  // executes the Jacobian update
        VectorXd f(totalSize);
        writeBlockToEigen(fBlock, f);
        return f;
        };

    double tol = 1e-7;

    /* **************************************************
     * test of the Jacobian on an easy Newton Iteration *
     ************************************************** */
    std::cout << "Newton iteration with the implicit Euler " << std::endl; 


    VectorXd x = x_init;

    // 0. first guess (explicit Euler)    
    // VectorXd x_n = x + dt * rhs(x);
 
    // 0. first guess (midpoint Euler - RK2)    
    // VectorXd intermediate = x + 0.5 * dt * rhs(x);
    // VectorXd x_n = x + dt * rhs(intermediate);

    dt = 1e7;

    VectorXd x_n = x_init;

    // 1. first evaluation of the implicit Euler with the guess
    VectorXd fx_n = -x_n + x_init + dt * rhs(x_n);

    // 2. initialize the Jacobi matrix
    MatrixXd J = JacobianImplicitEuler(seasop->getJacobian(), dt);

    int k = 0;

    double maxErrorPSI=0;
    double maxErrorS=0;

    while (fx_n.squaredNorm() > tol * tol){
        k++;
        for ( int noFault = 0; noFault < numFaultElements; ++noFault){
            for (int i = 0; i < nbf; i++){
                maxErrorS = std::max(maxErrorS, abs(fx_n(noFault * blockSize + i)));
                maxErrorPSI = std::max(maxErrorPSI, abs(fx_n(noFault * blockSize + i+nbf)));
            }
        }
        std::cout<<"iteration " << k << " with norm: " << fx_n.norm() << ", max error in S: "<<maxErrorS<< " and max error in PSI: "<< maxErrorPSI << std::endl;
        maxErrorS = 0;
        maxErrorPSI = 0;
        // 4. calculate next iteration step
        x_n = x_n - J.fullPivLu().solve(fx_n);

        // 5. update Jacobian
        fx_n = -x_n + x_init + dt * rhs(x_n);
        J = JacobianImplicitEuler(seasop->getJacobian(), dt);
    }
    std::cout << "final norm: "<<fx_n.norm()<<std::endl;
    VectorXd solutionDirect = x_n;


    J = 1 / dt * (J + MatrixXd::Identity(totalSize, totalSize));

    /* ***********************************************************
     * verification of the Jacobian on an easy Broyden Iteration *
     *********************************************************** */
    std::cout << " Broyden iteration on the same problem " << std::endl;

    VectorXd x_n_1 = x_init;

    // 0. first guess (explicit Euler)    
    x_n = x_n_1 + dt * rhs(x_n_1);


    // 1. first evaluation of the implicit Euler with the guess
    VectorXd f_n = -x_n + x_init + dt * rhs(x_n);
    // 2. initialize the Jacobi matrix 
    MatrixXd J_b = JacobianImplicitEuler(seasop->getJacobian(), dt);

    // difference to the previous iteration step
    VectorXd dx_n(totalSize);
    VectorXd df_n(totalSize);
    VectorXd f_n_1(totalSize);

    k = 0;

    while (f_n.norm() > 1e-15){
        k++;
        std::cout<<"iteration " << k << " with norm: " << f_n.norm() << std::endl;
        x_n_1 = x_n;
        f_n_1 = f_n;

        // 4. calculate next iteration step
        x_n = x_n_1 - J_b.fullPivLu().solve(f_n);
        f_n = -x_n + x_init + dt * rhs(x_n);

        dx_n = x_n - x_n_1;
        df_n = f_n - f_n_1;

        // 5. update Jacobian
        J_b = J_b + (df_n - J_b * dx_n) / (dx_n.squaredNorm()) * dx_n.transpose();
    }

    J_b = 1 / dt * (J_b + MatrixXd::Identity(totalSize, totalSize));

    std::cout << "final norm: "<<f_n.norm()<<std::endl;
    MatrixXd diff = J - J_b;
 
    std::cout << "L2-norm of the difference between the Jacobians: " << diff.norm() << std::endl;
    std::cout << "inf-norm of the difference between the Jacobians: " << diff.lpNorm<Infinity>() << std::endl;

    for (int i = 0; i < numFaultElements; ++i){
        std::cout << " Difference in the Jacobi for element " << i << ": " << std::endl << diff.block(0,i*blockSize,totalSize,blockSize) << std::endl;
    }
    std::cout << " Jacobi of Broyden: " << std::endl << J_b.block(0,4*blockSize,totalSize,blockSize) << std::endl;
    std::cout << " Analytical Jacobi: " << std::endl << J.block(0,4*blockSize,totalSize,blockSize) << std::endl;
};


void writeBlockToEigen(PetscBlockVector& BlockVector, VectorXd& EigenVector){

    auto AccessHandle = BlockVector.begin_access_readonly();

    for (int faultNo = 0; faultNo < numFaultElements; faultNo++){
        auto localBlock = BlockVector.get_block(AccessHandle, faultNo);    
        for (int i = 0; i < blockSize; i++){
            // fill the solution vector
            EigenVector(i + faultNo * blockSize) = localBlock.data()[i];
        }
    }
    BlockVector.end_access_readonly(AccessHandle);
}


void writeEigenToBlock(VectorXd& EigenVector, PetscBlockVector& BlockVector){

    auto AccessHandle = BlockVector.begin_access();

    for (int faultNo = 0; faultNo < numFaultElements; faultNo++){
        auto localBlock = BlockVector.get_block(AccessHandle, faultNo);   
        for (int i = 0; i < blockSize; i++){
            // fill the solution vector
            localBlock.data()[i] = EigenVector(i + faultNo * blockSize);
        }
    }
    BlockVector.end_access(AccessHandle);
}

MatrixXd JacobianImplicitEuler(Mat& J, double dt){
    // transform J from Petsc dense matrix(column major!) to Eigen matrix
    MatrixXd JE(totalSize, totalSize);
    const double* Ja;
    CHKERRTHROW(MatDenseGetArrayRead(J, &Ja));
    for(int i = 0; i < totalSize; i++){
        for(int j = 0; j < totalSize; j++){
            JE(i,j) = Ja[i + j * totalSize];
        }
    }
    CHKERRTHROW(MatDenseRestoreArrayRead(J, &Ja));
//    std::cout << "Jacobian is: " << std::endl << JE.block(0, 0, totalSize, 6) << std::endl;
    return -MatrixXd::Identity(totalSize, totalSize) + dt * JE;
}


/**
 * right hand side of the implicit Euler method F = -x_n + x_n_1 + dt * f(x_n)
 */
VectorXd F(double dt,VectorXd& x, VectorXd& x_old, VectorXd rhs(VectorXd&)){
    return -x + x_old + dt * rhs(x);
}


} // namespace tndm::detail

namespace tndm {

void testJacobianScript(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    if (cfg.seas.type == SeasType::Poisson) {
        detail::solve_Jacobian<SeasType::Poisson>(mesh, cfg);
    } else if (cfg.seas.type == SeasType::Elasticity) {
       // detail::solve_Jacobian<SeasType::Elasticity>(mesh, cfg);
    }
}
} // namespace tndm
