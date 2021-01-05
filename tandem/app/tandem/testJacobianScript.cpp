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

/**
 * Calculates the Jacobi matrix for the implicit Euler
 * @param J Jacobian of the rhs
 * @param dt timestep size
 */
MatrixXd JacobianImplicitEuler(MatrixXd& J, double dt){
    return -MatrixXd::Identity(totalSize, totalSize) + dt * J;
}

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

    double dt = 1e5;    
    double nextTime = dt;


    // initialize full solution vector
    PetscBlockVector xB = PetscBlockVector(blockSize, numFaultElements, seasop->comm());
    seasop->initial_condition(xB);
    VectorXd x_init(totalSize);
    writeBlockToEigen(xB, x_init);


    /* **************************************************
     * test of the Jacobian on an easy Newton Iteration *
     ************************************************** */

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
    VectorXd x = x_init;

    // 0. first guess (explicit Euler)    
    VectorXd x_n = x + dt * rhs(x);

    // 1. first evaluation of the implicit Euler with the guess
    VectorXd fx_n = -x_n + x_init + dt * rhs(x_n);

    // 2. initialize the Jacobi matrix
    MatrixXd J = JacobianImplicitEuler(seasop->getJacobian(), dt);

    std::cout<<"First columns of the Implicit Euler Jacobian (corresponds to fault element 1): "<<std::endl;
    std::cout<<J.block(0, 0, totalSize, blockSize)<<std::endl;

    // difference to the previous iteration step
    VectorXd dx_n(totalSize);

    int k = 0;

    while (fx_n.squaredNorm() > tol * tol){
        k++;
        std::cout<<"iteration " << k << " with norm: " << fx_n.norm() << std::endl;
        // 4. calculate next iteration step
        x_n = x_n - J.fullPivLu().solve(fx_n);

        // 5. update Jacobian
        fx_n = -x_n + x_init + dt * rhs(x_n);
        J = JacobianImplicitEuler(seasop->getJacobian(), dt);
    }
    std::cout << "final norm: "<<fx_n.norm()<<std::endl;

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
