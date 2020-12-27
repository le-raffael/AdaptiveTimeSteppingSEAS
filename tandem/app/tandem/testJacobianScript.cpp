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
    auto topo = std::make_shared<DGOperatorTopo>(mesh, MPI_COMM_WORLD);
    auto adapt = adapter<type>::make(cfg, scenario, cl, topo, fop->space().clone());

    auto seasop = std::make_shared<seas_op_t>(std::move(fop), std::move(adapt));
    seasop->lop().set_constant_params(friction_scenario.constant_params());
    seasop->lop().set_params(friction_scenario.param_fun());
    if (friction_scenario.source_fun()) {
        seasop->lop().set_source_fun(*friction_scenario.source_fun());
    }
    if (scenario.boundary()) {
        seasop->set_boundary(*scenario.boundary());
    } 


    /************ BEGIN of new code ***************/

    blockSize = seasop->block_size();
    nbf = 0.5 * blockSize;
    numFaultElements = seasop->numLocalElements();
    totalSize = blockSize * numFaultElements;

    double dt = 1.0;    
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
        seasop->rhs(nextTime, xBlock, fBlock);
        VectorXd f(totalSize);
        writeBlockToEigen(fBlock, f);
        return f;
        };

    double tol = 1e-7;
    VectorXd x = x_init;

    VectorXd fx = -x + x_init + dt * rhs(x);;

    // 1. first guess (explicit Euler)
//    VectorXd x_n = x + dt * rhs(x);
    VectorXd x_n = x;
    VectorXd fx_n = rhs(x_n);
//    VectorXd fx_n = -x_n + x_init + dt * rhs(x_n);

    // 2. initialize the Jacobi matrix by finite differences
    MatrixXd J = seasop->getJacobian();
//    for(int i=0; i<totalSize;i++){
//        for(int j=0; j<totalSize;j++){
//            J(i,j) = (fx_n(i) - fx(i)) / (x_n(j) - x(j));
//        }
//    }

    std::cout<<J<<std::endl;

    // difference to the previous iteration step
    VectorXd dx_n(totalSize);

    int k = 0;

    while (fx_n.squaredNorm() > tol * tol){
        k++;
        std::cout<<k<<" with norm: "<<fx_n.norm()<<std::endl;
        // 4. calculate next time step
        x = x_n;
        fx = fx_n;
        x_n = x - J.fullPivLu().solve(fx);

        // 5. update Jacobian
//        fx_n = rhs(x_n);
        fx_n = -x_n + x_init + dt * rhs(x_n);
        dx_n = x_n - x;
        VectorXd df =  fx_n - fx;       

        J = J + (fx_n - fx - J.fullPivLu().solve(dx_n)) * dx_n.transpose() / x_n.squaredNorm();
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

/********
 * right hand side of the implicit Euler method F = -x_n + x_n_1 + dt * f(x_n)
 * ******/
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
