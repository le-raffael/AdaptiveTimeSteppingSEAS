#include "GlobalVariables.h"
#include "SEAS.h"
#include "common/PetscTimeSolver.h"
#include "config.h"
#include "localoperator/Elasticity.h"
#include "localoperator/Poisson.h"
#include "tandem/Config.h"
#include "tandem/DieterichRuinaAgeing.h"
#include "tandem/FrictionConfig.h"
#include "tandem/RateAndState.h"
#include "tandem/SeasElasticityAdapter.h"
#include "tandem/SeasOperator.h"
#include "tandem/SeasPoissonAdapter.h"
#include "tandem/SeasScenario.h"
#include "tandem/SeasWriter.h"

#include "form/DGOperatorTopo.h"
#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"

#include <mpi.h>
#include <petscsys.h>
#include <petscksp.h>

#include <algorithm>
#include <ctime>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tndm::detail {

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
void solve_seas_problem(LocalSimplexMesh<DomainDimension> const& mesh, Config& cfg) {
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

    // detect the aseismic phase/earthquake formulations
    if((cfg.solver->solver_aseismicslip->solution_size == "compact")  
    && (cfg.solver->solver_aseismicslip->problem_formulation == "ode")) 
        cfg.solver->solver_aseismicslip->formulation = tndm::FIRST_ORDER_ODE;
    if((cfg.solver->solver_aseismicslip->solution_size == "compact")  
    && (cfg.solver->solver_aseismicslip->problem_formulation == "dae")) 
        cfg.solver->solver_aseismicslip->formulation = tndm::COMPACT_DAE;
    if((cfg.solver->solver_aseismicslip->solution_size == "extended") 
    && (cfg.solver->solver_aseismicslip->problem_formulation == "dae")) 
        cfg.solver->solver_aseismicslip->formulation = tndm::EXTENDED_DAE;
    if((cfg.solver->solver_aseismicslip->solution_size == "extended") 
    && (cfg.solver->solver_aseismicslip->problem_formulation == "ode")) 
        cfg.solver->solver_aseismicslip->formulation = tndm::SECOND_ORDER_ODE;

    if((cfg.solver->solver_earthquake->solution_size == "compact")  
    && (cfg.solver->solver_earthquake->problem_formulation == "ode")) 
        cfg.solver->solver_earthquake->formulation = tndm::FIRST_ORDER_ODE;
    if((cfg.solver->solver_earthquake->solution_size == "compact")  
    && (cfg.solver->solver_earthquake->problem_formulation == "dae")) 
        cfg.solver->solver_earthquake->formulation = tndm::COMPACT_DAE;
    if((cfg.solver->solver_earthquake->solution_size == "extended") 
    && (cfg.solver->solver_earthquake->problem_formulation == "dae")) 
    cfg.solver->solver_earthquake->formulation = tndm::EXTENDED_DAE;
    if((cfg.solver->solver_earthquake->solution_size == "extended")
    && (cfg.solver->solver_earthquake->problem_formulation == "ode"))
        cfg.solver->solver_earthquake->formulation = tndm::SECOND_ORDER_ODE;


    auto seasop = std::make_shared<seas_op_t>(std::move(fop), std::move(adapt), cfg.solver);
    seasop->lop().set_constant_params(friction_scenario.constant_params());
    seasop->lop().set_params(friction_scenario.param_fun());
    if (friction_scenario.source_fun()) {
        seasop->lop().set_source_fun(*friction_scenario.source_fun());
    }
    if (scenario.boundary()) {
        seasop->set_boundary(*scenario.boundary());
    } 

    std::cout<<"Perform simulation with "<<seasop->numLocalElements()<<" fault elements " << std::endl;

    auto t_begin = std::chrono::high_resolution_clock::now();

    int ierr = 0;
    auto ts = PetscTimeSolver(*seasop, ksp, ierr);
    if (ierr != 0) return;

    std::unique_ptr<seas_writer_t> writer;
    if (cfg.output) {
        writer = std::make_unique<seas_writer_t>(cfg.output->prefix, mesh, cl, seasop,
                                                 PolynomialDegree, cfg.output->V_ref,
                                                 cfg.output->t_min, cfg.output->t_max);
        ts.set_monitor(*writer);
    }

    auto t_intermediate = std::chrono::high_resolution_clock::now();

    ts.solve(cfg.final_time);

    auto t_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_initialization = std::chrono::duration_cast<std::chrono::duration<double>>(t_intermediate - t_begin);
    std::chrono::duration<double> time_solve          = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_intermediate);
    std::chrono::duration<double> time_total          = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_begin);

    auto filename = writer->writeTimes(time_initialization.count(), time_solve.count(), time_total.count());

    std::cout << "Finished simulation successfully after " << time_total.count() << " seconds "<< std::endl; 
    std::cout << "Results written to: " << filename << std::endl;

    auto solution = scenario.solution(cfg.final_time);
    if (solution) {
        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        auto numeric = seasop->adapter().displacement();
        double error =
            tndm::Error<DomainDimension>::L2(*cl, numeric, *solution, 0, PETSC_COMM_WORLD);
        if (rank == 0) {
            std::cout << "L2 error: " << error << std::endl;
        }
    }
}

} // namespace tndm::detail

namespace tndm {

void solveSEASProblem(LocalSimplexMesh<DomainDimension> const& mesh, Config& cfg) {
    if (cfg.seas.type == SeasType::Poisson) {
        detail::solve_seas_problem<SeasType::Poisson>(mesh, cfg);
    } else if (cfg.seas.type == SeasType::Elasticity) {
        detail::solve_seas_problem<SeasType::Elasticity>(mesh, cfg);
    }
}

} // namespace tndm
