#include "common/CmdLine.h"
#include "common/MeshConfig.h"
#include "config.h"
#include "tandem/AdaptiveOutputStrategy.h"
#include "tandem/Config.h"
#include "tandem/FrictionConfig.h"
#include "tandem/SEAS.h"
#include "tandem/SeasScenario.h"

#include "io/GMSHParser.h"
#include "io/GlobalSimplexMeshBuilder.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"

#include <argparse.hpp>
#include <mpi.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace tndm;

int main(int argc, char** argv) {
    int pArgc = 0;
    char** pArgv = nullptr;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--petsc") == 0) {
            pArgc = argc - i;
            pArgv = argv + i;
            argc = i;
            break;
        }
    }

    argparse::ArgumentParser program("tandem");
    program.add_argument("--petsc").help("PETSc options, must be passed last!");
    program.add_argument("config").help("Configuration file (.toml)");

    auto makePathRelativeToConfig =
        MakePathRelativeToOtherPath([&program]() { return program.get("config"); });

    TableSchema<Config> schema;
    schema.add_value("resolution", &Config::resolution)
        .validator([](auto&& x) { return x > 0; })
        .help("Non-negative resolution parameter");
    schema.add_value("final_time", &Config::final_time)
        .validator([](auto&& x) { return x >= 0; })
        .help("Non-negative final time of simulation");
    schema.add_value("mesh_file", &Config::mesh_file)
        .converter(makePathRelativeToConfig)
        .validator(PathExists());
    auto& seasSchema = schema.add_table("seas", &Config::seas);
    SeasScenarioConfig::setSchema(seasSchema, makePathRelativeToConfig);
    auto& frictionSchema = schema.add_table("friction", &Config::friction);
    DieterichRuinaAgeingConfig::setSchema(frictionSchema, makePathRelativeToConfig);
    auto& genMeshSchema = schema.add_table("generate_mesh", &Config::generate_mesh);
    GenMeshConfig<DomainDimension>::setSchema(genMeshSchema);
    auto& outputSchema = schema.add_table("output", &Config::output);
    outputSchema.add_value("prefix", &OutputConfig::prefix).help("Output file name prefix");
    outputSchema.add_value("V_ref", &OutputConfig::V_ref)
        .validator([](auto&& x) { return x > 0; })
        .default_value(0.1)
        .help("Output is written every t_min if this slip-rate is reached");
    outputSchema.add_value("t_min", &OutputConfig::t_min)
        .validator([](auto&& x) { return x > 0; })
        .default_value(0.1)
        .help("Minimum output interval");
    outputSchema.add_value("t_max", &OutputConfig::t_max)
        .validator([](auto&& x) { return x > 0; })
        .default_value(365 * 24 * 3600)
        .help("Maximum output interval");
    outputSchema.add_value("strategy", &OutputConfig::strategy)
        .default_value(AdaptiveOutputStrategy::Threshold)
        .help("Adaptive output strategy")
        .converter([](std::string_view value) {
            if (iEquals(value, "threshold")) {
                return AdaptiveOutputStrategy::Threshold;
            } else if (iEquals(value, "exponential")) {
                return AdaptiveOutputStrategy::Exponential;
            } else {
                return AdaptiveOutputStrategy::Unknown;
            }
        })
        .validator([](AdaptiveOutputStrategy const& type) {
            return type != AdaptiveOutputStrategy::Unknown;
        });
    auto& solverSchema = schema.add_table("solver", &Config::solver);
    solverSchema.add_value("adapt_wnormtype", &SolverConfigGeneral::adapt_wnormtype)
        .default_value("infinity")
        .help("norm to estimate the local truncation error");
    solverSchema.add_value("ksp_type", &SolverConfigGeneral::ksp_type)
        .default_value("preonly")
        .help("type of the ksp matrix-vector multiplication procedure");
    solverSchema.add_value("pc_type", &SolverConfigGeneral::pc_type)
        .default_value("lu")
        .help("type of the preconditioner");
    solverSchema.add_value("pc_factor_mat_solver_type", &SolverConfigGeneral::pc_factor_mat_solver_type)
        .default_value("mumps")
        .help("type of the matrix solver procedure in the preconditioner");

    auto& earthquakeSchema = solverSchema.add_table("earthquake", &SolverConfigGeneral::solver_earthquake);
    earthquakeSchema.add_value("solution_size", &SolverConfigSpecific::solution_size)
        .validator([](auto&& x) { return ((x == "compact") || (x == "extended")); })
        .default_value("compact")
        .help("Whether to include the velocity in the solution vector. ['compact', 'extended']");
    earthquakeSchema.add_value("problem_formulation", &SolverConfigSpecific::problem_formulation)
        .validator([](auto&& x) { return ((x == "ode") || (x == "dae")); })
        .default_value("ode")
        .help("The formulation of the SEAS problem. ['ode', 'dae']");
    earthquakeSchema.add_value("type", &SolverConfigSpecific::type)
        .default_value("rk")
        .help("type of the time integration scheme. ['rk', 'bdf', ... ]");
    earthquakeSchema.add_value("rk_type", &SolverConfigSpecific::rk_type)
        .default_value("5dp")
        .help("type of the Runge-Kutta scheme (use Petsc standard). Does not need to be provided if no Runge-Kutta scheme is used. ['3bs', '5dp', ... ]");
    earthquakeSchema.add_value("bdf_order", &SolverConfigSpecific::bdf_order)
        .validator([](auto&& x) { return ((x >= 0) || (x <= 6)); })
        .default_value(4)
        .help("Order of the BDF scheme. Does not need to be provided if no BDF scheme is used. Set 0 to adaptively change the order.");
    earthquakeSchema.add_value("S_atol", &SolverConfigSpecific::S_atol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("absolute tolerance for the slip");
    earthquakeSchema.add_value("S_rtol", &SolverConfigSpecific::S_rtol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("relative tolerance for the slip");
    earthquakeSchema.add_value("psi_atol", &SolverConfigSpecific::psi_atol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("absolute tolerance for the state variable psi");
    earthquakeSchema.add_value("psi_rtol", &SolverConfigSpecific::psi_rtol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("relative tolerance for the state variable psi");
    earthquakeSchema.add_value("V_atol", &SolverConfigSpecific::V_atol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("absolute tolerance for the slip rate");
    earthquakeSchema.add_value("V_rtol", &SolverConfigSpecific::V_rtol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("relative tolerance for the slip rate");
    earthquakeSchema.add_value("bdf_custom_error_evaluation", &SolverConfigSpecific::bdf_custom_error_evaluation)
        .default_value(false)
        .help("Only for BDF methods:\n  -[false] to use the default error estimate with Lagrange polynomials. \n  -[true] for a custom evaluation with an other BDF evaluation of lower order");
    earthquakeSchema.add_value("bdf_custom_Newton_iteration", &SolverConfigSpecific::bdf_custom_Newton_iteration)
        .default_value(false)
        .help("Only for BDF methods:\n  -[false] to use the default Newton iteration.\n  - [true] for a custom implementation (necessary for the extended ODE formulation)");
    earthquakeSchema.add_value("bdf_custom_LU_solver", &SolverConfigSpecific::bdf_custom_LU_solver)
        .default_value(false)
        .help("Only for BDF methods:\n  -[false] to use the default linear solver in the Newton iteration O(n^3).\n  - [true] for a custom implementation with partial initial LU in O(n^2)");
    earthquakeSchema.add_value("custom_time_step_adapter", &SolverConfigSpecific::custom_time_step_adapter)
        .default_value(false)
        .help("Estimation of the new time step size:\n  -[false] to use the basic PETSc built-in adapter.\n  -[true] to use a custom, implementation-dependent adapter.\n Deverloper note: this parameter is likely to be changed to string to allow the choice between different custom adapter");
    earthquakeSchema.add_value("bdf_ksp_type", &SolverConfigSpecific::bdf_ksp_type)
        .default_value("preonly")
        .help("type of the ksp matrix-vector multiplication procedure for the Jacobian");
    earthquakeSchema.add_value("bdf_pc_type", &SolverConfigSpecific::bdf_pc_type)
        .default_value("lu")
        .help("type of the preconditioner for the Jacobian");

    auto& aseismicSlipSchema = solverSchema.add_table("aseismic_slip", &SolverConfigGeneral::solver_aseismicslip);
    aseismicSlipSchema.add_value("solution_size", &SolverConfigSpecific::solution_size)
        .validator([](auto&& x) { return ((x == "compact") || (x == "extended")); })
        .default_value("compact")
        .help("Whether to include the velocity in the solution vector. ['compact', 'extended']");
    aseismicSlipSchema.add_value("problem_formulation", &SolverConfigSpecific::problem_formulation)
        .validator([](auto&& x) { return ((x == "ode") || (x == "dae")); })
        .default_value("ode")
        .help("The formulation of the SEAS problem. ['ode', 'dae']");
    aseismicSlipSchema.add_value("type", &SolverConfigSpecific::type)
        .default_value("rk")
        .help("type of the time integration scheme. ['rk', 'bdf', ... ]");
    aseismicSlipSchema.add_value("rk_type", &SolverConfigSpecific::rk_type)
        .default_value("5dp")
        .help("type of the Runge-Kutta scheme (use Petsc standard). Does not need to be provided if no Runge-Kutta scheme is used. ['3bs', '5dp', ... ]");
    aseismicSlipSchema.add_value("bdf_order", &SolverConfigSpecific::bdf_order)
        .validator([](auto&& x) { return ((x > 0) || (x <= 6)); })
        .default_value(4)
        .help("Order of the BDF scheme. Does not need to be provided if no BDF scheme is used");
    aseismicSlipSchema.add_value("S_atol", &SolverConfigSpecific::S_atol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("absolute tolerance for the slip");
    aseismicSlipSchema.add_value("S_rtol", &SolverConfigSpecific::S_rtol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("relative tolerance for the slip");
    aseismicSlipSchema.add_value("psi_atol", &SolverConfigSpecific::psi_atol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("absolute tolerance for the state variable psi");
    aseismicSlipSchema.add_value("psi_rtol", &SolverConfigSpecific::psi_rtol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("relative tolerance for the state variable psi");
    aseismicSlipSchema.add_value("V_atol", &SolverConfigSpecific::V_atol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("absolute tolerance for the slip rate - Only nededed if the solution size is extended");
    aseismicSlipSchema.add_value("V_rtol", &SolverConfigSpecific::V_rtol)
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-7)
        .help("relative tolerance for the slip rate - Only nededed if the solution size is extended");
    aseismicSlipSchema.add_value("bdf_custom_error_evaluation", &SolverConfigSpecific::bdf_custom_error_evaluation)
        .default_value(false)
        .help("Only for BDF methods:\n  -[false] to use the default error estimate with Lagrange polynomials. \n  -[true] for a custom evaluation with an other BDF evaluation of lower order");
    aseismicSlipSchema.add_value("bdf_custom_Newton_iteration", &SolverConfigSpecific::bdf_custom_Newton_iteration)
        .default_value(false)
        .help("Only for BDF methods:\n  -[false] to use the default Newton iteration.\n  - [true] for a custom implementation (necessary for the extended ODE formulation)");
    aseismicSlipSchema.add_value("bdf_custom_LU_solver", &SolverConfigSpecific::bdf_custom_LU_solver)
        .default_value(false)
        .help("Only for BDF methods:\n  -[false] to use the default linear solver in the Newton iteration O(n^3).\n  - [true] for a custom implementation with partial initial LU in O(n^2)");
    aseismicSlipSchema.add_value("custom_time_step_adapter", &SolverConfigSpecific::custom_time_step_adapter)
        .default_value(false)
        .help("Estimation of the new time step size:\n  -[false] to use the basic PETSc built-in adapter.\n  -[true] to use a custom, implementation-dependent adapter.\n Deverloper note: this parameter is likely to be changed to string to allow the choice between different custom adapter");
    aseismicSlipSchema.add_value("bdf_ksp_type", &SolverConfigSpecific::bdf_ksp_type)
        .default_value("preonly")
        .help("type of the ksp matrix-vector multiplication procedure for the Jacobian");
    aseismicSlipSchema.add_value("bdf_pc_type", &SolverConfigSpecific::bdf_pc_type)
        .default_value("lu")
        .help("type of the preconditioner for the Jacobian");

    std::optional<Config> cfg = readFromConfigurationFileAndCmdLine(schema, program, argc, argv);
    if (!cfg) {
        return -1;
    }

    PetscErrorCode ierr;
    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

    std::unique_ptr<GlobalSimplexMesh<DomainDimension>> globalMesh;
    if (cfg->mesh_file) {
        bool ok = false;
        GlobalSimplexMeshBuilder<DomainDimension> builder;
        if (rank == 0) {
            GMSHParser parser(&builder);
            ok = parser.parseFile(*cfg->mesh_file);
            if (!ok) {
                std::cerr << *cfg->mesh_file << std::endl << parser.getErrorMessage();
            }
        }
        MPI_Bcast(&ok, 1, MPI_CXX_BOOL, 0, PETSC_COMM_WORLD);
        if (ok) {
            globalMesh = builder.create(PETSC_COMM_WORLD);
        }
        if (procs > 1) {
            // ensure initial element distribution for metis
            globalMesh->repartitionByHash();
        }
    } else if (cfg->generate_mesh && cfg->resolution) {
        auto meshGen = cfg->generate_mesh->create(*cfg->resolution, PETSC_COMM_WORLD);
        globalMesh = meshGen.uniformMesh();
    }
    if (!globalMesh) {
        std::cerr
            << "You must either provide a valid mesh file or provide the mesh generation config "
               "(including the resolution parameter)."
            << std::endl;
        PetscFinalize();
        return -1;
    }
    globalMesh->repartition();
    auto mesh = globalMesh->getLocalMesh(1);

    solveSEASProblem(*mesh, *cfg);

    ierr = PetscFinalize();

    return ierr;
}
