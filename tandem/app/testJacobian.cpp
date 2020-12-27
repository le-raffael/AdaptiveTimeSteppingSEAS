#include "common/CmdLine.h"
#include "common/MeshConfig.h"
#include "config.h"
#include "tandem/AdaptiveOutputStrategy.h"
#include "tandem/Config.h"
#include "tandem/FrictionConfig.h"
#include "tandem/testJacobianScript.h"
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

    PetscInitialize(&pArgc, &pArgv, nullptr, nullptr);

    argparse::ArgumentParser program("tandem");
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

    std::optional<Config> cfg = readFromConfigurationFileAndCmdLine(schema, program, argc, argv);
    if (!cfg) {
        return -1;
    }

    int rank, procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

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
        MPI_Bcast(&ok, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        if (ok) {
            globalMesh = builder.create(MPI_COMM_WORLD);
        }
        if (procs > 1) {
            // ensure initial element distribution for metis
            globalMesh->repartitionByHash();
        }
    } else if (cfg->generate_mesh && cfg->resolution) {
        auto meshGen = cfg->generate_mesh->create(*cfg->resolution, MPI_COMM_WORLD);
        globalMesh = meshGen.uniformMesh();
    }
    if (!globalMesh) {
        std::cerr
            << "You must either provide a valid mesh file or provide the mesh generation config "
               "(including the resolution parameter)."
            << std::endl;
        return -1;
    }
    globalMesh->repartition();
    auto mesh = globalMesh->getLocalMesh(1);

    testJacobianScript(*mesh, *cfg);

    return 0;
}