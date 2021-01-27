#ifndef CONFIG_20200825_H
#define CONFIG_20200825_H

#include "common/MeshConfig.h"
#include "tandem/AdaptiveOutputStrategy.h"
#include "tandem/FrictionConfig.h"
#include "tandem/SeasScenario.h"

#include <optional>
#include <string>

namespace tndm {

struct OutputConfig {
    std::string prefix;
    double V_ref;
    double t_min;
    double t_max;
    AdaptiveOutputStrategy strategy;
};

struct SolverConfig {
    bool use_monitor;
    std::string ts_type;
    std::string ts_rk_type;
    std::string ts_adapt_wnormtype;    
    bool bdf_manual_error_evaluation;
    bool bdf_manual_Newton_iteration;
    double S_rtol;
    double S_atol;
    double psi_rtol;
    double psi_atol_eq;
    double psi_atol_as;
    int ts_bdf_order;
    std::string ksp_type;
    std::string pc_type;
    std::string pc_factor_mat_solver_type;
};

struct Config {
    std::optional<double> resolution;
    double final_time;
    std::optional<std::string> mesh_file;
    SeasScenarioConfig seas;
    DieterichRuinaAgeingConfig friction;
    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
    std::optional<OutputConfig> output;
    std::optional<SolverConfig> solver;
};

} // namespace tndm

#endif // CONFIG_20200825_H
