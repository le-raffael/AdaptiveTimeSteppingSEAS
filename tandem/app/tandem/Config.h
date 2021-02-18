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

struct SolverConfigSpecific {
    std::string solution_size;
    std::string problem_formulation;
    std::string type;
    std::string rk_type;
    int bdf_order;
    double S_rtol;
    double S_atol;
    double V_rtol;
    double V_atol;
    double psi_rtol;
    double psi_atol;
};

struct SolverConfigGeneral {
    bool use_monitor;
    bool bdf_custom_error_evaluation;
    bool bdf_custom_Newton_iteration;
    std::string adapt_wnormtype;
    bool custom_time_step_adapter;
    std::string ksp_type;
    std::string pc_type;
    std::string pc_factor_mat_solver_type;
    std::optional<SolverConfigSpecific> solver_earthquake;
    std::optional<SolverConfigSpecific> solver_aseismicslip;
};



struct Config {
    std::optional<double> resolution;
    double final_time;
    std::optional<std::string> mesh_file;
    SeasScenarioConfig seas;
    DieterichRuinaAgeingConfig friction;
    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
    std::optional<OutputConfig> output;
    std::optional<SolverConfigGeneral> solver;
};

} // namespace tndm

#endif // CONFIG_20200825_H
