#!/bin/bash
cd ../build
make
./app/tandem ../examples/tandem/2d/bp1_sym.toml --petsc -ts_monitor -ts_type rk -ts_rk_type 5dp -ts_rtol 1e-7 -ts_atol 1e-7 -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -ts_adapt_wnormtype infinity
mv timeAnalysis.csv timeAnalysis_RKDP5.csv
./app/tandem ../examples/tandem/2d/bp1_sym.toml --petsc -ts_monitor -ts_type rk -ts_rk_type 3bs -ts_rtol 1e-7 -ts_atol 1e-7 -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -ts_adapt_wnormtype infinity
mv timeAnalysis.csv timeAnalysis_RKBS3.csv
cd ../scriptsResults


