#ifndef PETSCTS_20201001_H
#define PETSCTS_20201001_H

#include "common/PetscBlockVector.h"
#include "common/PetscUtil.h"

#include "tandem/Config.h"

#include <petscsystypes.h>
#include <petscts.h>
#include <petscvec.h>
#include <petscksp.h>
#include "petscpc.h" 
#include <petsc/private/tsimpl.h>   // hack to directly access petsc stuff of the library

#include <iostream>

#include <memory>

namespace tndm {

class PetscTimeSolver {
public:
    /**
     * set up the Petsc time solver 
     * @param timeop instance of the seas operator
     * @param cfg solver configuration read from .toml file
     * @param ksp object to solve the linear system
     */
    template <typename TimeOp> PetscTimeSolver(TimeOp& timeop, const std::optional<tndm::SolverConfig>& cfg, KSP& ksp) {
        state_ = std::make_unique<PetscBlockVector>(timeop.block_size(), timeop.numLocalElements(),
                                                    timeop.comm());
        timeop.initial_condition(*state_);

        CHKERRTHROW(TSCreate(timeop.comm(), &ts_));
        CHKERRTHROW(TSSetProblemType(ts_, TS_NONLINEAR));
        CHKERRTHROW(TSSetSolution(ts_, state_->vec()));
        CHKERRTHROW(TSSetRHSFunction(ts_, nullptr, RHSFunction<TimeOp>, &timeop));
        CHKERRTHROW(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP));
        if (cfg){
            // read petsc options from file
            std::cout<<"Read PETSc options from configuration file. Eventual PETSc command line arguments will overwrite settings from the configuration file."<<std::endl;
            TSAdapt adapt;
            CHKERRTHROW(TSGetAdapt(ts_, &adapt));

            // integrator type
            CHKERRTHROW(TSSetType(ts_, cfg->ts_type.c_str()));

            // rk settings
            if (cfg->ts_type == "rk"){
                CHKERRTHROW(TSRKSetType(ts_, cfg->ts_rk_type.c_str()));
            }

            // norm type
            if (cfg->ts_adapt_wnormtype == "2") {ts_->adapt->wnormtype = NormType::NORM_2; }    // this is very hacky 
            else if (cfg->ts_adapt_wnormtype == "infinity") {ts_->adapt->wnormtype = NormType::NORM_INFINITY; }
            else {std::cerr<<"Unknown norm! use \"2\" or \"infinity\""<<std::endl; }

            // set tolerances
            setTolerancesVAndPSI(timeop, cfg);

            // set ksp type
            CHKERRTHROW(KSPSetType(ksp, cfg->ksp_type.c_str()));

            // set preconditioner options
            PC pc;
            CHKERRTHROW(KSPGetPC(ksp, &pc));
            CHKERRTHROW(PCSetType(pc, cfg->pc_type.c_str()));
            CHKERRTHROW(PCFactorSetMatSolverType(pc, cfg->pc_factor_mat_solver_type.c_str()));
            CHKERRTHROW(TSSetFromOptions(ts_));

        } else {
            // read petsc options from command line
            std::cout<<"Read PETSc options from command line"<<std::endl;
            CHKERRTHROW(TSSetFromOptions(ts_));
        }
    }

    /**
     * destructor
     */
    ~PetscTimeSolver() { TSDestroy(&ts_); }

    /**
     * Start the Petsc time solver. It will perform all time iterations
     * @param upcoming_time final simulation time
     */
    void solve(double upcoming_time) {
        CHKERRTHROW(TSSetMaxTime(ts_, upcoming_time));
        CHKERRTHROW(TSSolve(ts_, state_->vec()));
    }

    /**
     * return the solution vector of the ODE. It has the form: 
     * [1_V1, 1_V2, 1_V3, 1_psi1, 1_psi2, 1_psi3, 2_V1, 2_V2, 2_V3, 2_psi1, 2_psi2, 2_psi3, ...]
     * @return PetscBlockVector instance 
     */
    auto& state() { return *state_; }

    /**
     * return the solution vector of the ODE. It has the form: 
     * [1_V1, 1_V2, 1_V3, 1_psi1, 1_psi2, 1_psi3, 2_V1, 2_V2, 2_V3, 2_psi1, 2_psi2, 2_psi3, ...]
     * @return constant PetscBlockVector instance 
     */
    auto const& state() const { return *state_; }

    /**
     * get the TS instance 
     * used in the error evaluation in the monitor (SeasWriter.h)
     * @return TS object
     */
    auto getTS() {return ts_; }

    /**
     * Initialize the PETSC monitor to print metrics at certain iteration
     * @param monitor reference to the monitor (instance of SeasWriter.h)
     */
    template <class Monitor> void set_monitor(Monitor& monitor) {
        CHKERRTHROW(TSMonitorSet(ts_, &MonitorFunction<Monitor>, &monitor, nullptr));
    }

private:
    /**
     * evaluate the rhs of the ODE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains V and psi on all fault nodes in block format)
     * @param F first time derivative of u (solution vector to be written to)
     * @param ctx pointer to Seas operator instance
     */
    template <typename TimeOp>
    static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec u, Vec F, void* ctx) {
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        auto u_view = PetscBlockVectorView(u);
        auto F_view = PetscBlockVectorView(F);
        self->rhs(t, u_view, F_view);
        return 0;
    }

    /**
     * call the PETSC monitor at each iteration when printing is desired
     * @param ts TS object
     * @param steps current iteration step number
     * @param time current simulation time
     * @param u current state vector (contains V and psi on all fault nodes in block format)
     * @param ctx pointer to the Seas writer instance
     */
    template <class Monitor>
    static PetscErrorCode MonitorFunction(TS ts, PetscInt steps, PetscReal time, Vec u, void* ctx) {
        Monitor* self = reinterpret_cast<Monitor*>(ctx);
        auto u_view = PetscBlockVectorView(u);
        self->monitor(time, u_view);
        return 0;
    }

    /**
     * Sets the absolute and relative tolerances 
     * @param timeop instance of the seas operator
     * @param cfg solver configuration read from .toml file
     */
    template <typename TimeOp> 
    void setTolerancesVAndPSI(TimeOp& timeop, const std::optional<tndm::SolverConfig>& cfg){
        // get domain characteristics
        size_t nbf = timeop.lop().space().numBasisFunctions();
        size_t bs = timeop.block_size();
        size_t numElem = timeop.numLocalElements();
        size_t PsiIndex = RateAndStateBase::TangentialComponents * nbf;
        size_t totalSize = bs * numElem;

        // initialize tolerance vectors
        PetscBlockVector atol(bs, numElem, timeop.comm());   
        PetscBlockVector rtol(bs, numElem, timeop.comm()); 
        atol.set_zero();
        rtol.set_zero();

        // iterate through all fault elements
        auto a_access = atol.begin_access();
        auto r_access = rtol.begin_access();

        for (size_t faultNo = 0; faultNo < numElem; faultNo++){
            auto a = atol.get_block(a_access, faultNo);
            auto r = rtol.get_block(r_access, faultNo);
            // set velocity tolerances
            for ( int i = 0; i < PsiIndex; i++){
                a(i) = cfg->V_atol;
                r(i) = cfg->V_rtol;
            }
            // set state variable tolerances
            for ( int i = PsiIndex; i < PsiIndex + nbf; i++){
                a(i) = cfg->psi_atol;
                r(i) = cfg->psi_rtol;
            }
        }
        atol.end_access(a_access);
        rtol.end_access(r_access);

        // write tolerances to PETSc solver
        CHKERRTHROW(TSSetTolerances(ts_, 0, atol.vec(), 0, rtol.vec()));

    }

    std::unique_ptr<PetscBlockVector> state_;
    TS ts_ = nullptr;
};

} // namespace tndm

#endif // PETSCTS_20201001_H
