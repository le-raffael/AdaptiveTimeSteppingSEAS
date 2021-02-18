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
#include <petscdm.h>
#include <petsc/private/tsimpl.h>   // hack to directly access petsc stuff of the library

#include "tandem/RateAndStateBase.h"

#include <iostream>

#include <memory>

namespace tndm {

class PetscTimeSolver {
public:
    /**
     * set up the Petsc time solver 
     * @param timeop instance of the seas operator
     * @param ksp object to solve the linear system
     */
    template <typename TimeOp> PetscTimeSolver(TimeOp& timeop, KSP& ksp) {

        // get general configuration
        const auto& cfg = timeop.getGeneralSolverConfiguration();

        // start up PETSc
        CHKERRTHROW(TSCreate(timeop.comm(), &ts_));
        CHKERRTHROW(TSSetProblemType(ts_, TS_NONLINEAR));
 //       CHKERRTHROW(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP)); - think about how to add that later
 

        if (cfg){
            // read petsc options from file
            std::cout<<"Read PETSc options from configuration file. Eventual PETSc command line arguments will overwrite settings from the configuration file."<<std::endl;

            // get specific configuration files
            const auto& cfg_as = timeop.getEarthquakeSolverConfiguration();
            const auto& cfg_eq = timeop.getAseismicSlipSolverConfiguration();

            // read norm type
            TSAdapt adapt;
            CHKERRTHROW(TSGetAdapt(ts_, &adapt));
            if (cfg->adapt_wnormtype == "2") {ts_->adapt->wnormtype = NormType::NORM_2; }    // this is very hacky 
            else if (cfg->adapt_wnormtype == "infinity") {ts_->adapt->wnormtype = NormType::NORM_INFINITY; }
            else {std::cerr<<"Unknown norm! use \"2\" or \"infinity\""<<std::endl; }

            // set ksp type for the linear solver
            CHKERRTHROW(KSPSetType(ksp, cfg->ksp_type.c_str()));

            // set preconditioner options
            PC pc;
            CHKERRTHROW(KSPGetPC(ksp, &pc));
            CHKERRTHROW(PCSetType(pc, cfg->pc_type.c_str()));
            CHKERRTHROW(PCFactorSetMatSolverType(pc, cfg->pc_factor_mat_solver_type.c_str()));


            // Store the seas operator in the context if needed
            CHKERRTHROW(TSSetApplicationContext(ts_, &timeop));

            // apply changes at switch between aseismic slip and eaqrthquake phases
            CHKERRTHROW(TSSetPostEvaluate(ts_, updateAfterTimeStep<TimeOp>));

            // initialize everything in here for the time integration
            initializeSolver(timeop);

            // Overwrite settings by command line options
            CHKERRTHROW(TSSetFromOptions(ts_));
            if (cfg->custom_time_step_adapter) adapt->ops->choose = TSAdaptChoose_Custom;

        } else {
            // read petsc options from command line
            std::cout<<"Read PETSc options from command line"<<std::endl;
            state_ = std::make_unique<PetscBlockVector>(timeop.block_size(), timeop.numLocalElements(),
                                                        timeop.comm());
            timeop.initial_condition(*state_);

            CHKERRTHROW(TSSetProblemType(ts_, TS_NONLINEAR));
            CHKERRTHROW(TSSetSolution(ts_, state_->vec()));
            CHKERRTHROW(TSSetRHSFunction(ts_, nullptr, RHSFunctionCompactODE<TimeOp>, &timeop));
            CHKERRTHROW(TSSetRHSJacobian(ts_, timeop.getJacobianCompactODE(), timeop.getJacobianCompactODE(), RHSJacobianCompactODE<TimeOp>, &timeop));
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
//        CHKERRTHROW(TSSolve(ts_, state_->vec())); - needed for TS_EXACTFINALTIME_MATCHSTEP
        CHKERRTHROW(TSSolve(ts_, nullptr));
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
     * get the TS instance SNESSHELL
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
     * evaluate the rhs of the compact ODE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains V and psi on all fault nodes in block format)
     * @param F first time derivative of u (solution vector to be written to)
     * @param ctx pointer to Seas operator instanceTimeOp*
     */
    template <typename TimeOp>
    static PetscErrorCode RHSFunctionCompactODE(TS ts, PetscReal t, Vec u, Vec F, void* ctx) {
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        auto u_view = PetscBlockVectorView(u);
        auto F_view = PetscBlockVectorView(F);
        self->rhsCompactODE(t, u_view, F_view, true);
        return 0;
    }

    /**
     * evaluate the rhs of the extended ODE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains V and psi on all fault nodes in block format)
     * @param F first time derivative of u (solution vector to be written to)
     * @param ctx pointer to Seas operator instanceTimeOp*
     */
    template <typename TimeOp>
    static PetscErrorCode RHSFunctionExtendedODE(TS ts, PetscReal t, Vec u, Vec F, void* ctx) {
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        auto u_view = PetscBlockVectorView(u);
        auto F_view = PetscBlockVectorView(F);
        self->rhsExtendedODE(t, u_view, F_view, true);
        return 0;
    }

    /**
     * evaluate the rhs to 0 for DAEs
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains V and psi on all fault nodes in block format)
     * @param F first time derivative of u (solution vector to be written to)
     * @param ctx pointer to Seas operator instanceTimeOp*
     */
    static PetscErrorCode zeroRHS(TS ts, PetscReal t, Vec u, Vec F, void* ctx) {
        CHKERRTHROW(VecZeroEntries(F));
        return 0;
    }


    /**
     * evaluate the Jacobian of the compact ODE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains V and psi on all fault nodes in block format)
     * @param A Jacobian matrix
     * @param B matrix for the preconditioner (take the same as A)
     * @param ctx pointer to Seas operator instance
     */
    template <typename TimeOp>
    static PetscErrorCode RHSJacobianCompactODE(TS ts, PetscReal t, Vec u, Mat A, Mat B, void *ctx){
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        self->updateJacobianCompactODE(A);
        return 0;
    }

    /**
     * evaluate the Jacobian of the extended ODE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains V and psi on all fault nodes in block format)
     * @param A Jacobian matrix
     * @param B matrix for the preconditioner (take the same as A)
     * @param ctx pointer to Seas operator instance
     */
    template <typename TimeOp>
    static PetscErrorCode RHSJacobianExtendedODE(TS ts, PetscReal t, Vec u, Mat A, Mat B, void *ctx){
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        self->updateJacobianExtendedODE(A);
        return 0;
    }

    /**
     * evaluate the lhs F() of the compact DAE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains S and psi on all fault nodes in block format)
     * @param u_t current state derivative vector (contains V and dot{psi} on all fault nodes in block format)
     * @param F first time derivative of u (solution vector to be written to)
     * @param ctx pointer to Seas operator instanceTimeOp*
     */
    template <typename TimeOp>
    static PetscErrorCode LHSFunctionCompactDAE(TS ts, PetscReal t, Vec u, Vec u_t, Vec F, void* ctx) {
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        auto u_view = PetscBlockVectorView(u);
        auto u_t_view = PetscBlockVectorView(u_t);
        auto F_view = PetscBlockVectorView(F);
        self->lhsCompactDAE(t, u_view, u_t_view, F_view);
        return 0;
    }
    /**
     * evaluate the lhs F() of the extended DAE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains S and psi on all fault nodes in block format)
     * @param u_t current state derivative vector (contains V and dot{psi} on all fault nodes in block format)
     * @param F first time derivative of u (solution vector to be written to)
     * @param ctx pointer to Seas operator instanceTimeOp*
     */
    template <typename TimeOp>
    static PetscErrorCode LHSFunctionExtendedDAE(TS ts, PetscReal t, Vec u, Vec u_t, Vec F, void* ctx) {
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        auto u_view = PetscBlockVectorView(u);
        auto u_t_view = PetscBlockVectorView(u_t);
        auto F_view = PetscBlockVectorView(F);
        self->lhsExtendedDAE(t, u_view, u_t_view, F_view);
        return 0;
    }

    /**
     * evaluate the Jacobian of the compact DAE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains S and psi on all fault nodes in block format)
     * @param u_t current state derivative vector (contains V and dot{psi} on all fault nodes in block format)
     * @param sigma shift dx_dot/dx, depends on the numerical scheme
     * @param A Jacobian matrix
     * @param B matrix for the preconditioner (take the same as A)
     * @param ctx pointer to Seas operator instance
     */
    template <typename TimeOp>
    static PetscErrorCode LHSJacobianCompactDAE(TS ts, PetscReal t, Vec u, Vec u_t, PetscReal sigma, Mat A, Mat B, void *ctx){
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        self->updateJacobianCompactDAE(sigma,A);
        return 0;
    }
    /**
     * evaluate the Jacobian of the extended DAE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains S and psi on all fault nodes in block format)
     * @param u_t current state derivative vector (contains V and dot{psi} on all fault nodes in block format)
     * @param sigma shift dx_dot/dx, depends on the numerical scheme
     * @param A Jacobian matrix
     * @param B matrix for the preconditioner (take the same as A)
     * @param ctx pointer to Seas operator instance
     */
    template <typename TimeOp>
    static PetscErrorCode LHSJacobianExtendedDAE(TS ts, PetscReal t, Vec u, Vec u_t, PetscReal sigma, Mat A, Mat B, void *ctx){
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        self->updateJacobianExtendedDAE(sigma,A);
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
        // SNES snes;
        // CHKERRTHROW(TSGetSNES(ts, &snes));
        // SNESConvergedReason reason;
        // CHKERRTHROW(SNESGetConvergedReason(snes, &reason));
        // switch(reason) {
        //     case 2:
        //     std::cout << "Converged because of absolute error"<<std::endl;
        //     break;
        //     case 3:
        //     std::cout << "Converged because of relative error"<<std::endl;
        //     break;
        //     case 4:
        //     std::cout << "Converged because of small Newton step"<<std::endl;
        //     break;
        //     case 5:
        //     std::cout << "Converged because of maximum iteration number reached"<<std::endl;
        //     break;
        //     default:
        //     std::cout << "divergence." << std::endl;
        // }
        return 0;
    }


    /**
     * Initialize the solver with respect to the given parameters
     * @param timeop instance of the seas operator
     */
    template <typename TimeOp> 
    void initializeSolver(TimeOp& timeop){
        auto& solverStruct = timeop.getSolverParameters();
        auto const& cfg = timeop.getGeneralSolverConfiguration();
        auto const& cfg_as = timeop.getAseismicSlipSolverConfiguration();
        auto const& cfg_eq = timeop.getEarthquakeSolverConfiguration();

        // initialize the compact and extended solution vectors
        timeop.setExtendedFormulation(true);
        int bs_extended = timeop.block_size();
        timeop.setExtendedFormulation(false);
        int bs_compact = timeop.block_size();

        state_extended_ = std::make_shared<PetscBlockVector>(bs_extended, 
                                            timeop.numLocalElements(), timeop.comm());
        state_compact_ = std::make_shared<PetscBlockVector>(bs_compact, 
                                            timeop.numLocalElements(), timeop.comm());

        // apply the intitial condition in S and psi
        timeop.initial_condition(*state_compact_);

        // initialize Jacobians etc...
        if ((cfg_eq->type == "bdf") || (cfg_as->type == "bdf")) timeop.initialize_Jacobian(bs_compact, bs_extended);
        if (((cfg_eq->solution_size == "extended") && (cfg_eq->problem_formulation == "ode")) ||
            ((cfg_as->solution_size == "extended") && (cfg_as->problem_formulation == "ode"))){
                timeop.initialize_Jacobian(bs_compact, bs_extended);
                timeop.initialize_secondOrderDerivative();
            }


        // copy settings to solver struct
        solverStruct.customErrorFct = &evaluateEmbeddedMethodBDF;
        solverStruct.customNewtonFct = &solveNewton;
        solverStruct.checkAS = true;
        solverStruct.state_compact = state_compact_;
        solverStruct.state_extended = state_extended_;

        // initialize PETSc solver, set tolerances and general solver parameters (order, SNES, etc...)
        switchBetweenASandEQ(ts_, timeop, false,   // false to enter the aseismic slip at the beginning
                                          true);   // true  because it is the initial call

        // if a compact DAE is solved, the first step should have stol=0 in SNES (I don't know why)
        if ((cfg_as->solution_size == "compact") && (cfg_as->problem_formulation == "dae")) {
            SNES snes;
            CHKERRTHROW(TSGetSNES(ts_, &snes));
            double atol = 1e-50;    // absolute tolerance (default = 1e-50) 
            double rtol = 1e-8;     // relative tolerance (default = 1e-8)  
            double stol = 0;        // relative tolerance (default = 1e-8)  
            int maxit = 50;         // maximum number of iteration (default = 50)    
            int maxf = -1;          // maximum number of function evaluations (default = 1000)  
            CHKERRTHROW(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
            CHKERRTHROW(SNESSetFromOptions(snes));
        }
    }

    /**
     * Sets the absolute and relative tolerances 
     * @param ts the TS context
     * @param timeop instance of the seas operator
     * @param enterEQphase direction of the switch: true for as->eq, false for eq->as
     * @param initialCall only true to initialize the system at the very beginning
     */
    template <typename TimeOp> 
    static void switchBetweenASandEQ(TS ts, TimeOp& timeop, bool enterEQphase, bool initialCall){
        auto& solverStruct = timeop.getSolverParameters();
        auto const& cfg = timeop.getGeneralSolverConfiguration();
        auto const& cfg_as = timeop.getAseismicSlipSolverConfiguration();
        auto const& cfg_eq = timeop.getEarthquakeSolverConfiguration();

        std::string type;
        std::string rk_type;
        int bdf_order;

        // change formulation if needed
        if (initialCall || (cfg_as->solution_size != cfg_eq->solution_size) 
                        || (cfg_as->problem_formulation != cfg_as->problem_formulation)) {
            double time;
            CHKERRTHROW(TSGetTime(ts, &time));
            if (enterEQphase) {
                changeFormulation(ts, time, timeop, cfg_as, cfg_eq, initialCall);
            } else {
                changeFormulation(ts, time, timeop, cfg_eq, cfg_as, initialCall);
            }
        }

        // change tolerances and fetch solver parameters
        if (enterEQphase) {
            setTolerancesVector(ts, timeop, cfg_eq->solution_size, cfg_eq->S_rtol, cfg_eq->S_atol,   
                                 cfg_eq->psi_rtol, cfg_eq->psi_atol, cfg_eq->V_rtol, cfg_eq->V_atol);
            type = cfg_eq->type;
            rk_type = cfg_eq->rk_type;
            bdf_order = cfg_eq->bdf_order;

        } else {
            setTolerancesVector(ts, timeop, cfg_as->solution_size, cfg_as->S_rtol, cfg_as->S_atol,   
                                 cfg_as->psi_rtol, cfg_as->psi_atol, cfg_as->V_rtol, cfg_as->V_atol);
            type = cfg_as->type;
            rk_type = cfg_as->rk_type;
            bdf_order = cfg_as->bdf_order;
        }
        // integrator type 
        CHKERRTHROW(TSSetType(ts, type.c_str()));

        // rk settings
        if (type == "rk") CHKERRTHROW(TSRKSetType(ts, rk_type.c_str()));

        // bdf settings
        if (type == "bdf") setBDFParameters(ts, bdf_order, cfg, solverStruct);
    }

    /**
     * Set up the BDF scheme
     * @param ts TS instance
     * @param order order of the BDF scheme
     * @param cfg general configuration to see whether use manual implmentations or not
     * @param solverStruct contains the pointer to the manual implementations
     */
    template<typename T>
    static void setBDFParameters(TS ts, int order, 
        const std::optional<tndm::SolverConfigGeneral>& cfg, T& solverStruct){
            CHKERRTHROW(TSBDFSetOrder(ts, order));

            // set nonlinear solver settings
            SNES snes;
            KSP snes_ksp;
            PC snes_pc;
            
            CHKERRTHROW(TSGetSNES(ts, &snes));

            if (cfg->bdf_custom_error_evaluation)
                ts->ops->evaluatewlte = *solverStruct.customErrorFct;     // manual LTE evaluation

            if (cfg->bdf_custom_Newton_iteration) {                       // manual Newton iteration
                CHKERRTHROW(SNESSetType(snes, SNESSHELL));
                CHKERRTHROW(SNESShellSetContext(snes, &ts));
                CHKERRTHROW(SNESShellSetSolve(snes, *solverStruct.customNewtonFct));
            }

            CHKERRTHROW(SNESGetKSP(snes, &snes_ksp));
            CHKERRTHROW(KSPGetPC(snes_ksp, &snes_pc));
            CHKERRTHROW(KSPSetType(snes_ksp, cfg->ksp_type.c_str()));
            CHKERRTHROW(PCSetType(snes_pc, cfg->pc_type.c_str()));
            CHKERRTHROW(TSSetMaxSNESFailures(ts, -1));

            double atol = 1e-50;    // absolute tolerance (default = 1e-50) 
            double rtol = 1e-8;     // relative tolerance (default = 1e-8)  
            double stol = 1e-8;     // relative tolerance (default = 1e-8)  
            int maxit = 50;         // maximum number of iteration (default = 50)    
            int maxf = -1;          // maximum number of function evaluations (default = 1000)  
            CHKERRTHROW(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
            CHKERRTHROW(SNESSetFromOptions(snes));

    }


    /**
     * Change formulation of the problem. e.g. from extended ODE -> compact DAE
     * @param ts TS instance
     * @param time current simulation time
     * @param timeop instance of the SEAS operator
     * @param cfg_prev specific configuration of the previous section
     * @param cfg_next specific configuration of the next section
     * @param initialCall only true to initialize the system at the very beginning
     */
    template <typename TimeOp>
    static void changeFormulation(TS ts,double time, TimeOp& timeop, const std::optional<tndm::SolverConfigSpecific>& cfg_prev, const std::optional<tndm::SolverConfigSpecific>& cfg_next, bool initialCall){
        auto& solverStruct = timeop.getSolverParameters();
        if (cfg_next->solution_size == "compact") {
            if (initialCall || (cfg_prev->solution_size == "extended")) {
                // change the size of the solution vector
                timeop.setExtendedFormulation(false);
                if (!initialCall) timeop.makeSystemSmall(*solverStruct.state_compact, *solverStruct.state_extended);
                CHKERRTHROW(TSSetSolution(ts, solverStruct.state_compact->vec()));
            }
            if (cfg_next->problem_formulation == "ode"){
                CHKERRTHROW(TSSetEquationType(ts, TS_EQ_EXPLICIT));
                CHKERRTHROW(TSSetRHSFunction(ts, nullptr, RHSFunctionCompactODE<TimeOp>, &timeop));
                if (cfg_next->type == "bdf")
                    CHKERRTHROW(TSSetRHSJacobian(ts, timeop.getJacobianCompactODE(), timeop.getJacobianCompactODE(), RHSJacobianCompactODE<TimeOp>, &timeop));

            } else if (cfg_next->problem_formulation == "dae"){
                CHKERRTHROW(TSSetEquationType(ts, TS_EQ_IMPLICIT));
                CHKERRTHROW(TSSetIFunction(ts, nullptr, LHSFunctionCompactDAE<TimeOp>, &timeop));
                CHKERRTHROW(TSSetIJacobian(ts, timeop.getJacobianCompactDAE(), timeop.getJacobianCompactDAE(), LHSJacobianCompactDAE<TimeOp>, &timeop));
            }
        } else if (cfg_next->solution_size == "extended") {
            if (initialCall || (cfg_prev->solution_size == "compact")) {
                // change the size of the solution vector
                timeop.setExtendedFormulation(true);
                timeop.makeSystemBig(time, *solverStruct.state_compact, *solverStruct.state_extended);
                CHKERRTHROW(TSSetSolution(ts, solverStruct.state_extended->vec()));
            }
            if (cfg_next->problem_formulation == "ode"){
                CHKERRTHROW(TSSetEquationType(ts, TS_EQ_EXPLICIT));
                CHKERRTHROW(TSSetRHSFunction(ts, nullptr, RHSFunctionExtendedODE<TimeOp>, &timeop));
                if (cfg_next->type == "bdf")  
                    CHKERRTHROW(TSSetRHSJacobian(ts, timeop.getJacobianExtendedODE(), timeop.getJacobianExtendedODE(), RHSJacobianExtendedODE<TimeOp>, &timeop));

            } else if (cfg_next->problem_formulation == "dae"){
                CHKERRTHROW(TSSetEquationType(ts, TS_EQ_IMPLICIT));
                CHKERRTHROW(TSSetIFunction(ts, nullptr, LHSFunctionExtendedDAE<TimeOp>, &timeop));
                CHKERRTHROW(TSSetIJacobian(ts, timeop.getJacobianExtendedDAE(), timeop.getJacobianExtendedDAE(), LHSJacobianExtendedDAE<TimeOp>, &timeop));
            }                    
        }
        CHKERRTHROW(TSSetStepNumber(ts,0));
        CHKERRTHROW(TSSolve(ts,nullptr));
    }


    /**
     * Sets the absolute and relative tolerances for the formulation
     * @param ts the TS context
     * @param timeop instance of the seas operator
     * @param solution_size can be compact or extended
     * @param S_rtol relative error tolerance of the slip 
     * @param S_atol absolute error tolerance of the slip 
     * @param psi_rtol relative error tolerance of the state variable 
     * @param psi_atol absolute error tolerance of the state varaible 
     * @param V_rtol relative error tolerance of the slip rate
     * @param V_atol absolute error tolerance of the slip rate
     */
    template <typename TimeOp> 
    static void setTolerancesVector(TS ts, TimeOp& timeop, std::string solution_size, double S_rtol, double S_atol, double psi_rtol, double psi_atol, double V_rtol, double V_atol){
        // get domain characteristics
        size_t nbf = timeop.lop().space().numBasisFunctions();
        size_t bs = timeop.block_size();
        size_t numElem = timeop.numLocalElements();
        size_t PsiIndex = RateAndStateBase::TangentialComponents      * nbf;
        size_t VIndex =  (RateAndStateBase::TangentialComponents + 1) * nbf;
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
                a(i) = S_atol;
                r(i) = S_rtol;
            }
            // set state variable tolerances
            for ( int i = PsiIndex; i < PsiIndex + nbf; i++){
                a(i) = psi_atol;
                r(i) = psi_rtol;
            }
            // set slip rate tolerances
            if (solution_size == "extended") {
                for ( int i = VIndex; i < VIndex + PsiIndex; i++){
                    a(i) = V_atol;
                    r(i) = V_rtol;
                }
            }
        }
        atol.end_access(a_access);
        rtol.end_access(r_access);

        // write tolerances to PETSc solver
        CHKERRTHROW(TSSetTolerances(ts, 0, atol.vec(), 0, rtol.vec()));
    }



    /**
     * This function is executed at the end of each succesfull timestep. Used to update tolerances
     * @param ts the TS context
     */
    template <typename TimeOp>
    static PetscErrorCode updateAfterTimeStep(TS ts){
        TimeOp* seasop;
        CHKERRTHROW(TSGetApplicationContext(ts, &seasop));

        auto& solverStruct = seasop->getSolverParameters();

        if (solverStruct.checkAS && (seasop->VMax() > seasop->getV0())){          // change from as -> eq
            switchBetweenASandEQ(ts, *seasop, true, false);
            solverStruct.checkAS = false;
            std::cout << "Enter earthquake phase" << std::endl;
        } else if (!solverStruct.checkAS && (seasop->VMax() < seasop->getV0())){  // change from eq -> as
            switchBetweenASandEQ(ts, *seasop, false, false);
            solverStruct.checkAS = true;
            std::cout << "Exit earthquake phase" << std::endl;
        }

        // reset stol=1e-8 of SNES afther the first time step (suppose that you are still in the aseismic slip at the first time step)
        auto const& cfg = seasop->getAseismicSlipSolverConfiguration();
        if ((cfg->solution_size == "compact") && (cfg->problem_formulation == "dae")) {
            int n;
            CHKERRTHROW(TSGetStepNumber(ts,&n));
            if (n == 1) {
                SNES snes;
                CHKERRTHROW(TSGetSNES(ts, &snes));
                double atol = 1e-50;    // absolute tolerance (default = 1e-50) 
                double rtol = 1e-8;     // relative tolerance (default = 1e-8)  
                double stol = 1e-8;     // relative tolerance (default = 1e-8)  
                int maxit = 50;         // maximum number of iteration (default = 50)    
                int maxf = -1;          // maximum number of function evaluations (default = 1000)  
                CHKERRTHROW(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
                CHKERRTHROW(SNESSetFromOptions(snes));
            }
        }
        // Vec Xx;   Only for manual error evaluation
        // double* xx;
        // CHKERRTHROW(TSGetSolution(ts,&Xx));        
        // VecGetArray(Xx, &xx); // for some reason needed if BDF order > 4
        // VecRestoreArray(Xx, &xx); // for some reason needed if BDF order > 4

        return 0;
    }

    /**
     * Own implementation of the Newton algorithm (not used, only for testing purposes)
     * @param snes solving context
     * @param xout final solution vector 
     */
     static PetscErrorCode solveNewton(SNES snes, Vec xout){
        // get TS context
        void* ctx;
        CHKERRTHROW(SNESShellGetContext(snes, &ctx));
        TS* ts_void = reinterpret_cast<TS*>(ctx);
        TS ts = *ts_void;

        KSP ksp;
        CHKERRTHROW(SNESGetKSP(snes, &ksp));        


        Vec x;         // solution vector
        Vec f;         // residual
        Mat J;         // Jacobi matrix
        Mat J_pat;         // Jacobi matrix
        double res;    // norm of the residual

        Vec ratio;     // J(f(u))^{-1}f(u)


        CHKERRTHROW(TSGetSolution(ts, &x));     
        CHKERRTHROW(TSGetRHSJacobian(ts, &J_pat, nullptr, nullptr, nullptr));    // just to get the size of the matrices

        CHKERRTHROW(VecDuplicate(x, &f));
        CHKERRTHROW(VecDuplicate(x, &ratio));
        CHKERRTHROW(MatDuplicate(J_pat, MAT_DO_NOT_COPY_VALUES, &J));
        
        CHKERRTHROW(VecCopy(x, xout));      // solution from last time step as initial guess
        

        CHKERRTHROW(SNESTSFormFunction(snes, xout, f, ts));    //evaluate RHS of the ODE

        CHKERRTHROW(VecNorm(f, NORM_INFINITY, &res));

        int n = 0;
        while (res > 1e-12) {
            n++;
            CHKERRTHROW(SNESTSFormJacobian(snes, xout, J, J, ts));  // get the Jacobian
            CHKERRTHROW(KSPSetOperators(ksp, J, J));
            CHKERRTHROW(KSPSolve(ksp, f, ratio));     // solve the Jacobian system
            CHKERRTHROW(VecAXPY(xout, -1, ratio));     // update the solution vector
            CHKERRTHROW(SNESTSFormFunction(snes, xout, f, ts));    // evaluate RHS of the ODE
            CHKERRTHROW(VecNorm(f, NORM_INFINITY, &res));       // and its norm
            double *y;
            VecGetArray(f, &y);

            std::cout << "residual: " << std::endl;
            for (int i = 0; i < 30; ++i){
                std::cout << y[i] << ", ";
            }
            std::cout << std::endl;
            std::cout << std::endl;

            VecRestoreArray(f, &y);

        }

        CHKERRTHROW(VecDestroy(&f));
        CHKERRTHROW(VecDestroy(&ratio));
        CHKERRTHROW(MatDestroy(&J));
        

        return 0;
    }


    /**
     * Evaluate the local truncation error with the embedded BDF method 
     * @param ts main ts object [in]
     * @param NormType norm to be used to calculate the lte [in]
     * @param order order of the error evaluation [out]
     * @param wlte weighted local truncation error [out]
     */
    static PetscErrorCode evaluateEmbeddedMethodBDF(TS ts,NormType wnormtype,PetscInt *order,PetscReal *wlte){
        // presolve processing
        TS_BDF          *bdf = (TS_BDF*)ts->data;
        PetscReal      wltea,wlter;
        Vec            X = bdf->work[0], Y = bdf->vec_lte, Z;

        CHKERRTHROW(TSBDFGetOrder(ts, order));
        CHKERRTHROW(TSBDFSetOrder(ts, *order-1));

        PetscInt       n = PetscMax(bdf->k,1) + 1;
        Vec            V,V0;
        Vec            vecs[7];
        PetscScalar    alpha[7];

        V = bdf->vec_dot;
        V0 = bdf->vec_wrk;

        // get Lagrangian polynomials from time steps
        PetscInt  k,j,i;
        for (k=0; k<n; k++)
            for (alpha[k]=0, j=0; j<n; j++)
            if (j != k) {
                PetscReal L = 1/(bdf->time[k] - bdf->time[j]);
                for (i=0; i<n; i++)
                if (i != j && i != k)
                    L *= (bdf->time[0] - bdf->time[i])/(bdf->time[k] - bdf->time[i]);
                alpha[k] += L;
            }

        for (i=1; i<n; i++) {
                vecs[i] = bdf->transientvar ? bdf->tvwork[i] : bdf->work[i];
        }

        VecZeroEntries(V0);
        VecMAXPY(V0,n-1,alpha+1,vecs+1);
        bdf->shift = PetscRealPart(alpha[0]);

        V = nullptr;
        V0 = nullptr;

        // solve nonlinear system
        CHKERRTHROW(VecDuplicate(X, &Z));    
        CHKERRTHROW(VecCopy(X, Z));     // initial guess is previous time step
//        CHKERRTHROW(VecCopy(bdf->work[1], Z));     // initial guess is previous time step
        SNESSolve(ts->snes,nullptr,Z);

        // calculate the error between both solutions
        TSErrorWeightedNorm(ts,X,Z,wnormtype,wlte,&wltea,&wlter);

        std::cout << "local truncation error: " << *wlte << std::endl;
        // double *x, *y;
        // VecGetArray(bdf->work[0], &x);
        // VecGetArray(bdf->work[1], &y);

        // std::cout << "bdf->work[0]: " << std::endl;
        // for (i = 0; i < 30; ++i){
        //      std::cout << x[i]<< ", ";
        //  }
        // std::cout << std::endl;
        // std::cout << "bdf->work[1]: " << std::endl;
        // for (i = 0; i < 30; ++i){
        //      std::cout << y[i] << ", ";
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;

        //  VecRestoreArray(bdf->work[0], &x);
        //  VecRestoreArray(bdf->work[1], &y);

        // restore BDF order
        CHKERRTHROW(TSBDFSetOrder(ts, *order));

        CHKERRTHROW(VecAXPY(Z,-1,X));
        CHKERRTHROW(VecCopy(Z, Y));
        CHKERRTHROW(VecDestroy(&Z));     
        return 0;
    }
        

    /**
     * Custom time step adapter
     * @param adapt time adapter context
     * @param ts general time-stepping context
     * @param h previous time step
     * @param next_sc ?
     * @param next_h next time step
     * @param accept whether time step is accepted or not
     * @param wlte local truncation error
     * @param wltea absolute LTE
     * @param wlter relative LTE
     */
    static PetscErrorCode TSAdaptChoose_Custom(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter) {
        Vec            Y;
        DM             dm;
        PetscInt       order  = PETSC_DECIDE;
        PetscReal      enorm  = -1;
        PetscReal      enorma,enormr;
        PetscReal      safety = adapt->safety;
        PetscReal      hfac_lte,h_lte;

        *next_sc = 0;   /* Reuse the same order scheme */
        *wltea   = -1;  /* Weighted absolute local truncation error is not used */
        *wlter   = -1;  /* Weighted relative local truncation error is not used */

        if (ts->ops->evaluatewlte) {
            TSEvaluateWLTE(ts,adapt->wnormtype,&order,&enorm);
            std::cout << "previous time step size: " << h << " with error norm: " << enorm << std::endl;
            enorm = 0.1;
            if (enorm >= 0 && order < 1) SETERRQ1(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Computed error order %D must be positive",order);
        } else if (ts->ops->evaluatestep) {
            if (adapt->candidates.n < 1) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"No candidate has been registered");
            if (!adapt->candidates.inuse_set) SETERRQ1(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"The current in-use scheme is not among the %D candidates",adapt->candidates.n);
            order = adapt->candidates.order[0];
            TSGetDM(ts,&dm);
            DMGetGlobalVector(dm,&Y);
            TSEvaluateStep(ts,order-1,Y,NULL);
            TSErrorWeightedNorm(ts,ts->vec_sol,Y,adapt->wnormtype,&enorm,&enorma,&enormr);
            DMRestoreGlobalVector(dm,&Y);
        }

        if (enorm < 0) {
            *accept  = PETSC_TRUE;
            *next_h  = h;            /* Reuse the old step */
            *wlte    = -1;           /* Weighted local truncation error was not evaluated */
            return(0);
        }

        /* Determine whether the step is accepted of rejected */
        if (enorm > 1) {
            if (!*accept) safety *= adapt->reject_safety; /* The last attempt also failed, shorten more aggressively */
            if (h < (1 + PETSC_SQRT_MACHINE_EPSILON)*adapt->dt_min) {
            PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting because step size %g is at minimum\n",(double)enorm,(double)h);
            *accept = PETSC_TRUE;
            } else if (adapt->always_accept) {
            PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting step of size %g because always_accept is set\n",(double)enorm,(double)h);
            *accept = PETSC_TRUE;
            } else {
            PetscInfo2(adapt,"Estimated scaled local truncation error %g, rejecting step of size %g\n",(double)enorm,(double)h);
            *accept = PETSC_FALSE;
            }
        } else {
            PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting step of size %g\n",(double)enorm,(double)h);
            *accept = PETSC_TRUE;
        }

        /* The optimal new step based purely on local truncation error for this step. */
        if (enorm > 0)
            hfac_lte = safety * PetscPowReal(enorm,((PetscReal)-1)/order);
        else
            hfac_lte = safety * PETSC_INFINITY;
        if (adapt->timestepjustdecreased){
            hfac_lte = PetscMin(hfac_lte,1.0);
            adapt->timestepjustdecreased--;
        }
        h_lte = h * PetscClipInterval(hfac_lte,adapt->clip[0],adapt->clip[1]);

//        *next_h = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
        *next_h = 1.1 * h;
        *wlte   = enorm;
        return(0);
    }

    std::shared_ptr<PetscBlockVector> state_;
    std::shared_ptr<PetscBlockVector> state_compact_;
    std::shared_ptr<PetscBlockVector> state_extended_;
    TS ts_ = nullptr;

    /**
     * Struct of the internal BDF scheme in Petsc 
     */
    typedef struct {
        PetscInt  k,n;
        PetscReal time[6+2];
        Vec       work[6+2];
        Vec       tvwork[6+2];
        PetscReal shift;
        Vec       vec_dot;            /* Xdot when !transientvar, else Cdot where C(X) is the transient variable. */
        Vec       vec_wrk;
        Vec       vec_lte;

        PetscBool    transientvar;
        PetscInt     order;
        TSStepStatus status;
    } TS_BDF;
};

} // namespace tndm

#endif // PETSCTS_20201001_H
