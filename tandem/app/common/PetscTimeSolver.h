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
#include <petsc/private/snesimpl.h>   

#include "tandem/RateAndStateBase.h"
#include "tandem/SeasWriter.h"

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

            // set ksp type for the linear solver
            CHKERRTHROW(KSPSetType(ksp, cfg->ksp_type.c_str()));

            // set preconditioner options
            PC pc;
            CHKERRTHROW(KSPGetPC(ksp, &pc));
            CHKERRTHROW(PCSetType(pc, cfg->pc_type.c_str()));
            CHKERRTHROW(PCFactorSetMatSolverType(pc, cfg->pc_factor_mat_solver_type.c_str()));

            // initialize everything in here for the time integration
            initializeSolver(timeop);

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
        timeop.setTSInstance(ts_);

        // initialize Jacobians etc...
        if ((cfg_eq->type == "bdf") || (cfg_as->type == "bdf")) timeop.initialize_Jacobian(bs_compact, bs_extended);
        if (((cfg_eq->solution_size == "extended") && (cfg_eq->problem_formulation == "ode")) ||
            ((cfg_as->solution_size == "extended") && (cfg_as->problem_formulation == "ode"))){
                timeop.initialize_Jacobian(bs_compact, bs_extended);
                timeop.initialize_secondOrderDerivative();
            }
          // TODO: decide whether to keep or throw out 
//        if (cfg_as->bdf_custom_LU_solver || cfg_eq->bdf_custom_LU_solver) timeop.calculateLU_dfDS();


        // copy settings to solver struct
        solverStruct.customErrorFct = &evaluateEmbeddedMethodBDF;
        solverStruct.customNewtonFct = &solveNewton<TimeOp>;
        solverStruct.checkAS = true;
        solverStruct.current_solver_cfg = cfg_as;
        solverStruct.state_compact = state_compact_;
        solverStruct.state_extended = state_extended_;

        // initialize PETSc solver, set tolerances and general solver parameters (order, SNES, etc...)
        switchBetweenASandEQ(ts_, timeop, false,   // false to enter the aseismic slip at the beginning
                                          true);   // true  because it is the initial call
    }


    /**
     * Performs actions before each stage
     * - If it is the first call of a compact DAE, calculate the initial derivative vector
     *   and approximate the initial guess with an explicit RK4
     * @param ts TS context
     * @param t current simulation time
     */
    template <typename TimeOp> 
    static PetscErrorCode functionPreStage(TS ts, double t) {
        TimeOp* seasop;
        CHKERRTHROW(TSGetApplicationContext(ts, &seasop));

        auto& solverStruct = seasop->getSolverParameters();
        auto const& cfg    = seasop->getGeneralSolverConfiguration();

        std::cout << "bonjour" << std::endl;

        PetscFunctionBegin;
        // initialize the SNES solver with an explicit Rk4
        if (solverStruct.needRegularizationCompactDAE) {
            solverStruct.needRegularizationCompactDAE = false;

           TS_BDF *bdf = (TS_BDF*)ts->data;

           double dt = bdf->time[0] - bdf->time[1];
            RK4(ts, bdf->time[1], dt, bdf->work[1], bdf->work[0], RHSFunctionCompactODE<TimeOp>, seasop);
        }
        PetscFunctionReturn(0);
    }

    /**
     * Performs actions after each stage
     * - in the 2nd order ODE formulation, if an implicit method is used, the slip is calculated separately after the Newton iteration
     * @param ts TS context
     * @param t current simulation time
     * @param k stage index
     * @param Y current solution vector
     */
    template <typename TimeOp> 
    static PetscErrorCode functionPostStage(TS ts, double t, int k, Vec* Y) {
        TimeOp* seasop;

        PetscFunctionBegin;
        CHKERRTHROW(TSGetApplicationContext(ts, &seasop));
        auto& solverStruct = seasop->getSolverParameters();

        if ((solverStruct.current_formulation == TimeOp::SECOND_ORDER_ODE) && solverStruct.useImplicitSolver){
            TS_BDF         *bdf = (TS_BDF*)ts->data;
            PetscInt       i,n = PetscMax(bdf->k,1) + 1;
            Vec            vecs[7];
            PetscScalar    alpha[7];
            PetscErrorCode ierr;
            double shift;

            LagrangeBasisDers(n,bdf->time[0],bdf->time,alpha);
            for (i=1; i<n; i++) {
                vecs[i] = bdf->transientvar ? bdf->tvwork[i] : bdf->work[i];
            }
            shift = PetscRealPart(alpha[0]);

            seasop->setSlipAfterNewtonIteration(*Y,n,vecs,alpha,shift);
        }

        PetscFunctionReturn(0);
    }

    /**
     * This function is executed at the end of each succesfull timestep. Used to update tolerances
     * @param ts the TS context
     */
    template <typename TimeOp>
    static PetscErrorCode functionPostEvaluate(TS ts){
        TimeOp* seasop;
        CHKERRTHROW(TSGetApplicationContext(ts, &seasop));

        auto& solverStruct = seasop->getSolverParameters();
        auto const& cfg_as = seasop->getAseismicSlipSolverConfiguration();
        auto const& cfg_eq = seasop->getEarthquakeSolverConfiguration();

        if (solverStruct.checkAS && (seasop->VMax() > seasop->getV0())){          // change from as -> eq
            std::cout << "Enter earthquake phase" << std::endl;
            switchBetweenASandEQ(ts, *seasop, true, false);
            solverStruct.checkAS = false;
            solverStruct.current_solver_cfg = cfg_eq;
        } else if (!solverStruct.checkAS && (seasop->VMax() < seasop->getV0())){  // change from eq -> as
            std::cout << "Exit earthquake phase" << std::endl;
            switchBetweenASandEQ(ts, *seasop, false, false);
            solverStruct.checkAS = true;
            solverStruct.current_solver_cfg = cfg_as;
        }



        if ((solverStruct.useImplicitSolver) && 
            (solverStruct.current_solver_cfg->bdf_order == 0)) {
            int next_order;
            int order;
            CHKERRTHROW(TSBDFGetOrder(ts, &order));
            adaptBDFOrder<TimeOp>(ts, next_order);
            std::cout << "current order: " << order << ", next order: "<< next_order <<std::endl; 
            CHKERRTHROW(TSBDFSetOrder(ts, next_order));
        }
//        reducedTimeEachStep(ts, solverStruct.time_eq);

        // VecGetArray(Xx, &xx); // for some reason needed if BDF order > 4
        // VecRestoreArray(Xx, &xx); // for some reason needed if BDF order > 4

        // Vec Xx;   Only for manual error evaluation
        // double* xx;
        // CHKERRTHROW(TSGetSolution(ts,&Xx));        
        // VecGetArray(Xx, &xx); // for some reason needed if BDF order > 4
        // VecRestoreArray(Xx, &xx); // for some reason needed if BDF order > 4

        return 0;
    }




    /**
     * Sets the absolute and relative tolerances 
     * @param ts the TS context
     * @param timeop instance of the seas opTS contexterator
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
        std::string solution_size;
        std::string problem_formulation;
        int bdf_order;

        double time;
        CHKERRTHROW(TSGetTime(ts, &time));

        // change formulation, tolerances and fetch solver parameters
        if (enterEQphase) {
    //        if (!initialCall) reducedTimeBeginEQ(ts, solverStruct.time_eq);
            changeFormulation(ts, time, timeop, cfg_as, cfg_eq, initialCall);
            setTolerancesVector(ts, timeop, cfg_eq->solution_size, cfg_eq->S_rtol, cfg_eq->S_atol,   
                                 cfg_eq->psi_rtol, cfg_eq->psi_atol, cfg_eq->V_rtol, cfg_eq->V_atol);
            type = cfg_eq->type;
            rk_type = cfg_eq->rk_type;
            bdf_order = cfg_eq->bdf_order;
            solution_size = cfg_eq->solution_size;
            problem_formulation = cfg_eq->problem_formulation;

        } else {
    //        if (!initialCall) reducedTimeEndEQ(ts, solverStruct.time_eq);
            changeFormulation(ts, time, timeop, cfg_eq, cfg_as, initialCall);
            setTolerancesVector(ts, timeop, cfg_as->solution_size, cfg_as->S_rtol, cfg_as->S_atol,   
                                 cfg_as->psi_rtol, cfg_as->psi_atol, cfg_as->V_rtol, cfg_as->V_atol);
            type = cfg_as->type;
            rk_type = cfg_as->rk_type;
            bdf_order = cfg_as->bdf_order;
            solution_size = cfg_as->solution_size;
            problem_formulation = cfg_as->problem_formulation;
        }

        CHKERRTHROW(TSSetUp(ts));

        CHKERRTHROW(TSAdaptSetClip(ts->adapt, 0.1, 10));

        // some output prints
        if (type == "rk") {
            std::cout << "Solve the problem as a " << solution_size <<  " " << 
            problem_formulation << " with the explicit Runge-Kutta method " << rk_type << std::endl;
        } else if (type == "bdf") {
            std::string cardinal = (bdf_order==1) ? "st" : (bdf_order==2) ? "nd" : (bdf_order==3) ? "rd" : "th" ;
            std::cout << "Solve the problem as a " << solution_size <<  " " << 
            problem_formulation << " with the implicit " << 
            bdf_order << cardinal << " order BDF method " << std::endl; 
        }
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
        auto& cfg = timeop.getGeneralSolverConfiguration();

        if (initialCall || (cfg_prev->solution_size != cfg_next->solution_size) ||
            (cfg_prev->problem_formulation != cfg_next->problem_formulation)) {

            DM dm;
            DMTS tsdm;
            CHKERRTHROW(TSGetDM(ts,&dm));
            CHKERRTHROW(DMGetDMTS(dm,&tsdm));

            CHKERRTHROW(TSReset(ts));

            if (cfg_next->solution_size == "compact") {
                if (initialCall || (cfg_prev->solution_size == "extended")) {
                    // change the size of the solution vector
                    timeop.setExtendedFormulation(false);
                    if (!initialCall) timeop.makeSystemSmall(*solverStruct.state_compact, *solverStruct.state_extended);
                }
                CHKERRTHROW(TSSetSolution(ts, solverStruct.state_compact->vec()));

                if (cfg_next->problem_formulation == "ode"){
                    solverStruct.current_formulation = TimeOp::FIRST_ORDER_ODE;
                    CHKERRTHROW(TSSetEquationType(ts, TS_EQ_EXPLICIT));
                    CHKERRTHROW(TSSetRHSFunction(ts, nullptr, RHSFunctionCompactODE<TimeOp>, &timeop));
                    if (cfg_next->type == "bdf") {
                        CHKERRTHROW(TSSetRHSJacobian(ts, timeop.getJacobianCompactODE(), timeop.getJacobianCompactODE(), RHSJacobianCompactODE<TimeOp>, &timeop));
                    }
                    tsdm->ops->ifunction = NULL;
                    tsdm->ops->ijacobian = NULL;

                } else if (cfg_next->problem_formulation == "dae"){
                    solverStruct.current_formulation = TimeOp::COMPACT_DAE;
                    solverStruct.needRegularizationCompactDAE = true;
                    CHKERRTHROW(TSSetEquationType(ts, TS_EQ_IMPLICIT));
                    CHKERRTHROW(TSSetPreStage(ts, functionPreStage<TimeOp>));
                    CHKERRTHROW(TSSetIFunction(ts, nullptr, LHSFunctionCompactDAE<TimeOp>, &timeop));
                    CHKERRTHROW(TSSetIJacobian(ts, timeop.getJacobianCompactDAE(), timeop.getJacobianCompactDAE(), LHSJacobianCompactDAE<TimeOp>, &timeop));
                    tsdm->ops->rhsfunction = NULL;
                    tsdm->ops->rhsjacobian = NULL;
                }
            } else if (cfg_next->solution_size == "extended") {
                if (initialCall || (cfg_prev->solution_size == "compact")) {
                    // change the size of the solution vector
                    timeop.setExtendedFormulation(true);
                    timeop.makeSystemBig(time, *solverStruct.state_compact, *solverStruct.state_extended);
                }
                CHKERRTHROW(TSSetSolution(ts, solverStruct.state_extended->vec()));
                if (cfg_next->problem_formulation == "ode"){
                    solverStruct.current_formulation = TimeOp::SECOND_ORDER_ODE;
                    CHKERRTHROW(TSSetEquationType(ts, TS_EQ_EXPLICIT));
                    CHKERRTHROW(TSSetRHSFunction(ts, nullptr, RHSFunctionExtendedODE<TimeOp>, &timeop));
                    if (cfg_next->type == "bdf") {
                        CHKERRTHROW(TSSetRHSJacobian(ts, timeop.getJacobianExtendedODE(), timeop.getJacobianExtendedODE(), RHSJacobianExtendedODE<TimeOp>, &timeop));
                        CHKERRTHROW(TSSetPostStage(ts, functionPostStage<TimeOp>));
                    }
                    tsdm->ops->ifunction = NULL;
                    tsdm->ops->ijacobian = NULL;

                } else if (cfg_next->problem_formulation == "dae"){
                    solverStruct.current_formulation = TimeOp::EXTENDED_DAE;
                    CHKERRTHROW(TSSetEquationType(ts, TS_EQ_IMPLICIT));
                    CHKERRTHROW(TSSetPreStage(ts, functionPreStage<TimeOp>));       // remove that again!!
                    CHKERRTHROW(TSSetIFunction(ts, nullptr, LHSFunctionExtendedDAE<TimeOp>, &timeop));
                    CHKERRTHROW(TSSetIJacobian(ts, timeop.getJacobianExtendedDAE(), timeop.getJacobianExtendedDAE(), LHSJacobianExtendedDAE<TimeOp>, &timeop));
                    tsdm->ops->rhsfunction = NULL;
                    tsdm->ops->rhsjacobian = NULL;
                }
            }

            // set adapter parameters (for dynamic time-stepping)
            CHKERRTHROW(DMClearGlobalVectors(dm));  // to remove old solution vectors
            TSAdapt adapt;
            CHKERRTHROW(TSGetAdapt(ts, &adapt));
            CHKERRTHROW(TSAdaptSetType(adapt, TSADAPTBASIC));
            if (cfg->adapt_wnormtype == "2") {ts->adapt->wnormtype = NormType::NORM_2; }    // this is very hacky 
            else if (cfg->adapt_wnormtype == "infinity") {ts->adapt->wnormtype = NormType::NORM_INFINITY; }
            else { std::cerr<<"Unknown norm! use \"2\" or \"infinity\""<<std::endl; }

            // TODO: activate again if an alternative timestep adapter has been implemented
//            adapt->ops->choose = (cfg_next->custom_time_step_adapter)?TSAdaptChoose_Custom:NULL;

            // Store the seas operator in the context if needed
            CHKERRTHROW(TSSetApplicationContext(ts, &timeop));

            // apply changes at switch between aseismic slip and eaqrthquake phases
            CHKERRTHROW(TSSetPostEvaluate(ts, functionPostEvaluate<TimeOp>));

            // if a BDF method is used, start with a timestep size of maximal 0.1
            if (cfg_next->type == "bdf") {
                double h;
                CHKERRTHROW(TSGetTimeStep(ts, &h));
                h = PetscMin(h,0.1);
                CHKERRTHROW(TSSetTimeStep(ts, h));
            }

            // Overwrite settings by command line options
            CHKERRTHROW(TSSetFromOptions(ts));        
        }

        // for the compact ODE formulation, add Jacobian matrices
        if ((cfg_next->solution_size == "compact") && 
            (cfg_next->problem_formulation == "ode") && 
            (cfg_next->type == "bdf"))
                CHKERRTHROW(TSSetRHSJacobian(ts, timeop.getJacobianCompactODE(), 
                    timeop.getJacobianCompactODE(), RHSJacobianCompactODE<TimeOp>, &timeop));
        if ((cfg_next->solution_size == "extended") && 
            (cfg_next->problem_formulation == "ode") && 
            (cfg_next->type == "bdf"))
                CHKERRTHROW(TSSetRHSJacobian(ts, timeop.getJacobianExtendedODE(), 
                    timeop.getJacobianExtendedODE(), RHSJacobianExtendedODE<TimeOp>, &timeop));
        
        // integrator type 
        CHKERRTHROW(TSSetType(ts, cfg_next->type.c_str()));

        // rk settings
        if (cfg_next->type == "rk") {
            CHKERRTHROW(TSRKSetType(ts, cfg_next->rk_type.c_str()));
            solverStruct.useImplicitSolver = false;
        // bdf settings
        } else if (cfg_next->type == "bdf") {
            setBDFParameters(ts, timeop, cfg_next, solverStruct);
            solverStruct.useImplicitSolver = true;
        }
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
     * Sets the tolerances for the nonlinear solver
     * @param ts the TS context
     */
    static void setSNESTolerances(TS ts) {
        SNES snes;
        CHKERRTHROW(TSGetSNES(ts, &snes));
        double atol = 1e-10;                         // absolute tolerance (default = 1e-50) 
        double rtol = 0.;                            // relative tolerance (default = 1e-8)
        double stol = 0.;                            // relative tolerance (default = 1e-8)
        int maxit = 10;                              // maximum number of iteration (default = 50)    
        int maxf  = -1;                              // maximum number of function evaluations (default = 1000)  
        CHKERRTHROW(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
    }

    /**
     * Set up the BDF scheme
     * @param ts TS instance
     * @param solution_size compact or extended 
     * @param cfg specific configuration to see whether use manual implementations or not
     * @param solverStruct contains the pointer to the manual implementations
     */
    template <typename CFG, typename TimeOp>
    static void setBDFParameters(TS ts, TimeOp& seasop,
        const std::optional<tndm::SolverConfigSpecific>& cfg, CFG& solverStruct){

       if (cfg->bdf_order > 0) CHKERRTHROW(TSBDFSetOrder(ts, cfg->bdf_order));

        // set nonlinear solver settings
        SNES snes;
        KSP snes_ksp;
        PC snes_pc;
        TS_BDF          *bdf = (TS_BDF*)ts->data;

        solverStruct.previous_solution = &(bdf->work[1]);
        
        CHKERRTHROW(TSGetSNES(ts, &snes));

        if (cfg->bdf_custom_error_evaluation)
            ts->ops->evaluatewlte = *solverStruct.customErrorFct;     // manual LTE evaluation

        if (cfg->bdf_custom_Newton_iteration) {                       // manual Newton iteration
            CHKERRTHROW(SNESSetType(snes, SNESSHELL));
            CHKERRTHROW(SNESShellSetContext(snes, &seasop));
            CHKERRTHROW(SNESShellSetSolve(snes, *solverStruct.customNewtonFct));
        }
        CHKERRTHROW(SNESGetKSP(snes, &snes_ksp));
        CHKERRTHROW(KSPGetPC(snes_ksp, &snes_pc));
        CHKERRTHROW(KSPSetType(snes_ksp, cfg->bdf_ksp_type.c_str()));
        CHKERRTHROW(PCSetType(snes_pc, cfg->bdf_pc_type.c_str()));
        CHKERRTHROW(TSSetMaxSNESFailures(ts, -1));
        
        setSNESTolerances(ts);
    }
    /**
     * Custom implementation of the Newton algorithm 
     * @param snes solving context
     * @param x solution vector 
     */
    template <typename TimeOp> 
    static PetscErrorCode solveNewton(SNES snes, Vec x){

        Vec f = snes->vec_func;                     // rhs function for the KSP solver
        Vec Newton_step = snes->vec_sol_update;     // J(f(u))^{-1}f(u)
        Mat J,J_pre;                                // Jacobi matrices
        double norm_f, norm_init, norm_dx, norm_x, norm_f_prev;  // norm of the residual
        double atol, rtol, stol;
        int maxit, maxf;
        bool custom_solver;

        void* ctx;
        TimeOp* seasop;
        KSP ksp;        
        SNESConvergedReason reason;

        PetscFunctionBegin;
        SNESSetConvergedReason(snes, SNES_CONVERGED_ITERATING);

        CHKERRTHROW(SNESShellGetContext(snes, &ctx));
        seasop = reinterpret_cast<TimeOp*>(ctx);        
        auto& solverStruct = seasop->getSolverParameters();
        custom_solver = solverStruct.current_solver_cfg->bdf_custom_LU_solver;

        if (!f)             CHKERRTHROW(VecDuplicate(x, &f));
        if (!Newton_step)   CHKERRTHROW(VecDuplicate(x, &Newton_step));
        
        CHKERRTHROW(SNESGetJacobian(snes, &J, &J_pre, nullptr, nullptr));
        CHKERRTHROW(SNESGetTolerances(snes, &atol, &rtol, &stol, &maxit, &maxf));

        CHKERRTHROW(SNESGetKSP(snes, &ksp));

        CHKERRTHROW(SNESComputeFunction(snes, x, f));    //evaluate RHS of the ODE

        if (solverStruct.current_formulation == TimeOp::SECOND_ORDER_ODE) 
            seasop->updateRHSNewtonIteration(f);

        CHKERRTHROW(VecNorm(f, NORM_INFINITY, &norm_f));
        norm_init = norm_f;


        int total_it_num=0;
        for(int n = 0; n < maxit; n++) {
            CHKERRTHROW(SNESComputeJacobian(snes, x, J, J_pre));  // get the Jacobian
            if (custom_solver) {
                seasop->applyCustomLUSolver(Newton_step, f, J, ksp);
            } else {
                CHKERRTHROW(KSPSetOperators(ksp, J, J_pre));      // set it to the KSP
                CHKERRTHROW(KSPSolve(ksp, f, Newton_step));       // solve the Jacobian system
            }
            CHKERRTHROW(VecAXPY(x, -1, Newton_step));             // update the solution vector

            CHKERRTHROW(SNESComputeFunction(snes, x, f));         // evaluate the algebraic function
            if (solverStruct.current_formulation == TimeOp::SECOND_ORDER_ODE) 
                seasop->updateRHSNewtonIteration(f);

            norm_f_prev = norm_f;
            CHKERRTHROW(VecNorm(f, NORM_INFINITY, &norm_f));      // calculate some nomrs
            CHKERRTHROW(VecNorm(Newton_step, NORM_INFINITY, &norm_dx));
            CHKERRTHROW(VecNorm(x, NORM_INFINITY, &norm_x));

            int it_num;
            KSPGetIterationNumber(ksp, &it_num);
            total_it_num += it_num;

            // diverged
            if (norm_f > norm_init) {
                SNESSetConvergedReason(snes, SNES_DIVERGED_FUNCTION_DOMAIN );
                std::cout << "diverged with norm " << norm_f << " after " << n+1 << " iterations -- Restart Newton step" << std::endl;
                PetscFunctionReturn(0);
            }

            // converged
            if (norm_f < atol) {
                SNESSetConvergedReason(snes, SNES_CONVERGED_FNORM_ABS);
                solverStruct.FMax = norm_f;
                solverStruct.KSP_iteration_count = (double)total_it_num / (n+1);
                // std::cout << "converged with norm " << norm_f << " after " << n+1 << " iterations with " 
                //           << (double)total_it_num / (n+1) << " KSP iterations per step" << std::endl;
                PetscFunctionReturn(0);
            }
            if (norm_f / norm_init < rtol) {
                SNESSetConvergedReason(snes, SNES_CONVERGED_FNORM_RELATIVE);
                solverStruct.FMax = norm_f;
                solverStruct.KSP_iteration_count = (double)total_it_num / (n+1);
                // std::cout << "converged with norm " << norm_f << " after " << n+1 << " iterations with " 
                //           << (double)total_it_num / (n+1) << " KSP iterations per step" << std::endl;
                PetscFunctionReturn(0);
            }
            if (norm_dx / norm_x < stol) {
                SNESSetConvergedReason(snes, SNES_CONVERGED_SNORM_RELATIVE);
                solverStruct.FMax = norm_f;
                solverStruct.KSP_iteration_count = (double)total_it_num / (n+1);
                // std::cout << "converged with norm " << norm_f << " after " << n+1 << " iterations with " 
                //           << (double)total_it_num / (n+1) << " KSP iterations per step" << std::endl;
                PetscFunctionReturn(0);
            }

            // std::cout << "norm: "<<norm_f << std::endl;

            if ((norm_f < 1e-7) && (std::abs(norm_f - norm_f_prev) / norm_f < 1e0)) {     // only used for absolute best convergence
                solverStruct.FMax = norm_f; // = f_norm for the absolute error
                solverStruct.KSP_iteration_count = (double)total_it_num / (n+1);
                SNESSetConvergedReason(snes, SNES_CONVERGED_SNORM_RELATIVE);
                PetscFunctionReturn(0);
            }

        }

        solverStruct.FMax = norm_f;
        solverStruct.KSP_iteration_count = (double)total_it_num / maxit;
        // std::cout << "converged with norm " << norm_f << " after " << maxit-1 << " iterations with " 
        //             << (double)total_it_num / maxit << " KSP iterations per step" << std::endl;
        SNESSetConvergedReason(snes, SNES_CONVERGED_ITS );
        PetscFunctionReturn(0);
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
        Vec            X = bdf->work[0], Y = bdf->vec_lte;
        PetscInt       n;
        Vec            V,V0;
        Vec            vecs[7];
        PetscScalar    alpha[7];

        V = bdf->vec_dot;
        V0 = bdf->vec_wrk;

        PetscFunctionBegin;
        n = PetscMin(bdf->k+2, bdf->n);
        *order = n-1;

        // get Lagrangian polynomials from time steps
        LagrangeBasisDers(n,bdf->time[0],bdf->time,alpha);
        for (int i=1; i<n; i++) {
                vecs[i] = bdf->transientvar ? bdf->tvwork[i] : bdf->work[i];
        }

        VecZeroEntries(V0);
        VecMAXPY(V0,n-1,alpha+1,vecs+1);
        bdf->shift = PetscRealPart(alpha[0]);

        // solve nonlinear system
        CHKERRTHROW(VecCopy(X, Y));
        SNESSolve(ts->snes,nullptr,Y);

        // calculate the error between both solutions
        TSErrorWeightedNorm(ts,X,Y,wnormtype,wlte,&wltea,&wlter);

        CHKERRTHROW(VecAXPY(Y,-1,X));

        V = nullptr;
        V0 = nullptr;
        PetscFunctionReturn(0);
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
static PetscErrorCode TSAdaptChoose_Custom(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter)
{
  Vec            Y;
  DM             dm;
  PetscInt       order  = PETSC_DECIDE;
  PetscReal      enorm  = -1;
  PetscReal      enorma,enormr;
  PetscReal      safety = adapt->safety;
  PetscReal      hfac_lte,h_lte;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *next_sc = 0;   /* Reuse the same order scheme */
  *wltea   = -1;  /* Weighted absolute local truncation error is not used */
  *wlter   = -1;  /* Weighted relative local truncation error is not used */

  if (ts->ops->evaluatewlte) {
    ierr = TSEvaluateWLTE(ts,adapt->wnormtype,&order,&enorm);CHKERRQ(ierr);
    if (enorm >= 0 && order < 1) SETERRQ1(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Computed error order %D must be positive",order);
  } else if (ts->ops->evaluatestep) {
    if (adapt->candidates.n < 1) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"No candidate has been registered");
    if (!adapt->candidates.inuse_set) SETERRQ1(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"The current in-use scheme is not among the %D candidates",adapt->candidates.n);
    order = adapt->candidates.order[0];
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&Y);CHKERRQ(ierr);
    ierr = TSEvaluateStep(ts,order-1,Y,NULL);CHKERRQ(ierr);
    ierr = TSErrorWeightedNorm(ts,ts->vec_sol,Y,adapt->wnormtype,&enorm,&enorma,&enormr);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&Y);CHKERRQ(ierr);
  }

  if (enorm < 0) {
    *accept  = PETSC_TRUE;
    *next_h  = h;            /* Reuse the old step */
    *wlte    = -1;           /* Weighted local truncation error was not evaluated */
    PetscFunctionReturn(0);
  }

  /* Determine whether the step is accepted of rejected */
  if (enorm > 1) {
    if (!*accept) safety *= adapt->reject_safety; /* The last attempt also failed, shorten more aggressively */
    if (h < (1 + PETSC_SQRT_MACHINE_EPSILON)*adapt->dt_min) {
      ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting because step size %g is at minimum\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else if (adapt->always_accept) {
      ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting step of size %g because always_accept is set\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else {
      ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %g, rejecting step of size %g\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_FALSE;
    }
  } else {
    ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting step of size %g\n",(double)enorm,(double)h);CHKERRQ(ierr);
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

  *next_h = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
  *wlte   = enorm;
  PetscFunctionReturn(0);
}

    /**
     * Calculate the cofficients for the Lagrangian extrapolation
     * @param n number of interpolation points
     * @param t current time
     * @param T vector with the times at these points
     * @param dL vector with the coefficients
     */
    PETSC_STATIC_INLINE void LagrangeBasisDers(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar dL[])
    {
    PetscInt  k,j,i;
    for (k=0; k<n; k++)
        for (dL[k]=0, j=0; j<n; j++)
        if (j != k) {
            PetscReal L = 1/(T[k] - T[j]);
            for (i=0; i<n; i++)
            if (i != j && i != k)
                L *= (t - T[i])/(T[k] - T[i]);
            dL[k] += L;
        }
    }

    static PetscErrorCode TSBDF_VecLTE(TS ts,PetscInt order,Vec lte)
    {
        TS_BDF         *bdf = (TS_BDF*)ts->data;
        PetscInt       i,n = order+1;
        PetscReal      *time = bdf->time;
        Vec            *vecs = bdf->work;
        PetscScalar    a[8],b[8],alpha[8];
        PetscErrorCode ierr;

        PetscFunctionBegin;
        LagrangeBasisDers(n+0,time[0],time,a); a[n] =0;
        LagrangeBasisDers(n+1,time[0],time,b);
        for (i=0; i<n+1; i++) alpha[i] = (a[i]-b[i])/a[0];
        ierr = VecZeroEntries(lte);CHKERRQ(ierr);
        ierr = VecMAXPY(lte,n+1,alpha,vecs);CHKERRQ(ierr);
        PetscFunctionReturn(0);
    }

    template <typename TimeOp> 
    static PetscErrorCode adaptBDFOrder(TS ts, PetscInt& new_order)
    {
        TimeOp* seasop;
        CHKERRTHROW(TSGetApplicationContext(ts, &seasop));
        const auto& cfg = seasop->getGeneralSolverConfiguration();

        TS_BDF         *bdf = (TS_BDF*)ts->data;
        PetscInt       k = bdf->k;
        PetscReal      wlte,wltea,wlter, hfac, best_hfac;
        Vec            X = bdf->work[0], Y = bdf->vec_lte;
        PetscErrorCode ierr;

        PetscFunctionBegin;
        k = PetscMin(k,bdf->n-1);
        new_order = PetscMax(1,k-1);
        best_hfac = 0.0;

        double * x;

        for (int i = PetscMax(1,k-1); i <= PetscMin(bdf->n-1,k+1); i++ ) {
            ierr = TSBDF_VecLTE(ts,i,Y);CHKERRQ(ierr);
            ierr = VecAXPY(Y,1,X);CHKERRQ(ierr);
            ierr = TSErrorWeightedNorm(ts,X,Y,ts->adapt->wnormtype,&wlte,&wltea,&wlter);CHKERRQ(ierr);
            hfac = PetscPowReal(wlte,((PetscReal)-1)/(i+1));
            if (best_hfac < hfac) {
                best_hfac = hfac;
                new_order = i;
            }            
        }
        PetscFunctionReturn(0);
    }


    /**
     * Resets the time to 0 at the beginning of the earthquake because of bad precision
     * @param ts TS instance
     * @param time_eq variable to store the time at the earthquake start
     */
    static PetscErrorCode reducedTimeBeginEQ(TS ts, double& time_eq){
        TS_BDF         *bdf = (TS_BDF*)ts->data;
        PetscInt       n = (PetscInt)(sizeof(bdf->work)/sizeof(Vec));

        PetscFunctionBegin;
        time_eq = bdf->time[n-1];
        for (int i = 0; i < n; i++) bdf->time[i] -= time_eq;

        ts->ptime -= time_eq;
        PetscFunctionReturn(0);
    }



    /**
     * Restores the time at the end of the earthquake
     * @param ts TS instance
     * @param time_eq variable to retrieve the time at the earthquake start
     */
    static PetscErrorCode reducedTimeEndEQ(TS ts, double& time_eq){
        TS_BDF         *bdf = (TS_BDF*)ts->data;
        PetscInt       n = (PetscInt)(sizeof(bdf->work)/sizeof(Vec));

        PetscFunctionBegin;
        for (int i = 0; i < n; i++) bdf->time[i] += time_eq;

        ts->ptime += time_eq;

        time_eq = 0.;
        PetscFunctionReturn(0);
    }


    /**
     * Resets the time to 0 at the end of each step
     * @param ts TS instance
     * @param time_eq variable to store the time at the earthquake start
     */
    static PetscErrorCode reducedTimeEachStep(TS ts, double& time_eq){
        TS_BDF         *bdf = (TS_BDF*)ts->data;
        PetscInt       n = (PetscInt)(sizeof(bdf->work)/sizeof(Vec));
        double         time_increase;

        PetscFunctionBegin;
        time_increase = bdf->time[n-1];
        time_eq += time_increase;
        for (int i = 0; i < n; i++) bdf->time[i] -= time_increase;
        ts->ptime -= time_increase;
        PetscFunctionReturn(0);
    }


    /**
     * Performs the fourth order RK scheme (wihout error correction) 
     * @param ts TS instance
     * @param t previous simulation time
     * @param dt time step size to current simulation time
     * @param X solution vector at previous time
     * @param Y solution vector at current time
     * @param f function to evaluate the right-hand side
     * @param ctx pointer to the SEAS instance
     */
    static PetscErrorCode RK4(TS ts, double t, double dt, Vec& X, Vec& Y, PetscErrorCode (*f)(TS,PetscReal,Vec,Vec,void*), void* ctx){
        Vec F[4];      // working vector for the right hand-side
        double a[4];

        for(int i = 0; i < 4; i++) CHKERRTHROW(VecDuplicate(X, &F[i]));

        CHKERRTHROW(VecCopy(X,Y));  // Y is a working vector for now
        f(ts, t + 0.0 * dt, Y, F[0], ctx);

        CHKERRTHROW(VecAXPY( Y, 0.5 * dt, F[0]));
        f(ts, t + 0.5 * dt, Y, F[1], ctx);

        CHKERRTHROW(VecWAXPY(Y, 0.5 * dt, F[1], X));
        f(ts, t + 0.5 * dt, Y, F[2], ctx);

        CHKERRTHROW(VecWAXPY(Y, 1.0 * dt, F[2], X));
        f(ts, t + 1.0 * dt, Y, F[3], ctx);

        a[0] = 1./6. * dt;
        a[1] = 1./3. * dt;
        a[2] = 1./3. * dt;
        a[3] = 1./6. * dt;

        CHKERRTHROW(VecCopy(X,Y));  // Y is now the next solution vector
        CHKERRTHROW(VecMAXPY(Y, 4, a, F));

        return 0;

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
