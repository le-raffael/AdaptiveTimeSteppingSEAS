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
        CHKERRTHROW(TSSetRHSJacobian(ts_, timeop.getJacobian(), timeop.getJacobian(), RHSJacobian<TimeOp>, &timeop));
        CHKERRTHROW(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP));
        if (cfg){
            // read petsc options from file
            std::cout<<"Read PETSc options from configuration file. Eventual PETSc command line arguments will overwrite settings from the configuration file."<<std::endl;

            // integrator type
            CHKERRTHROW(TSSetType(ts_, cfg->ts_type.c_str()));

            // rk settings
            if (cfg->ts_type == "rk"){
                CHKERRTHROW(TSRKSetType(ts_, cfg->ts_rk_type.c_str()));
            }

            // bdf settings
            if (cfg->ts_type == "bdf"){

                CHKERRTHROW(TSBDFSetOrder(ts_, cfg->ts_bdf_order));

                // set nonlinear solver settings
                SNES snes;
                KSP snes_ksp;
                PC snes_cp;
                CHKERRTHROW(TSGetSNES(ts_, &snes));

                if (cfg->bdf_manual_error_evaluation)
                    ts_->ops->evaluatewlte = evaluateEmbeddedMethodBDF;     // manual LTE evaluation

                if (cfg->bdf_manual_Newton_iteration) {
                    CHKERRTHROW(SNESSetType(snes, SNESSHELL));
                    CHKERRTHROW(SNESShellSetContext(snes, &ts_));
                    CHKERRTHROW(SNESShellSetSolve(snes, solveNewton));
                }

                CHKERRTHROW(SNESGetKSP(snes, &snes_ksp));
                CHKERRTHROW(KSPGetPC(snes_ksp, &snes_cp));
                CHKERRTHROW(KSPSetType(snes_ksp, cfg->ksp_type.c_str()));
                CHKERRTHROW(PCSetType(snes_cp, cfg->pc_type.c_str()));
                CHKERRTHROW(TSSetMaxSNESFailures(ts_, -1));

                double atol = 1e-50;    // absolute tolerance (default = 1e-50) 
                double rtol = 1e-8;    // relative tolerance (default = 1e-8)  
                double stol = 1e-8;    // relative tolerance (default = 1e-8)  
                int maxit = 50;      // absolute tolerance (default = 50)    
                int maxf = -1;       // absolute tolerance (default = 1000)  
                CHKERRTHROW(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
                CHKERRTHROW(SNESSetFromOptions(snes));
            }

            // norm type
            TSAdapt adapt;
            CHKERRTHROW(TSGetAdapt(ts_, &adapt));
            if (cfg->ts_adapt_wnormtype == "2") {ts_->adapt->wnormtype = NormType::NORM_2; }    // this is very hacky 
            else if (cfg->ts_adapt_wnormtype == "infinity") {ts_->adapt->wnormtype = NormType::NORM_INFINITY; }
            else {std::cerr<<"Unknown norm! use \"2\" or \"infinity\""<<std::endl; }

            // set ksp type
            CHKERRTHROW(KSPSetType(ksp, cfg->ksp_type.c_str()));

            // set preconditioner options
            PC pc;
            CHKERRTHROW(KSPGetPC(ksp, &pc));
            CHKERRTHROW(PCSetType(pc, cfg->pc_type.c_str()));
            CHKERRTHROW(PCFactorSetMatSolverType(pc, cfg->pc_factor_mat_solver_type.c_str()));


            // set initial tolerances
            initializeTolerances(timeop, cfg);

            // Store the seas operator in the context if needed
            CHKERRTHROW(TSSetApplicationContext(ts_, &timeop));

            // adapt tolerances to earthquake / not earthquake
            CHKERRTHROW(TSSetPostEvaluate(ts_, updateAfterTimeStep<TimeOp>));

            // Overwrite settings by command line options
            CHKERRTHROW(TSSetFromOptions(ts_));
            // adapt->ops->choose = TSAdaptChoose_Custom;            

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
     * evaluate the rhs of the ODE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains V and psi on all fault nodes in block format)
     * @param F first time derivative of u (solution vector to be written to)
     * @param ctx pointer to Seas operator instanceTimeOp*
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
     * evaluate the rhs of the ODE
     * @param ts TS object
     * @param t current simulation time
     * @param u current state vector (contains V and psi on all fault nodes in block format)
     * @param A Jacobian matrix
     * @param B matrix for the preconditioner (take the same as A)
     * @param ctx pointer to Seas operator instance
     */
    template <typename TimeOp>
    static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec u, Mat A, Mat B, void *ctx){
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        A = self->getJacobian();
        B = self->getJacobian();
//	    CHKERRTHROW(MatView(A,PETSC_VIEWER_STDOUT_SELF));         
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
     * Initialize the tolerances in the SEAS operator
     * @param timeop instance of the seas operator
     * @param cfg solver configuration read from .toml file
     */
    template <typename TimeOp> 
    void initializeTolerances(TimeOp& timeop, const std::optional<tndm::SolverConfig>& cfg){
        auto& tolStruct = timeop.getTolerances();
        tolStruct.S_rtol = cfg->S_rtol;
        tolStruct.S_atol = cfg->S_atol;
        tolStruct.psi_rtol = cfg->psi_rtol;
        tolStruct.psi_atol_as = cfg->psi_atol_as;
        tolStruct.psi_atol_eq = cfg->psi_atol_eq;
        tolStruct.checkAS = true;

        setTolerancesSAndPSI(ts_, timeop, cfg->S_rtol, cfg->S_atol, cfg->psi_rtol, cfg->psi_atol_as);
    }

    /**
     * Sets the absolute and relative tolerances 
     * @param ts the TS context
     * @param timeop instance of the seas operator
     * @param S_rtol relative error tolerance of the slip 
     * @param S_atol absolute error tolerance of the slip 
     * @param psi_rtol relative error tolerance of the state variable 
     * @param psi_atol absolute error tolerance of the state varaible 
     */
    template <typename TimeOp> 
    static void setTolerancesSAndPSI(TS ts, TimeOp& timeop, double S_rtol, double S_atol, double psi_rtol, double psi_atol){
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
                a(i) = S_atol;
                r(i) = S_rtol;
            }
            // set state variable tolerances
            for ( int i = PsiIndex; i < PsiIndex + nbf; i++){
                a(i) = psi_atol;
                r(i) = psi_rtol;
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

        auto& tolStruct = seasop->getTolerances();

        if (tolStruct.checkAS && (seasop->VMax() > seasop->getV0())){          // change from as -> eq
            setTolerancesSAndPSI(ts, *seasop, tolStruct.S_rtol, tolStruct.S_atol,   
                                 tolStruct.psi_rtol, tolStruct.psi_atol_eq);
//            CHKERRTHROW(TSSetType(ts, TSRK));
//            CHKERRTHROW(TSRKSetType(ts, TSRK5DP));
            tolStruct.checkAS = false;
            std::cout << "Enter earthquake phase" << std::endl;
        } else if (!tolStruct.checkAS && (seasop->VMax() < seasop->getV0())){  // change from eq -> as
            setTolerancesSAndPSI(ts, *seasop, tolStruct.S_rtol, tolStruct.S_atol,
                                 tolStruct.psi_rtol, tolStruct.psi_atol_as);
            tolStruct.checkAS = true;
            std::cout << "Exit earthquake phase" << std::endl;
        }

        double time; 
        Vec new_x;
        CHKERRTHROW(TSGetTime(ts, &time));
        CHKERRTHROW(TSGetSolution(ts, &new_x));        
        seasop->calculateApproximateJacobian(time, new_x);



        // Vec Xx;
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

    std::unique_ptr<PetscBlockVector> state_;
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
