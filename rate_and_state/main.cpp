#include <petscsys.h>
#include <petscts.h>
#include <petscvec.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "Problem.h"
#include "Zero.h"
#include "math_functions.h"

double computeSlipRate(PetscReal t, double psi) {
    double tau = tau_star(t);

    auto F = [&tau, &psi](double V) { return fl(tau, V, psi); };
    double Vother = tndm::zeroIn(-tau / eta, tau / eta, F);

    return Vother;
}

PetscErrorCode rhs(TS ts, PetscReal t, Vec U, Vec F, void* ctx) {
    PetscScalar const* u;
    PetscScalar* f;
    VecGetArrayRead(U, &u);
    VecGetArray(F, &f);

    double const psi = *u;

    double V = computeSlipRate(t, psi);
    *f = 1.0 - V * psi / L - (1.0 - V_star(t) * psi_star(t) / L) +
     dpsi_stardt(t);
    //*f = (b * V0 / L) * (exp((f0 - psi) / b) - V / V0 -
    //                     exp((f0 - psi_star(t)) / b) + V_star(t) / V0) +
         dpsi_stardt(t);

    VecRestoreArray(F, &f);
    VecRestoreArrayRead(U, &u);
    return 0;
}

double rhs(size_t step, double t, double U, void* ctx) {

    double const psi = U;

    //std::cout<<"psi: "<<psi<<std::endl;

    double V = computeSlipRate(t, psi);
    return 1.0 - V * psi / L - (1.0 - V_star(t) * psi_star(t) / L) +
     dpsi_stardt(t);
    //return (b * V0 / L) * (exp((f0 - psi) / b) - V / V0 -
    //                     exp((f0 - psi_star(t)) / b) + V_star(t) / V0) +
    //     dpsi_stardt(t);
}

struct MonitorData {
    std::ofstream file;
    double V_err = 0.0;
    double psi_err = 0.0;
};

struct paramsData {
    double psi_err;
    double V_err;
    size_t timeSteps;
    double iterationSteps;
};

struct PIController {
    double k = 1;
    double factor = 0.98;
    double errorTolerance = 1e-6;
    double alpha = 0.4;
    double beta = 0.1;
};

double stepSizeUpdatePIController(PIController* c, double h, double error, double new_error){
    if (error == 0) error = new_error;
    if (new_error != 0){
        return pow(c->factor * c->errorTolerance / new_error, c->alpha / c->k) * pow(error / new_error, c->beta / c->k) * h;
    } else {
        return h;
    }
}

PetscErrorCode monitor(TS ts, PetscInt step, PetscReal time, Vec U, void* ctx) {
    auto dat = reinterpret_cast<MonitorData*>(ctx);
    PetscScalar const* u;
    VecGetArrayRead(U, &u);
    double psi = *u;
    double V = computeSlipRate(time, psi);
    double V_err = fabs(V_star(time) - V);
    double psi_err = fabs(psi_star(time) - psi);
    dat->file << time << "," << V << "," << psi << "," << V_err << ","
              << psi_err << std::endl;
    dat->V_err = std::max(V_err, dat->V_err);
    dat->psi_err = std::max(psi_err, dat->psi_err);
    VecRestoreArrayRead(U, &u);
    return 0;
}

void monitor(size_t step, double time, std::vector<double> U, MonitorData* dat) {
    double psi = U[step];
    double V = computeSlipRate(time, psi);
    double V_err = fabs(V_star(time) - V);
    double psi_err = fabs(psi_star(time) - psi);
    dat->file << time << "," << V << "," << psi << "," << V_err << ","
              << psi_err << std::endl;
    dat->V_err = std::max(V_err, dat->V_err);
    dat->psi_err = std::max(psi_err, dat->psi_err);
}

int solvePETSC(int argc, char* argv[], double endTime){
    TS ts;
    Vec x;
    PetscScalar* X;

    CHKERRQ(PetscInitialize(&argc, &argv, nullptr, nullptr));

    CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD, 1, &x));
    MonitorData dat;
    dat.file.open("result.csv");
    dat.file << "time,V,psi,V_err,psi_err" << std::endl;

    CHKERRQ(TSCreate(PETSC_COMM_WORLD, &ts));
    CHKERRQ(TSSetProblemType(ts, TS_NONLINEAR));
    CHKERRQ(TSSetSolution(ts, x));
    CHKERRQ(TSSetRHSFunction(ts, nullptr, rhs, nullptr));
    CHKERRQ(TSSetMaxTime(ts, endTime));
    CHKERRQ(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    CHKERRQ(TSMonitorSet(ts, monitor, &dat, nullptr));

    CHKERRQ(TSSetFromOptions(ts));
    CHKERRQ(TSSetUp(ts));

    VecGetArray(x, &X);
    *X = psi_star(0.0);
    VecRestoreArray(x, &X);


    CHKERRQ(TSSolve(ts, x));
    dat.file.close();

    std::cout << "V_err: " << dat.V_err << std::endl;
    std::cout << "psi_err: " << dat.psi_err << std::endl;

    CHKERRQ(TSDestroy(&ts));
    CHKERRQ(VecDestroy(&x));
    auto ierr = PetscFinalize();
    return ierr;



}

void solveLapusta(double endTime){
    // set up monitor to print results
    MonitorData dat;
    dat.file.open("result.csv");
    dat.file << "time,V,psi,V_err,psi_err" << std::endl;

    // initialization of the solution vector
    double dt_min = 1e-5;
    double dt_max = 1e5;
    double dt = dt_min;

    size_t max_num_steps = ceil(endTime / dt_min);

    double Vel = V0;

    double time = 0.0;
    size_t step = 0;

    std::vector<double> solution_vector;
    solution_vector.reserve(max_num_steps);

    // initial value
    solution_vector.push_back(psi_star(0.0));

    // time loop
    while (time < endTime){

        // print to file
        monitor(step, time, solution_vector, &dat);
        std::cout<<step<<": time: "<<time<<", timestep: "<<dt<<std::endl;

        // get functional for the right hand side
        std::function<double(double,double)> func = [&dat, &step](double U, double t){return rhs(step, t, U, &dat);};

        // apply numerical scheme
        RK4(solution_vector, func, time, dt, step);

        // update time iteration
        step++;
        time += dt;

        // evaluate time step to choose
        double xi = 0.5;    // take this value as long as tau is constant 
        Vel = computeSlipRate(time, solution_vector[step]);

        dt = xi * L / Vel;
        dt = std::max(dt, dt_min);
        dt = std::min(dt, dt_max);
        if (time + dt > endTime) dt = endTime - time;

        
    }

    monitor(step, time, solution_vector, &dat);
    std::cout<<step<<": time: "<<time<<", timestep: "<<dt<<std::endl;

    dat.file.close();

    std::cout << "V_err: " << dat.V_err << std::endl;
    std::cout << "psi_err: " << dat.psi_err << std::endl;

}

void solvePID(double endTime, PIController* controller,  std::string timeIntegrator, paramsData* data = nullptr) {
    // set up monitor to print results
    MonitorData dat;
    dat.file.open("../results/result_"+timeIntegrator+".csv");
    dat.file << "time,V,psi,V_err,psi_err" << std::endl;

    // file to print error estimates vs actual error
    std::ofstream fileErrorEstimate;
    fileErrorEstimate.open("../results/errorEstimate_"+timeIntegrator+".csv");
    fileErrorEstimate << "time,local_err,err_expected" << std::endl;


    // initialization of the solution vector
    double dt_min = 1e-5;
    double dt_max = 1e5;
    double dt = 1e-5;
    double dt_n_1 = dt;
    double dt_n_2 = dt;

    size_t max_num_steps = ceil(endTime / dt_min);

    double time = 0.0;
    size_t step = 0;

    std::vector<double> solution_vector;
    solution_vector.reserve(max_num_steps);

    // initial value
    solution_vector.push_back(psi_star(0.0));

    // adaptive time step parameters
    double localError = 0;
    double previouslocalError = 0;
    size_t iterationSteps = 0;


    std::function<double(double,double)> func = [&dat, &step](double U, double t){return rhs(step, t, U, &dat);};

    int i_max = 0;
    if (timeIntegrator == "BDF12") i_max = 1; // perform one additional time step to apply BDF2
    if (timeIntegrator == "BDF23") i_max = 2;// perform two additional time step to apply BDF3

    // initial steps for BDF methods
    for (int i = 0; i<i_max;++i){
        RKF45(solution_vector, localError, func, time, dt, step);

 //       std::cout<<step<<": time: "<<time<<", timestep: "<<dt<<std::endl;

        dt_n_2 = dt_n_1;
        dt_n_1 = dt;
        time+= dt;
        step++;

        monitor(step, time, solution_vector, &dat);
        fileErrorEstimate << time << "," << localError << "," << fabs(fabs(psi_star(time) - solution_vector[step])-previouslocalError) << std::endl;
    }


    if (timeIntegrator == "RKF45") RKF45(solution_vector, localError, func, time, dt, step);
    if (timeIntegrator == "RKDP45") RKDP45(solution_vector, localError, func, time, dt, step);

    previouslocalError = localError;

    // time loop
    while (time < endTime){
        // evaluate time step to choose
        func = [&dat, &step](double U, double t){return rhs(step, t, U, &dat);};

        // apply numerical scheme
        dt = stepSizeUpdatePIController(controller, dt, previouslocalError, localError);
        dt = std::max(dt, dt_min);
        dt = std::min(dt, dt_max);
        if (time + dt > endTime) dt = endTime - time;        


        if (timeIntegrator == "RKF45") RKF45(solution_vector, localError, func, time, dt, step);
        if (timeIntegrator == "RKDP45") RKDP45(solution_vector, localError, func, time, dt, step);
        if (timeIntegrator == "BDF12") timeAdaptiveBDF12Method(solution_vector, localError, func, time, dt, dt_n_1, step);
        if (timeIntegrator == "BDF23") timeAdaptiveBDF23Method(solution_vector, localError, func, time, dt, dt_n_1, dt_n_2, step);

        iterationSteps++;
        while (localError > controller->errorTolerance){
            dt = stepSizeUpdatePIController(controller, dt, previouslocalError, localError);            

            if (timeIntegrator == "RKF45") RKF45(solution_vector, localError, func, time, dt, step);
            if (timeIntegrator == "RKDP45") RKDP45(solution_vector, localError, func, time, dt, step);
            if (timeIntegrator == "BDF12") timeAdaptiveBDF12Method(solution_vector, localError, func, time, dt, dt_n_1, step);
            if (timeIntegrator == "BDF23") timeAdaptiveBDF23Method(solution_vector, localError, func, time, dt, dt_n_1, dt_n_2, step);

            iterationSteps++;
            if (iterationSteps > 1000) break; 
            if (dt < dt_min) dt = dt_min; break;
        }

        // print metrics
    //    std::cout<<step<<": time: "<<time<<", timestep: "<<dt<<std::endl;

        // update time iteration
        step++;
        dt_n_2 = dt_n_1;
        dt_n_1 = dt;
        time += dt;

        monitor(step, time, solution_vector, &dat);
        double trueLocalTruncationError = fabs(fabs(psi_star(time) - solution_vector[step]) -
                                          fabs(psi_star(time-dt_n_1) - solution_vector[step-1]));
        fileErrorEstimate << time << "," << localError << "," << trueLocalTruncationError << std::endl;

        previouslocalError = localError;

    }

  //  std::cout<<step<<": time: "<<time<<", timestep: END"<<std::endl;

    dat.file.close();

    std::cout << "V_err: " << dat.V_err << ", psi_err: " << dat.psi_err << ", number of timesteps: "<< step << std::endl;

    if (data != nullptr){
        data->V_err = dat.V_err;
        data->psi_err = dat.psi_err;
        data->timeSteps = step;
        data->iterationSteps = iterationSteps * 1./ step;
    }
}

void solvePIDparams(double endTime, PIController* controller, std::string timeIntegrator){
    std::ofstream fileParams;
    fileParams.open("../params_"+timeIntegrator+".csv");
    fileParams<<"alpha,beta,V_err,psi_err,num_timesteps,num_iter"<<std::endl;

    paramsData data;

    size_t num_alpha = 500;
    size_t num_beta = 500;

    double alpha_min = 0.01;
    double alpha_max = 1;

    double beta_min = 0.01;
    double beta_max = 1;

    double alpha, beta;
    for (int i=0; i<num_alpha; ++i){
        alpha = alpha_min + i * (alpha_max - alpha_min) / (num_alpha - 1);
        controller->alpha = alpha;

        for (int j=0; j<num_beta; ++j){
            beta = beta_min + j * (beta_max - beta_min) / (num_beta - 1);

            std::cout<<"Start program with alpha = "<<alpha<<" and beta = "<<beta<<std::endl;

            controller->beta = beta;

            solvePID(endTime, controller, timeIntegrator, &data);

            fileParams<<alpha<<","<<beta<<","<<data.V_err<<","<<data.psi_err<<","<<data.timeSteps<<","<<data.iterationSteps<<std::endl;
        }
    }

   fileParams.close(); 

}   

int main(int argc, char* argv[]) {
    double endTime = 5.0;

    PIController controller;

    std::string solverType = "PETSc";
    
    if (argc >= 2 && argv[1][0] != '-') {
        solverType = argv[1];
    }

    if (solverType == "PETSc") {
        solvePETSC(argc, argv, endTime); 
    }else if (solverType == "lapusta"){
        solveLapusta(endTime);
    }else if (solverType == "PID"){
        std::string integrationType = "RKF45";
        if (argc >= 3) integrationType = argv[2];
        bool solveParams = false; 
        if ((argc >= 4) && (argv[3][0] == 'P')) solveParams = true;
        if (integrationType == "RKF45"){
            controller.k = 5;
            controller.alpha = 0.4;
            controller.beta = 0.6;             
            solveParams ? solvePIDparams(endTime, &controller, integrationType)
                        : solvePID(endTime, &controller, integrationType);
        } else if (integrationType == "RKDP45"){
            controller.k = 5;
            solveParams ? solvePIDparams(endTime, &controller, integrationType)
                        : solvePID(endTime, &controller, integrationType);
        } else if (integrationType == "BDF12"){
            controller.alpha = 0.9;
            controller.beta = 0.4;             
            controller.k = 2;
            solveParams ? solvePIDparams(endTime, &controller, integrationType)
                        : solvePID(endTime, &controller, integrationType);
        } else if (integrationType == "BDF23"){
            controller.alpha = 0.9;
            controller.beta = 0.4;             
            controller.k = 3;
            solveParams ? solvePIDparams(endTime, &controller, integrationType)
                        : solvePID(endTime, &controller, integrationType);
        } else if (integrationType == "PIparams"){
            controller.k = 2;
            solvePIDparams(endTime, &controller, integrationType);
        } else {
            std::cout<<"unknown integration type!! use either 'RKF45' (default), 'RKDP45', 'BDF12', 'BDF23' or 'PIparams'"<<std::endl;
        }
    } else {
        std::cout<<"unknown solver!! Use either 'PETSc', 'lapusta', or 'PID' as first argument (default is PETSc)"<<std::endl<<"Exiting program..."<<std::endl;
    }
}
