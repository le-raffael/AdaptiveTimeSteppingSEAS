#include <petscmat.h>
#include <petscsys.h>
#include <petscts.h>
#include <petscvec.h>

#include <fstream>
#include <iostream>

#include "Problem.h"

PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, void* ctx) {
    PetscScalar const* u;
    PetscScalar const* u_t;
    PetscScalar* f;
    CHKERRQ(VecGetArrayRead(U, &u));
    CHKERRQ(VecGetArrayRead(U_t, &u_t));
    CHKERRQ(VecGetArray(F, &f));

    double const V = u[0];
    double const psi = u[1];
    double const psi_t = u_t[1];
    f[0] = fl(tau_star(t), V, psi);
    f[1] = -psi_t + 1.0 - V * psi / L - (1.0 - V_star(t) * psi_star(t) / L) +
           dpsi_stardt(t);

    CHKERRQ(VecRestoreArray(F, &f));
    CHKERRQ(VecRestoreArrayRead(U_t, &u_t));
    CHKERRQ(VecRestoreArrayRead(U, &u));
    return 0;
}

PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec U_t, PetscReal shift,
                         Mat A, Mat P, void*) {
    PetscScalar const* u;
    CHKERRQ(VecGetArrayRead(U, &u));
    double const V = u[0];
    double const psi = u[1];
    CHKERRQ(VecRestoreArrayRead(U, &u));

    double J00 = dfldV(V, psi);
    double J01 = dfldpsi(V, psi);
    double J10 = -psi / L;
    double J11 = -shift - V / L;

    MatSetValue(A, 0, 0, J00, INSERT_VALUES);
    MatSetValue(A, 0, 1, J01, INSERT_VALUES);
    MatSetValue(A, 1, 0, J10, INSERT_VALUES);
    MatSetValue(A, 1, 1, J11, INSERT_VALUES);

    CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatCopy(A, P, SAME_NONZERO_PATTERN));
    return 0;
}

struct MonitorData {
    std::ofstream file;
    double V_err = 0.0;
    double psi_err = 0.0;
};

PetscErrorCode monitor(TS ts, PetscInt step, PetscReal time, Vec U, void* ctx) {
    auto dat = reinterpret_cast<MonitorData*>(ctx);
    PetscScalar const* u;
    CHKERRQ(VecGetArrayRead(U, &u));
    double V = u[0];
    double psi = u[1];
    double V_err = fabs(V_star(time) - V);
    double psi_err = fabs(psi_star(time) - psi);
    dat->file << time << "," << V << "," << psi << "," << V_err << ","
              << psi_err << std::endl;
    dat->V_err = std::max(V_err, dat->V_err);
    dat->psi_err = std::max(psi_err, dat->psi_err);
    CHKERRQ(VecRestoreArrayRead(U, &u));
    return 0;
}

int main(int argc, char* argv[]) {
    TS ts;
    Mat A;
    Vec x;
    PetscScalar* X;

    std::cout<<"start main_im.cpp"<<std::endl;

    double endTime = 3.0;
    if (argc >= 2 && argv[1][0] != '-') {
        endTime = atof(argv[1]);
    }

    CHKERRQ(PetscInitialize(&argc, &argv, nullptr, nullptr));

    CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD, 2, &x));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
    CHKERRQ(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
    CHKERRQ(MatSetFromOptions(A));
    CHKERRQ(MatSetUp(A));

    MonitorData dat;
    dat.file.open("result.csv");
    dat.file << "time,V,psi,V_err,psi_err" << std::endl;

    CHKERRQ(TSCreate(PETSC_COMM_WORLD, &ts));
    CHKERRQ(TSSetProblemType(ts, TS_NONLINEAR));
    CHKERRQ(TSSetSolution(ts, x));
    CHKERRQ(TSSetIFunction(ts, nullptr, IFunction, nullptr));
    CHKERRQ(TSSetIJacobian(ts, A, A, IJacobian, nullptr));
    CHKERRQ(TSSetMaxTime(ts, endTime));
    CHKERRQ(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    CHKERRQ(TSMonitorSet(ts, monitor, &dat, nullptr));

    CHKERRQ(TSSetFromOptions(ts));
    CHKERRQ(TSSetUp(ts));

    VecGetArray(x, &X);
    X[0] = V_star(0.0);
    X[1] = psi_star(0.0);
    VecRestoreArray(x, &X);

    CHKERRQ(TSSolve(ts, x));
    dat.file.close();

    std::cout << "V_err: " << dat.V_err << std::endl;
    std::cout << "psi_err: " << dat.psi_err << std::endl;

    CHKERRQ(TSDestroy(&ts));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(MatDestroy(&A));
    auto ierr = PetscFinalize();
    return ierr;
}
