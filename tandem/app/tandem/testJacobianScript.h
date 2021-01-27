#ifndef SEAS_20200825_H
#define SEAS_20200825_H

#include "config.h"
#include "tandem/Config.h"

#include "config.h"
#include "localoperator/Elasticity.h"
#include "localoperator/Poisson.h"
#include "tandem/Config.h"
#include "tandem/DieterichRuinaAgeing.h"
#include "tandem/FrictionConfig.h"
#include "tandem/RateAndState.h"
#include "tandem/SeasElasticityAdapter.h"
#include "tandem/SeasOperator.h"
#include "tandem/SeasPoissonAdapter.h"
#include "tandem/SeasScenario.h"
#include "tandem/SeasWriter.h"

#include "form/DGOperatorTopo.h"
#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"

#include <mpi.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>


#include "mesh/LocalSimplexMesh.h"

namespace tndm {
    void testJacobianScript(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg);
}

namespace tndm::detail {
    using namespace Eigen;
    void writeBlockToEigen(PetscBlockVector& BlockVector, VectorXd& EigenVector);
    void writeEigenToBlock(VectorXd& EigenVector, PetscBlockVector& BlockVector);

    /**
     * Calculates the Jacobi matrix for the implicit Euler
     * @param J Jacobian of the rhs
     * @param dt timestep size
     */
    MatrixXd JacobianImplicitEuler(Mat& J, double dt);

    /********
     * right hand side of the implicit Euler method F = -x_n + x_n_1 + dt * f(x_n)
     * ******/
    VectorXd F(double dt, VectorXd& x, VectorXd& x_old, VectorXd rhs(VectorXd&));
}

#endif // SEAS_20200825_H
