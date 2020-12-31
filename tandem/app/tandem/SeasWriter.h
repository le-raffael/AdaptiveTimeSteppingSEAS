#ifndef SEASWRITER_20201006_H
#define SEASWRITER_20201006_H

#include "tandem/AdaptiveOutputStrategy.h"

#include "common/PetscTimeSolver.h"
#include "petscts.h"  
#include "petscdm.h" 
#include "common/PetscBlockVector.h"

#include "geometry/Curvilinear.h"
#include "io/PVDWriter.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "mesh/LocalSimplexMesh.h"

#include <mpi.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace tndm {

template <std::size_t D, class SeasOperator> class SeasWriter {
public:
    SeasWriter(std::string_view baseName, LocalSimplexMesh<D> const& mesh,
               std::shared_ptr<Curvilinear<D>> cl, std::shared_ptr<SeasOperator> seasop,
               unsigned degree, double V_ref, double t_min, double t_max, PetscTimeSolver& ts,
               AdaptiveOutputStrategy strategy = AdaptiveOutputStrategy::Threshold)
        : seasop_(std::move(seasop)), fault_adapter_(mesh, cl, seasop_->faultMap().fctNos()),
          adapter_(std::move(cl), seasop_->adapter().numLocalElements()), degree_(degree),
          fault_base_(baseName), base_(baseName), V_ref_(V_ref), t_min_(t_min), t_max_(t_max), 
          ts_(ts), strategy_(strategy){
        

        timeAnalysis.open("timeAnalysis.csv");
        timeAnalysis << "time,Vmax,count_rhs,errorVrel,errorPSIrel,errorVabs,errorPSIabs" << std::endl;

        fault_base_ += "-fault";
        MPI_Comm_rank(seasop_->comm(), &rank_);
    }

    ~SeasWriter(){
        timeAnalysis.close();
    }

    double output_interval(double VMax) const {
        double interval = 0.0;
        switch (strategy_) {
        case AdaptiveOutputStrategy::Threshold:
            interval = VMax >= V_ref_ ? t_min_ : t_max_;
            break;
        case AdaptiveOutputStrategy::Exponential: {
            double falloff = log(t_min_ / t_max_);
            VMax = std::min(V_ref_, VMax);
            interval = t_max_ * exp(falloff * VMax / V_ref_);
            break;
        }
        default:
            break;
        }
        return interval;
    }

    template <class BlockVector> void monitor(double time, BlockVector const& state) {

        calculateMaxErrors(state);

        double Vmax = seasop_->VMax();
        timeAnalysis <<std::setprecision(18)<< time << "," << Vmax << "," << 
        seasop_->rhs_count() << ","  << 
        errorVrel_ << "," << errorPSIrel_ << "," << errorVabs_ << "," << errorPSIabs_ << std::endl;
        seasop_->reset_rhs_count();
        
        auto interval = output_interval(seasop_->VMax());
        if (time - last_output_time_ >= interval) {
            auto fault_writer = VTUWriter<D - 1u>(degree_, true, seasop_->comm());
            fault_writer.addFieldData("time", &time, 1);
            auto fault_piece = fault_writer.addPiece(fault_adapter_);
            fault_piece.addPointData("state", seasop_->state(state));
            auto fault_base_step = name(fault_base_);
            fault_writer.write(fault_base_step);
            if (rank_ == 0) {
                pvd_fault_.addTimestep(time, fault_writer.pvtuFileName(fault_base_step));
                pvd_fault_.write(fault_base_);
            }

            auto displacement = seasop_->adapter().displacement();
            auto writer = VTUWriter<D>(degree_, true, seasop_->comm());
            writer.addFieldData("time", &time, 1);
            auto piece = writer.addPiece(adapter_);
            piece.addPointData("u", displacement);
            auto base_step = name(base_);
            writer.write(base_step);
            if (rank_ == 0) {
                pvd_.addTimestep(time, writer.pvtuFileName(base_step));
                pvd_.write(base_);
            }

            ++output_step_;
            last_output_time_ = time;
        }
    }

private:
    std::string name(std::string const& base) const {
        std::stringstream ss;
        ss << base << "_" << output_step_;
        return ss.str();
    }

    /**
     * Calculate the maximum relative and absolute error in V and PSI from the built-in error estimate
     * @param state current state of the system
     */
    template <class BlockVector> void calculateMaxErrors( BlockVector const& state){
        // reset error values 
        errorVrel_ = 0;
        errorPSIrel_ = 0;
        errorVabs_ = 0;
        errorPSIabs_ = 0;

        // create embeded solution vector
        Vec embeddedSolution;

        // obtain the order of the numerical scheme
        int order;                                      
        CHKERRTHROW(TSRKGetOrder(ts_.getTS(), &order)); // find a more elegant way that also works for non RK schemes

        // evaluate X for the embedded method of lower order at the current time step
        DM dm;
        CHKERRTHROW(TSGetDM(ts_.getTS(), &dm));
        CHKERRTHROW(DMGetGlobalVector(dm, &embeddedSolution));
        CHKERRTHROW(TSEvaluateStep(ts_.getTS(), order-1, embeddedSolution, nullptr));

        // get domain characteristics
        int nbf = seasop_->lop().space().numBasisFunctions();
        int PsiIndex = RateAndStateBase::TangentialComponents * nbf;

        // initialize access to Petsc vectors
        auto s = state.begin_access_readonly();
        const double* e;
        CHKERRTHROW(VecGetArrayRead(embeddedSolution, &e));     // can I avoid that with the DM ???

        // calculate relative and absolute errors in V and PSI
        for (int noFault = 0; noFault < seasop_->numLocalElements(); noFault++){
            auto eB = e + noFault * seasop_->block_size();
            auto sB = state.get_block(s, noFault);

            for (int component = 0; component < RateAndStateBase::TangentialComponents; component++){
                for (int node = 0; node < nbf; node++){
                    int i = component * nbf + node;
                    errorVabs_ = std::max(errorVabs_, abs(sB(i) - eB[i]));
                    errorVrel_ = std::max(errorVrel_, 
                        (sB(i) != 0) ? abs((sB(i) - eB[i]) / sB(i)) : 0);
                }
            }
            for (int node = 0; node < nbf; node++){
                int i = PsiIndex + node;
                errorPSIabs_ = std::max(errorPSIabs_, abs(sB(i) - eB[i]));
                errorPSIrel_ = std::max(errorPSIrel_, 
                    (sB(i) != 0) ? abs((sB(i) - eB[i]) / sB(i)) : 0);
            }
        }

        // restore access to Petsc vectors
        CHKERRTHROW(VecRestoreArrayRead(embeddedSolution, &e));
        state.end_access_readonly(s);
        CHKERRTHROW(DMRestoreGlobalVector(dm, &embeddedSolution));

    } 
    
    std::shared_ptr<SeasOperator> seasop_;
    CurvilinearBoundaryVTUAdapter<D> fault_adapter_;
    CurvilinearVTUAdapter<D> adapter_;
    PVDWriter pvd_;
    PVDWriter pvd_fault_;
    int rank_;

    std::ofstream timeAnalysis;

    std::string fault_base_;
    std::string base_;
    unsigned degree_;
    std::size_t output_step_ = 0;
    double last_output_time_ = std::numeric_limits<double>::lowest();

    double V_ref_;
    double t_min_;
    double t_max_;
    AdaptiveOutputStrategy strategy_;

    PetscTimeSolver& ts_;
    double errorVrel_;    
    double errorPSIrel_;    
    double errorVabs_;    
    double errorPSIabs_;    
};

} // namespace tndm

#endif // SEASWRITER_20201006_H
