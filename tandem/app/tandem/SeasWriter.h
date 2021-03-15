#ifndef SEASWRITER_20201006_H
#define SEASWRITER_20201006_H

#include "tandem/AdaptiveOutputStrategy.h"

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
    /**
     * Initialize the monitor
     * @param baseName name of the ouput file
     * @param mesh reference to the local simplex mesh (to set up the fault adapter)
     * @param cl pointer to the curvilinear description (to set up the fault adapter)
     * @param seasop pointer to the SEAS operator instance
     * @param degree ???
     * @param V_ref reference velocity
     * @param t_min minimum simulation time between two writing operations
     * @param t_max maximum simulation time between two writing operations
     * @param strategy method to calculate the interval between two writing operations (available: Threshold, Exponential)
     */
    SeasWriter(std::string_view baseName, LocalSimplexMesh<D> const& mesh,
               std::shared_ptr<Curvilinear<D>> cl, std::shared_ptr<SeasOperator> seasop,
               unsigned degree, double V_ref, double t_min, double t_max,
               AdaptiveOutputStrategy strategy = AdaptiveOutputStrategy::Threshold)
        : seasop_(std::move(seasop)), fault_adapter_(mesh, cl, seasop_->faultMap().fctNos()),
          adapter_(std::move(cl), seasop_->adapter().numLocalElements()), degree_(degree),
          fault_base_(baseName), base_(baseName), V_ref_(V_ref), t_min_(t_min), t_max_(t_max), 
          strategy_(strategy){
        

        timeAnalysis.open("timeAnalysis.csv");
        timeAnalysis << "time,Vmax,count_rhs,errorSrel,errorPSIrel,errorSabs,errorPSIabs,maxPSI,minPSI,maxS,minS,fmax,ratio_addition" << std::endl;

        fault_base_ += "-fault";
        MPI_Comm_rank(seasop_->comm(), &rank_);
    }

    /**
     * Destructor
     */
    ~SeasWriter(){
        timeAnalysis.close();
    }

    /**
     * calculate the interval between two writing operations
     * @param VMax current maximum velocity on the fault
     */
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

    /**
     * write operation
     * called after each timestep, only writes if the output interval is smaller 
     * than the passed simulated time since the last write operation
     */
    template <class BlockVector> void monitor(double time, BlockVector const& state) {

//        calculateMaxErrors(state);

        double Vmax = seasop_->VMax();
        double fmax = seasop_->fMax();
        auto solverStruct = seasop_->getSolverParameters();
        double ratio = solverStruct.ratio_addition;
        
        timeAnalysis <<std::setprecision(18)<< time << "," << Vmax << "," << 
        seasop_->rhs_count() << ","  << 
        errorSrel_ << "," << errorPSIrel_ << "," << errorSabs_ << "," << errorPSIabs_ << "," <<
        maxPSI_ << "," << minPSI_ << "," << maxS_ << "," << minS_ <<"," << fmax << "," << ratio << std::endl;
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
    /**
     * creates the file name of the step output file
     * @param base file name base string
     * @return string with the output file name
     */
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
        // // reset error values 
        // errorSrel_ = 0;
        // errorPSIrel_ = 0;
        // errorSabs_ = 0;
        // errorPSIabs_ = 0;
        // maxS_ = 0;
        // maxPSI_ = 0;
        // minS_ = 10;
        // minPSI_ = 1;

        // // create embeded solution vector
        // Vec embeddedSolution;

        // // obtain the order of the numerical scheme
        // int order;                                      
        // CHKERRTHROW(TSRKGetOrder(ts_.getTS(), &order)); // find a more elegant way that also works for non RK schemes

        // // evaluate X for the embedded method of lower order at the current time step
        // DM dm;
        // CHKERRTHROW(TSGetDM(ts_.getTS(), &dm));
        // CHKERRTHROW(DMGetGlobalVector(dm, &embeddedSolution));
        // CHKERRTHROW(TSEvaluateStep(ts_.getTS(), order-1, embeddedSolution, nullptr));

        // // get domain characteristics
        // int nbf = seasop_->lop().space().numBasisFunctions();
        // int PsiIndex = RateAndStateBase::TangentialComponents * nbf;

        // // initialize access to Petsc vectors
        // auto s = state.begin_access_readonly();
        // const double* e;
        // CHKERRTHROW(VecGetArrayRead(embeddedSolution, &e));     // can I avoid that with the DM ???

        // // calculate relative and absolute errors in S and PSI
        // for (int noFault = 0; noFault < seasop_->numLocalElements(); noFault++){
        //     auto eB = e + noFault * seasop_->block_size();
        //     auto sB = state.get_block(s, noFault);

        //     for (int component = 0; component < RateAndStateBase::TangentialComponents; component++){
        //         for (int node = 0; node < nbf; node++){
        //             int i = component * nbf + node;
        //             maxS_ = std::max(maxS_,sB(i));
        //             minS_ = std::min(minS_,sB(i));
        //             errorSabs_ = std::max(errorSabs_, abs(sB(i) - eB[i]));
        //             errorSrel_ = std::max(errorSrel_, 
        //                 (sB(i) != 0) ? abs((sB(i) - eB[i]) / sB(i)) : 0);
        //         }
        //     }
        //     for (int node = 0; node < nbf; node++){
        //         int i = PsiIndex + node;
        //         maxPSI_ = std::max(maxPSI_,sB(i));
        //         minPSI_ = std::min(minPSI_,sB(i));
        //         errorPSIabs_ = std::max(errorPSIabs_, abs(sB(i) - eB[i]));
        //         errorPSIrel_ = std::max(errorPSIrel_, 
        //             (sB(i) != 0) ? abs((sB(i) - eB[i]) / sB(i)) : 0);
        //     }
        // }

        // // restore access to Petsc vectors
        // CHKERRTHROW(VecRestoreArrayRead(embeddedSolution, &e));
        // state.end_access_readonly(s);
        // CHKERRTHROW(DMRestoreGlobalVector(dm, &embeddedSolution));

    } 

    // geometric parameters   
    std::shared_ptr<SeasOperator> seasop_;
    CurvilinearBoundaryVTUAdapter<D> fault_adapter_;
    CurvilinearVTUAdapter<D> adapter_;
    PVDWriter pvd_;
    PVDWriter pvd_fault_;
    int rank_;

    // file name of the csv output file
    std::ofstream timeAnalysis;

    // output parametes
    std::string fault_base_;
    std::string base_;
    unsigned degree_;
    std::size_t output_step_ = 0;
    double last_output_time_ = std::numeric_limits<double>::lowest();

    // adaptive output writing parameters
    double V_ref_;
    double t_min_;
    double t_max_;
    double maxS_;
    double minS_;
    AdaptiveOutputStrategy strategy_;

    // error evaluation parameters
    double errorSrel_;    
    double errorPSIrel_;    
    double errorSabs_;    
    double errorPSIabs_;    
    double maxPSI_;
    double minPSI_;
};

} // namespace tndm

#endif // SEASWRITER_20201006_H
