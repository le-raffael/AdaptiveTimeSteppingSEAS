#ifndef RATEANDSTATE_20201026_H
#define RATEANDSTATE_20201026_H

#include "config.h"
#include "tandem/RateAndStateBase.h"

#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <iostream>

namespace tndm {

template <class Law> class RateAndState : public RateAndStateBase {
public:
    using RateAndStateBase::RateAndStateBase;

    using param_fun_t =
        std::function<typename Law::Params(std::array<double, DomainDimension> const&)>;
    using source_fun_t =
        std::function<std::array<double, 1>(std::array<double, DomainDimension + 1> const&)>;

    /**
     * Initialize global parameters (V0, b, L, f0)
     * @param cps parameters
     */
    void set_constant_params(typename Law::ConstantParams const& cps) {
        law_.set_constant_params(cps);
    }

    /**
     * Initialize local parameters (a, eta, sn_pre, tau_pre, V_init, S_init)
     */
    void set_params(param_fun_t pfun) {
        auto num_nodes = fault_.storage().size();
        law_.set_num_nodes(num_nodes);
        for (std::size_t index = 0; index < num_nodes; ++index) {
            auto params = pfun(fault_.storage()[index].template get<Coords>());
            law_.set_params(index, params);
        }
    }

    /**
     * Set source functional??
     * @param source functional
     */
    void set_source_fun(source_fun_t source) { source_ = std::make_optional(std::move(source)); }

    /**
     * Set initial slip to the state vector
     * @param faultNo index of th curent fault
     * @param state current solution vector
     * @param . some scratch
     */
    void pre_init(std::size_t faultNo, Vector<double>& state, LinearAllocator<double>&) const;


    /**
     * Set initial state variable
     * @param faultNo index of the current fault
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state current solution vector
     * @param . some scratch
     * */
    void init(std::size_t faultNo, Matrix<double> const& traction, Vector<double>& state,
              LinearAllocator<double>&) const;

    /**
     * Evaluate the rhs of the ODE 
     * - first solve the algebraic equation f(V, psi) = 0 for the slip rate
     * - assign the derivative of dS_dt = V
     * - assign the derivative of dpsi_dt = g(V, psi)
     * @param faultNo index of the current fault
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state current solution vector
     * @param result vector with the right hand side of the ODE dstate_dt
     * @param . some scratch
     * @return the maximal velocity encountered in this element
     * */
    double rhs(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& result, LinearAllocator<double>&) const;

    /**
     * Evaluate some derivatives for the Jacobian 
     * - assign the derivative of df/dV
     * - assign the derivative of df/dpsi
     * @param faultNo index of the current fault
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state current solution vector
     * @param result vector with the derivatives df/dV and df/dpsi in the order [nbf, nbf]
     * @param . some scratch
     * */
    void getDerivativesDfDVAndDfDpsi(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& result, LinearAllocator<double>&) const;



    /**
     * Extract some values from the current state
     * @param faultNo index of the current fault
     * @param state current solution vector
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param result matrix with rows: nbf and cols: [psi, S, tau, V, sn]
     * @param . some scratch
     */
    void state(std::size_t faultNo, Matrix<double> const& traction, Vector<double const>& state,
               Matrix<double>& result, LinearAllocator<double>&) const;

    /**
     * get the ageing law
     * @return law 
     */
    Law& getLaw(){return law_;}

private:
    Law law_;
    std::optional<source_fun_t> source_;
};

template <class Law>
void RateAndState<Law>::pre_init(std::size_t faultNo, Vector<double>& state,
                                 LinearAllocator<double>&) const {

    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        state(node) = law_.S_init(index + node);
    }
}

template <class Law>
void RateAndState<Law>::init(std::size_t faultNo, Matrix<double> const& traction,
                             Vector<double>& state, LinearAllocator<double>&) const {

    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        state(nbf + node) = law_.psi_init(index + node, traction(node, 0), traction(node, 1));
    }
}

template <class Law>
double RateAndState<Law>::rhs(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, Vector<double>& result,
                              LinearAllocator<double>&) const {
    double VMax = 0.0;
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto tau = traction(node, 1);
        auto psi = state(nbf + node);
        double V = law_.slip_rate(index + node, sn, tau, psi);
        VMax = std::max(VMax, std::fabs(V));
        result(node) = V;
        result(nbf + node) = law_.state_rhs(index + node, V, psi);
    }
    if (source_) {
        auto coords = fault_[faultNo].template get<Coords>();
        std::array<double, DomainDimension + 1> xt;
        for (std::size_t node = 0; node < nbf; ++node) {
            auto const& x = coords[node];
            std::copy(x.begin(), x.end(), xt.begin());
            xt.back() = time;
            result(nbf + node) += (*source_)(xt)[0];
        }
    }
    return VMax;
}

template <class Law>
void RateAndState<Law>::getDerivativesDfDVAndDfDpsi(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& result, LinearAllocator<double>&) const {
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto tau = traction(node, 1);
        auto psi = state(nbf + node);
        double V = law_.slip_rate(index + node, sn, tau, psi);
        result(node) = law_.df_dV(index + node, sn, V, psi);
        result(nbf + node) = law_.df_dpsi(index + node, sn, V, psi);
        result(2 * nbf + node) = law_.dg_dpsi(index + node, psi, -result(node + nbf) / result(node));
    }
}               

template <class Law>
void RateAndState<Law>::state(std::size_t faultNo, Matrix<double> const& traction,
                              Vector<double const>& state, Matrix<double>& result,
                              LinearAllocator<double>&) const {
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto tau = traction(node, 1);
        auto psi = state(nbf + node);
        double V = law_.slip_rate(index + node, sn, tau, psi);
        result(node, 0) = psi;
        result(node, 1) = state(node);
        result(node, 2) = law_.tau_pre(index + node) + tau;
        result(node, 3) = V;
        result(node, 4) = law_.sn_pre(index + node) + sn;
    }
}

} // namespace tndm

#endif // RATEANDSTATE_20201026_H
