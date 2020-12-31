#ifndef RATEANDSTATE_20201026_H
#define RATEANDSTATE_20201026_H

#include "config.h"
#include "tandem/RateAndStateBase.h"

#include "geometry/Vector.h"
#include "tensor/Reshape.h"
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
    static constexpr std::size_t PsiIndex = TangentialComponents;

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
    template <typename T> auto mat(Vector<T>& state) const {
        std::size_t nbf = space_.numBasisFunctions();
        return reshape(state, nbf, NumQuantities);
    }
    auto get_tau(std::size_t node, Matrix<double> const& traction) const {
        std::array<double, TangentialComponents> result;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            result[t] = traction(node, t + 1);
        }
        return result;
    }

    Law law_;
    std::optional<source_fun_t> source_;
};

template <class Law>
void RateAndState<Law>::pre_init(std::size_t faultNo, Vector<double>& state,
                                 LinearAllocator<double>&) const {

    auto s_mat = mat(state);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto Sini = law_.S_init(index + node);
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            s_mat(node, t) = Sini[t];
        }
    }
}

template <class Law>
void RateAndState<Law>::init(std::size_t faultNo, Matrix<double> const& traction,
                             Vector<double>& state, LinearAllocator<double>&) const {
    auto s_mat = mat(state);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        s_mat(node, PsiIndex) =
            law_.psi_init(index + node, traction(node, 0), get_tau(node, traction));
    }
}

// what about the last component in each block? what does it contain ??? 
template <class Law>
double RateAndState<Law>::rhs(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, Vector<double>& result,
                              LinearAllocator<double>&) const {
    double VMax = 0.0;
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    auto s_mat = mat(state);
    auto r_mat = mat(result);
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto psi = s_mat(node, PsiIndex);
        auto Vi = law_.slip_rate(index + node, sn, get_tau(node, traction), psi);
        double V = norm(Vi);
        VMax = std::max(VMax, V);
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            r_mat(node, t) = Vi[t];
        }
        r_mat(node, PsiIndex) = law_.state_rhs(index + node, V, psi);
    }
    if (source_) {
        auto coords = fault_[faultNo].template get<Coords>();
        std::array<double, DomainDimension + 1> xt;
        for (std::size_t node = 0; node < nbf; ++node) {
            auto const& x = coords[node];
            std::copy(x.begin(), x.end(), xt.begin());
            xt.back() = time;
            r_mat(node, PsiIndex) += (*source_)(xt)[0];
        }
    }
    return VMax;
}

template <class Law>
void RateAndState<Law>::getDerivativesDfDVAndDfDpsi(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& result, LinearAllocator<double>&) const {
    auto s_mat = mat(state);
    auto r_mat = mat(result);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto psi = s_mat(node, PsiIndex);
        auto tau = get_tau(node, traction);
        auto Vi = law_.slip_rate(index + node, sn, tau, psi);
        double V = norm(Vi);
        result(node) = law_.df_dV(index + node, sn, V, psi);
        result(nbf + node) = law_.df_dpsi(index + node, sn, V, psi);
        result(2 * nbf + node) = law_.dg_dpsi(index + node, psi, -result(node + nbf) / result(node));
    }
}               

template <class Law>
void RateAndState<Law>::state(std::size_t faultNo, Matrix<double> const& traction,
                              Vector<double const>& state, Matrix<double>& result,
                              LinearAllocator<double>&) const {
    auto s_mat = mat(state);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto tau = get_tau(node, traction);
        auto psi = s_mat(node, PsiIndex);
        auto V = law_.slip_rate(index + node, sn, tau, psi);
        auto tauAbs = law_.tau_pre(index + node) + tau;
        std::size_t out = 0;
        result(node, out++) = psi;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            result(node, out++) = s_mat(node, t);
        }
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            result(node, out++) = tauAbs[t];
        }
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            result(node, out++) = V[t];
        }
        result(node, out++) = law_.sn_pre(index + node) - sn;
    }
}

} // namespace tndm

#endif // RATEANDSTATE_20201026_H
