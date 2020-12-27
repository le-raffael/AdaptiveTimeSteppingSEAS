#ifndef DIETERICHRUINAAGEING_20201027_H
#define DIETERICHRUINAAGEING_20201027_H

#include "RateAndState.h"

#include "util/Zero.h"

#include <algorithm>
#include <cassert>
#include <iostream>

namespace tndm {

class DieterichRuinaAgeing {
public:
    struct ConstantParams {
        double V0;
        double b;
        double L;
        double f0;
    };
    struct Params {
        double a;
        double eta;
        double sn_pre;
        double tau_pre;
        double Vinit;
        double Sinit;
    };

    /**
     * resize the local parameter array
     * @param num_nodes number of nodes on the fault
     */
    void set_num_nodes(std::size_t num_nodes) { p_.resize(num_nodes); }

    /**
     * Set global parameters
     * @param params parameters
     */
    void set_constant_params(ConstantParams const& params) { cp_ = params; }

    /**
     * Set local parameters
     * @param index of the current node
     * @param params local parameters at this node
     */
    void set_params(std::size_t index, Params const& params) {
        p_[index].get<A>() = params.a;
        p_[index].get<Eta>() = params.eta;
        p_[index].get<SnPre>() = params.sn_pre;
        p_[index].get<TauPre>() = params.tau_pre;
        p_[index].get<Vinit>() = params.Vinit;
        p_[index].get<Sinit>() = params.Sinit;
    }

    /**
     * Initialize the state variable
     * @param index of the current node
     * @param sn sigma_n parameter at current node
     * @param tau tau parameter at current node
     * @return psi_init at the current node
     */
    double psi_init(std::size_t index, double sn, double tau) const {
        double snAbs = sn + p_[index].get<SnPre>();
        auto tauAbs = tau + p_[index].get<TauPre>();
        auto Vi = p_[index].get<Vinit>();
        auto a = p_[index].get<A>();
        auto eta = p_[index].get<Eta>();
        double s = sinh((tauAbs - eta * Vi) / (a * snAbs));
        double l = log((2.0 * cp_.V0 / Vi) * s);
        return a * l;
    }

    /**
     * get sn_pre
     * @param index of the current node
     * @return sn_pre at this node
     * */
    double sn_pre(std::size_t index) const { return p_[index].get<SnPre>(); }

    /**
     * get tau_pre
     * @param index of the current node
     * @return tau_pre at this node
     * */
    double tau_pre(std::size_t index) const { return p_[index].get<TauPre>(); }


    /**
     * get initial slip
     * @param index of the current node
     * @return slip at this node
     * */
    double S_init(std::size_t index) const { return p_[index].get<Sinit>(); }

    /**
     * calculate the derivative dg/dpsi of the algebraic equation g(V,psi)=0 w.r.t to the state variable
     * @param index of the current node
     * @param psi state variable at the current node
     * @param dV_dpsi the derivative dV/dPSI
     * @return deriviative dg/dpsi
     * */
    double dg_dpsi(std::size_t index, double psi, double dV_dpsi) const {
        return -cp_.V0 / cp_.L * (exp((cp_.f0 - psi) / cp_.b) - dV_dpsi / cp_.V0);
    }

    /**
     * calculate the derivative df/dpsi of the algebraic equation f(V,psi)=0 w.r.t to the state variable
     * @param index of the current node
     * @param sn at the current node
     * @param V slip rate at the current node
     * @param psi state variable at the current node
     * @return deriviative df/dpsi
     * */
    double df_dpsi(std::size_t index, double sn, double V, double psi) const {
        auto eta = p_[index].get<Eta>();
        auto a = p_[index].get<A>();
        double snAbs = sn + p_[index].get<SnPre>();
        return -snAbs * asinh(V / (2.0 * cp_.V0)) * exp(psi / a) - eta;
    }

    /**
     * calculate the derivative df/dV of the algebraic equation f(V,psi)=0 w.r.t to the slip rate
     * @param index of the current node
     * @param sn at the current node
     * @param V slip rate at the current node
     * @param psi state variable at the current node
     * @return deriviative df/dV
     * */
    double df_dV(std::size_t index, double sn, double V, double psi) const {
        auto a = p_[index].get<A>();
        double snAbs = sn + p_[index].get<SnPre>();
        double twoV0 = 2.0 * cp_.V0;
        return -snAbs * a / (twoV0 * sqrt(V * V / (twoV0 * twoV0) + 1)) * exp(psi / a);
    }

    /**
     * Solve the algebraic equation f(V,psi) = 0 for the slip rate
     * @param index of the current node
     * @param sn at the current node
     * @param tau at the current node
     * @param psi state variable at the current node
     * @return slip rate
     * */
    double slip_rate(std::size_t index, double sn, double tau, double psi) const {
        auto eta = p_[index].get<Eta>();
        double tauAbs = tau + p_[index].get<TauPre>();
        double a = 0.0;
        double b = tauAbs / eta;
        if (a > b) {
            std::swap(a, b);
        }
        auto fF = [this, &index, &sn, &tau, &psi](double V) {
            return this->F(index, sn, tau, V, psi);
        };
        return zeroIn(a, b, fF);
    }

    /**
     * Evaluate the rhs of the state variable dpsi/dt = g(psi, V)
     * @param index of the current node
     * @param V slip rate at the current node
     * @param psi state variable at the current node
     * @return rhs of the state variable ODE
     * */
    double state_rhs(std::size_t index, double V, double psi) const {
        return cp_.b * cp_.V0 / cp_.L * (exp((cp_.f0 - psi) / cp_.b) - V / cp_.V0);
    }

    /**
     * get the environment velocity V_0
     * @return the velocity
     */
    double getV0() const {return cp_.V0;}

private:
    /**
     * Evaluate the algebraic function f(psi,V)
     * @param index of the current node
     * @param sn at the current node
     * @param tau at the current node
     * @param V slip rate at the current node
     * @param psi state variable at the current node
     * @return value of f(psi,V) for the given parameters
     * */
    double F(std::size_t index, double sn, double tau, double V, double psi) const {
        double snAbs = sn + p_[index].get<SnPre>();
        double tauAbs = tau + p_[index].get<TauPre>();
        auto a = p_[index].get<A>();
        auto eta = p_[index].get<Eta>();
        double e = exp(psi / a);
        double f = a * asinh((V / (2.0 * cp_.V0)) * e);
        return tauAbs - snAbs * f - eta * V;
    }

    ConstantParams cp_;

    struct SnPre {
        using type = double;
    };
    struct TauPre {
        using type = double;
    };
    struct A {
        using type = double;
    };
    struct Eta {
        using type = double;
    };
    struct Vinit {
        using type = double;
    };
    struct Sinit {
        using type = double;
    };
    mneme::MultiStorage<mneme::DataLayout::SoA, SnPre, TauPre, A, Eta, Vinit, Sinit> p_;
};

} // namespace tndm

#endif // DIETERICHRUINAAGEING_20201027_H
