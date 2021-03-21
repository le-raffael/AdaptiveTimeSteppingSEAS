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
    static constexpr std::size_t TangentialComponents = DomainDimension - 1u;
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
        std::array<double, TangentialComponents> tau_pre;
        std::array<double, TangentialComponents> Vinit;
        std::array<double, TangentialComponents> Sinit;
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
    double psi_init(std::size_t index, double sn,
                    std::array<double, TangentialComponents> const& tau) const {
        double snAbs = -sn + p_[index].get<SnPre>();
        double tauAbs = norm(tau + p_[index].get<TauPre>());
        auto Vi = norm(p_[index].get<Vinit>());
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
    auto tau_pre(std::size_t index) const { return p_[index].get<TauPre>(); }

    /**
     * get initial slip
     * @param index of the current node
     * @return slip at this node
     * */
    auto S_init(std::size_t index) const { return p_[index].get<Sinit>(); }



    /**
     * calculate the partial derivative dg/dpsi of g(V,psi) w.r.t to the state variable
     * @param psi state variable at the current node
     * @return deriviative dg/dpsi
     * */
    double dg_dpsi(double psi) const {
        return -cp_.V0 / cp_.L * exp((cp_.f0 - psi) / cp_.b);
    }

    /**
     * calculate the partial derivative dg/dV of g(V,psi)=0 w.r.t to the slip rate
+     * @return deriviative dg/dV
     * */
    double dg_dV() const {
        return -cp_.b / cp_.L;
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
        auto a = p_[index].get<A>();
        double snAbs = sn + p_[index].get<SnPre>();
        double twoV0 = 2.0 * cp_.V0;
        double e = exp(psi / a);
        return -snAbs * V * e / (twoV0 * sqrt((V * e / twoV0) * (V * e / twoV0) + 1));
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
        auto eta = p_[index].get<Eta>();
        double snAbs = sn + p_[index].get<SnPre>();
        double twoV0 = 2.0 * cp_.V0;
        double e = exp(psi / a);
        return -snAbs * a * e / (twoV0 * sqrt((V * e / twoV0) * (V * e / twoV0) + 1)) - eta;
    }


    /**
     * Calculate the derivative dxi/dpsi needed to set up the derivative dh/dpsi in the 2nd order ODE
     * @param index of the current node
     * @param sn at the current node
     * @param V slip rate at the current node
     * @param psi state variable at the current node
     * @return deriviative dxi/dpsi
     */
    double dxi_dpsi(std::size_t index, double sn, double V, double psi) const {
        auto a = p_[index].get<A>();
        auto eta = p_[index].get<Eta>();
        double snAbs = sn + p_[index].get<SnPre>();
        double twoV0 = 2.0 * cp_.V0;
        double p = sqrt(exp(2. * psi / a) * V * V / (twoV0 * twoV0) + 1.);
        return       twoV0 * snAbs * exp(psi/a) / 
          (p * pow(twoV0 * eta * p + a * snAbs * exp(psi/a), 2));
    }

    /**
     * Calculate the derivative dxi/dV needed to set up the derivative dh/dV in the 2nd order ODE
     * @param index of the current node
     * @param sn at the current node
     * @param V slip rate at the current node
     * @param psi state variable at the current node
     * @return deriviative dxi/dV
     */
    double dxi_dV(std::size_t index, double sn, double V, double psi) const {
        auto a = p_[index].get<A>();
        auto eta = p_[index].get<Eta>();
        double snAbs = sn + p_[index].get<SnPre>();
        double twoV0 = 2.0 * cp_.V0;
        double p = sqrt(exp(2. * psi / a) * V * V / (twoV0 * twoV0) + 1.);
        return       -a * snAbs * V * exp(3.*psi/a) / 
          (twoV0 * p * pow(twoV0 * eta * p + a * snAbs * exp(psi/a), 2));
    }

    /**
     * Calculate the derivative dzeta/dpsi needed to set up the derivative dh/dpsi in the 2nd order ODE
     * @param index of the current node
     * @param sn at the current node
     * @param V slip rate at the current node
     * @param psi state variable at the current node
     * @return deriviative dzeta/dpsi
     */
    double dzeta_dpsi(std::size_t index, double sn, double V, double psi) const {
        auto a = p_[index].get<A>();
        auto eta = p_[index].get<Eta>();
        double snAbs = sn + p_[index].get<SnPre>();
        double twoV0 = 2.0 * cp_.V0;
        double p = sqrt(exp(2. * psi / a) * V * V / (twoV0 * twoV0) + 1.);
        return -snAbs * V * exp(psi/a) / 
               (a * twoV0 * pow(p, 3));
    }

    /**
     * Calculate the derivative dzeta/dV needed to set up the derivative dh/dV in the 2nd order ODE
     * @param index of the current node
     * @param sn at the current node
     * @param V slip rate at the current node
     * @param psi state variable at the current node
     * @return deriviative dzeta/dV
     */
    double dzeta_dV(std::size_t index, double sn, double V, double psi) const {
        auto a = p_[index].get<A>();
        auto eta = p_[index].get<Eta>();
        double snAbs = sn + p_[index].get<SnPre>();
        double twoV0 = 2.0 * cp_.V0;
        double p = sqrt(exp(2. * psi / a) * V * V / (twoV0 * twoV0) + 1.);
        return -snAbs * exp(psi/a) / 
               (twoV0 * pow(p, 3));
    }

    /**
     * Evaluate the friction law for given S, psi and V
     * @param index of the current node
     * @param sn at the current node - Scalar
     * @param tau at the current node - Vector
     * @param psi state variable at the current node - Scalar
     * @param V slip rate at the current node - Scalar
     * @return evaluation of the friction law - Scalar
     * */
    double friction_law(std::size_t index, double sn, 
                   std::array<double, TangentialComponents> const& tau, double psi, double V) const {
        auto eta = p_[index].get<Eta>();
        auto tauAbsVec = tau + p_[index].get<TauPre>();
        double snAbs = -sn + p_[index].get<SnPre>();
        double tauAbs = norm(tauAbsVec);
        return tauAbs - this->F(index, snAbs, V, psi) - eta * V; 
   }

    /**
     * Evaluate the friction law for given S, psi and V
     * @param index of the current node
     * @param tau at the current node - Vector
     * @return get the vector expression of tau - Scalar
     * */
    std::array<double, TangentialComponents> getTauVec(std::size_t index, std::array<double, TangentialComponents> const& tau) const {
        return tau + p_[index].get<TauPre>();
   }


    /**
     * Solve the algebraic equation f(V,psi) = 0 for the slip rate
     * @param index of the current node
     * @param sn at the current node
     * @param tau at the current node
     * @param psi state variable at the current node
     * @return slip rate
     * */
    auto slip_rate(std::size_t index, double sn, 
                   std::array<double, TangentialComponents> const& tau, double psi) const
         -> std::array<double, TangentialComponents> {
        auto eta = p_[index].get<Eta>();
        auto tauAbsVec = tau + p_[index].get<TauPre>();
        double snAbs = -sn + p_[index].get<SnPre>();
        double tauAbs = norm(tauAbsVec);
        double a = 0.0;
        double b = tauAbs / eta;
        if (a > b) {
            std::swap(a, b);
        }
        auto fF = [this, &index, &snAbs, &tauAbs, &psi, &eta](double V) {
            return tauAbs - this->F(index, snAbs, V, psi) - eta * V;
        };
        double V = zeroIn(a, b, fF);
        return (V / (F(index, snAbs, V, psi) + eta * V)) * tauAbsVec; // == V * tauAbsVec / tauAbs
    }

    /**
     * Evaluate the rhs of the state variable dpsi/dt = g(psi, V)
     * @param index of the current node
     * @param V norm of the slip rate at the current node
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

    /**
     * get the characteristic length L
     * @return the velocity
     */
    double getL() const {return cp_.L;}


    /**
     * evaluate the denominator of the ratio between absolute errors in slip and state variable
     * @param sn maximal value
     * @param psi_max maximal value
     * @param V_max maximal value
     */
    double evaluateErrorRatioDenominator(double sn, double psi_max, double V_max) const {
        double snAbs = -sn + p_[0].get<SnPre>();
        double a_min = 0.01;    // from Lua file
        V_max = 1e-6;
        std::cout << "sn: "<<snAbs<<", a: "<<p_[0].get<A>()<<", V0: "<<cp_.V0<<", asinh: "<<asinh(V_max / (2.0 * cp_.V0))<<std::endl;
        return abs(snAbs * asinh(V_max / (2.0 * cp_.V0)) * exp(psi_max / a_min));
    }

private:
    /**
     * Evaluate the algebraic function f(psi,V)
     * @param index of the current node
     * @param sn at the current node
     * @param V slip rate at the current node
     * @param psi state variable at the current node
     * @return value of f(psi,V) for the given parameters
     * */
    double F(std::size_t index, double sn, double V, double psi) const {
        auto a = p_[index].get<A>();
        double e = exp(psi / a);
        double f = a * asinh((V / (2.0 * cp_.V0)) * e);
        return sn * f;
    }

    ConstantParams cp_;

    struct SnPre {
        using type = double;
    };
    struct TauPre {
        using type = std::array<double, TangentialComponents>;
    };
    struct A {
        using type = double;
    };
    struct Eta {
        using type = double;
    };
    struct Vinit {
        using type = std::array<double, TangentialComponents>;
    };
    struct Sinit {
        using type = std::array<double, TangentialComponents>;
    };
    mneme::MultiStorage<mneme::DataLayout::SoA, SnPre, TauPre, A, Eta, Vinit, Sinit> p_;
};

} // namespace tndm

#endif // DIETERICHRUINAAGEING_20201027_H
