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
    static constexpr std::size_t VIndex = TangentialComponents + 1u;

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
    void initCompact(std::size_t faultNo, Matrix<double> const& traction, Vector<double>& state,
              LinearAllocator<double>&) const;

    /**
     * copy and set slip rate
     * @param faultNo index of the current fault
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state_small compact solution vector
     * @param state_big extended solution vector
     * @param . some scratch
     * */
    void makeBig(std::size_t faultNo, Matrix<double> const& traction, Vector<double>& state_small,
              Vector<double>& state_big, LinearAllocator<double>&) const;

    /**
     * just copy
     * @param faultNo index of the current fault
     * @param state_small compact solution vector
     * @param state_big extended solution vector
     * */
    void makeSmall(std::size_t faultNo, Vector<double>& state_small,
              Vector<double>& state_big) const;


    /**
     * Evaluate the rhs of the compact ODE 
     * - first solve the algebraic equation f(V, psi) = 0 for the slip rate
     * - assign the derivative of dS_dt = V
     * - assign the derivative of dpsi_dt = g(V, psi)
     * @param faultNo index of the current fault
     * @param time current simulation time 
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state current solution vector
     * @param result vector with the right hand side of the ODE dstate_dt
     * @param . some scratch
     * @return the maximal velocity encountered in this element
     * */
    double rhsCompactODE(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& result, LinearAllocator<double>&) const;


    /**
     * Evaluate the rhs of the extended ODE 
     * - assign the derivative dS_dt = V
     * - assign the derivative dpsi_dt = g(V, psi)
     * - the derivative dV/dt will be calculated in a later step
     * @param faultNo index of the current fault
     * @param time current simulation time 
     * @param state current solution vector
     * @param result vector with the right hand side of the ODE dstate_dt
     * @param . some scratch
     * @param checkAS check whether the simulation is in the aseismic slip or in the earthquake
     * @return the maximal velocity encountered in this element
     * */
    double rhsExtendedODE(std::size_t faultNo, double time,
               Vector<double const>& state, Vector<double>& result, LinearAllocator<double>&, bool checkAS) const;

    /**
     * Evaluate the lhs of the compact DAE formulation 
     * @param faultNo index of the current fault
     * @param time current simulation time 
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state current solution vector 
     * @param state_der derivative of the current solution vector
     * @param result vector with the algebraic equation that has to be set to 0
     * @param . some scratch
     * @return the maximal velocity encountered in this element
     * */
    double lhsCompactDAE(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, Vector<double const>& state_der, Vector<double>& result,
                              LinearAllocator<double>&) const;               

    /**
     * Evaluate the lhs of the extended DAE formulation 
     * @param faultNo index of the current fault
     * @param time current simulation time 
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state current solution vector 
     * @param state_der derivative of the current solution vector
     * @param result vector with the algebraic equation that has to be set to 0
     * @param . some scratch
     * @return the maximal velocity encountered in this element
     * */
    double lhsExtendedDAE(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, Vector<double const>& state_der, Vector<double>& result,
                              LinearAllocator<double>&) const;               

    /**
     * Evaluate the friction law for the DAE formulation 
     * @param faultNo index of the current fault
     * @param time current simulation time 
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state current solution vector [S,psi,:] needed
     * @param state_der derivative of the current solution vector [V,:] needed
     * @param result vector with the evaluation of the algebraic equation
     * @param . some scratch
     * */
    void applyFrictionLaw(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, Vector<double>& state_der, Vector<double>& result,
                              LinearAllocator<double>&) const;

    /**
     * Evaluate the friction law for the extended ODE formulation 
     * @param faultNo index of the current fault
     * @param time current simulation time 
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state current solution vector [S,psi,V] needed
     * @param . some scratch
     * @return the maximal absolute value of the friction law
     * */
    double applyMaxFrictionLaw(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, LinearAllocator<double>&) const;


    /**
     * Evaluate some derivatives for the Jacobian 
     * - assign the derivative of df/dV
     * - assign the derivative of df/dpsi
     * - assign the derivative of dg/dV
     * - assign the derivative of dg/dpsi
     * @param faultNo index of the current fault
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state current solution vector [S,psi] needed
     * @param state_der derivatives of the current solution vector [V,:] needed
     * @param result vector with the derivatives df/dV, df/dpsi, psi, V, g 
     * @param . some scratch
     * */
    void getJacobianQuantitiesCompact(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double const>& state_der, Vector<double>& result, LinearAllocator<double>&) const;
    void getJacobianQuantitiesCompact(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& state_der, Vector<double>& result, LinearAllocator<double>&) const;

    /**
     * Evaluate some derivatives for the Jacobian 
     * - assign the derivative of df/dV
     * - assign the derivative of df/dpsi
     * - assign the derivative of dg/dV
     * - assign the derivative of dg/dpsi
     * @param faultNo index of the current fault
     * @param traction matrix with sigma and tau at all nodes in the element
     * @param state current solution vector [S,psi,V] needed
     * @param result vector with the derivatives df/dV, df/dpsi, psi, V, g 
     * @param . some scratch
     * */
    void getJacobianQuantitiesExtended(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& result, LinearAllocator<double>&) const;

    /**
     * Calculate derivatives of the ageing law dg/dpsi and dg/dS
     * @param delta Evaluation of the Dirac delta
     * @param psi state variable (provide 0 if an off-diagonal element is considered)
     * @param dV_dpsi derivative dV/dpsi with all matrix elements
     * @param dV_dS derivative dV/dS with all matrix elements
     * @param dg_dS return value of dg/dS - the corresponding matrix entry
     * @param dg_dpsi return value of dg/dpsi - the corresponding matrix entry
     */
    void getAgeingDerivatives(double delta, double psi, double dV_dS, double dV_dpsi, double& dg_dS, double& dg_dpsi);

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

    /**
     * get the ageing law
     * @return law 
     */
    double getV0(){return law_.getV0();}

private:
    template <typename T> auto mat(Vector<T>& state) const {
        std::size_t nbf = space_.numBasisFunctions();
        return (extendedFormulation) ? matExtended(state) : matCompact(state);
    }

    template <typename T> auto matCompact(Vector<T>& state) const {
        std::size_t nbf = space_.numBasisFunctions();
        return reshape(state, nbf, NumQuantitiesCompact);
    }

    template <typename T> auto matExtended(Vector<T>& state) const {
        std::size_t nbf = space_.numBasisFunctions();
        return reshape(state, nbf, NumQuantitiesExtended);
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
void RateAndState<Law>::initCompact(std::size_t faultNo, Matrix<double> const& traction,
                             Vector<double>& state, LinearAllocator<double>&) const {
    auto s_mat = mat(state);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        s_mat(node, PsiIndex) =
            law_.psi_init(index + node, traction(node, 0), get_tau(node, traction));
    }
}

template <class Law>
void RateAndState<Law>::makeBig(std::size_t faultNo, Matrix<double> const& traction,
                             Vector<double>& state_small, Vector<double>& state_big, LinearAllocator<double>&) const {
    auto s_mat = matCompact(state_small);
    auto b_mat = matExtended(state_big);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;

    for (std::size_t node = 0; node < nbf; ++node) {
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            b_mat(node, t) = s_mat(node, t);
        }
        b_mat(node, PsiIndex) = s_mat(node, PsiIndex);

        auto sn = traction(node, 0);           
        auto Vi = law_.slip_rate(index + node, sn, get_tau(node, traction), s_mat(node, PsiIndex));
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            b_mat(node, VIndex + t) = Vi[t];
        }
    }
}

template <class Law>
void RateAndState<Law>::makeSmall(std::size_t faultNo,
                             Vector<double>& state_small, Vector<double>& state_big) const {
    auto s_mat = matCompact(state_small);
    auto b_mat = matExtended(state_big);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;

    for (std::size_t node = 0; node < nbf; ++node) {
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            s_mat(node, t) = b_mat(node, t);
        }
        s_mat(node, PsiIndex) = b_mat(node, PsiIndex);
    }
}



template <class Law>
double RateAndState<Law>::rhsCompactODE(std::size_t faultNo, double time, Matrix<double> const& traction,
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
double RateAndState<Law>::rhsExtendedODE(std::size_t faultNo, double time,
                              Vector<double const>& state, Vector<double>& result,
                              LinearAllocator<double>&, bool checkAS) const {
    double VMax = 0.0;
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    auto s_mat = mat(state);
    auto r_mat = mat(result);
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = 0.0;      // Will it always stay like that?
        auto psi = s_mat(node, PsiIndex);
        std::array<double, TangentialComponents> Vi;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            Vi[t] = s_mat(node, VIndex + t);
        }        
        double V = norm(Vi);

        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            r_mat(node, t) = Vi[t];            
        }
        VMax = std::max(VMax, V);

        r_mat(node, PsiIndex) = law_.state_rhs(index + node, V, psi);
    }
    return VMax;
}

template <class Law>
double RateAndState<Law>::lhsCompactDAE(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, Vector<double const>& state_der, Vector<double>& result,
                              LinearAllocator<double>&) const {
    double VMax = 0.0;
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    auto s_mat      = mat(state);
    auto s_der_mat  = mat(state_der);
    auto r_mat      = mat(result);
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn         = traction(node, 0);
        auto tau_vec    = law_.getTauVec(index + node, get_tau(node, traction));

        auto psi        = s_mat(node, PsiIndex);        
        auto psi_dot    = s_der_mat(node, PsiIndex);

        std::array<double, TangentialComponents> Vi;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            Vi[t]       = s_der_mat(node, t);
        }

        double V = norm(Vi);
        VMax = std::max(VMax, V);

        r_mat(node, 0)          = law_.friction_law(index + node, sn, get_tau(node, traction), psi, V);
        if (TangentialComponents == 2)
            r_mat(node, 1)      = Vi[0] / V - tau_vec[0] / norm(tau_vec);
        r_mat(node, PsiIndex)   = law_.state_rhs(index + node, V, psi) - psi_dot;
    }
    return VMax;
}
template <class Law>
double RateAndState<Law>::lhsExtendedDAE(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, Vector<double const>& state_der, Vector<double>& result,
                              LinearAllocator<double>&) const {
    double VMax = 0.0;
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    auto s_mat = mat(state);
    auto s_der_mat = mat(state_der);
    auto r_mat = mat(result);
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto tau_vec = law_.getTauVec(index + node, get_tau(node, traction));
        auto tau = norm(tau_vec);

        auto psi     = s_mat(node, PsiIndex);        
        auto psi_dot = s_der_mat(node, PsiIndex);

        auto V       = s_mat(node, VIndex);

        std::array<double, TangentialComponents> Vi;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            Vi[t] = tau_vec[t] / tau * V ;
        }

        std::array<double, TangentialComponents> Si_dot;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            Si_dot[t] = s_der_mat(node, t);
        }

        VMax = std::max(VMax, V);

        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            r_mat(node, t)    = Vi[t] - Si_dot[t];
        }
        r_mat(node, PsiIndex) =  law_.state_rhs(index + node, V, psi) - psi_dot;
        r_mat(node, VIndex)   = law_.friction_law(index + node, sn, get_tau(node, traction), psi, V);
        if (TangentialComponents == 2) 
        r_mat(node, VIndex+1) =  Vi[0] / V - tau_vec[0] / tau;
    }
    return VMax;
}


template <class Law>
void RateAndState<Law>::applyFrictionLaw(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, Vector<double>& state_der, Vector<double>& result,
                              LinearAllocator<double>&) const {
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    auto s_mat = mat(state);
    auto s_der_mat = mat(state_der);
    auto r_mat = mat(result);
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto tau_vec = law_.getTauVec(index + node, get_tau(node, traction));
        auto psi = s_mat(node, PsiIndex);        
        std::array<double, TangentialComponents> Vi;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            Vi[t] = s_der_mat(node, t);
        }
        double V = norm(Vi);
        r_mat(node, 0) = law_.friction_law(index + node, sn, get_tau(node, traction), psi, V);
    }
}

template <class Law>
double RateAndState<Law>::applyMaxFrictionLaw(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, LinearAllocator<double>&) const {
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    auto s_mat = mat(state);
    double fmax = 0.0;

    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto tau_vec = law_.getTauVec(index + node, get_tau(node, traction));
        auto psi = s_mat(node, PsiIndex);        
        std::array<double, TangentialComponents> Vi;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            Vi[t] = s_mat(node, VIndex + t);
        }
        double V = norm(Vi);
        double f = law_.friction_law(index + node, sn, get_tau(node, traction), psi, V);
        fmax = std::max(fmax, abs(f));  
    }
    return fmax;
}


template <class Law>
void RateAndState<Law>::getJacobianQuantitiesCompact(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double const>& state_der, Vector<double>& result, LinearAllocator<double>&) const {
    auto s_mat = mat(state);
    auto s_der_mat = mat(state_der);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {

        auto sn = traction(node, 0);
        auto psi = s_mat(node, PsiIndex);

        std::array<double, TangentialComponents> Vi;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            Vi[t] = s_der_mat(node, t);
        }
        double V = norm(Vi);

        result(0 * nbf + node) = law_.df_dV(index + node, sn, V, psi);
        result(1 * nbf + node) = law_.df_dpsi(index + node, sn, V, psi);
        result(2 * nbf + node) = law_.dg_dV();
        result(3 * nbf + node) = law_.dg_dpsi(psi);           
    }
}

template <class Law>
void RateAndState<Law>::getJacobianQuantitiesCompact(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& state_der, Vector<double>& result, LinearAllocator<double>&) const {
    auto s_mat = mat(state);
    auto s_der_mat = mat(state_der);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {

        auto sn = traction(node, 0);
        auto psi = s_mat(node, PsiIndex);

        std::array<double, TangentialComponents> Vi;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            Vi[t] = s_der_mat(node, t);
        }
        double V = norm(Vi);

        result(0 * nbf + node) = law_.df_dV(index + node, sn, V, psi);
        result(1 * nbf + node) = law_.df_dpsi(index + node, sn, V, psi);
        result(2 * nbf + node) = law_.dg_dV();
        result(3 * nbf + node) = law_.dg_dpsi(psi);           
    }
}


template <class Law>
void RateAndState<Law>::getJacobianQuantitiesExtended(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& result, LinearAllocator<double>&) const {
    auto s_mat = mat(state);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {

        auto sn  = traction(node, 0);
        auto psi = s_mat(node, PsiIndex);
        auto V   = s_mat(node, VIndex);

        result(0 * nbf + node) = law_.df_dV(index + node, sn, V, psi);
        result(1 * nbf + node) = law_.df_dpsi(index + node, sn, V, psi);
        result(2 * nbf + node) = law_.dg_dV();
        result(3 * nbf + node) = law_.dg_dpsi(psi);           
    }
}


template <class Law>
void RateAndState<Law>::getAgeingDerivatives(double delta, double psi, double dV_dS, double dV_dpsi, double& dg_dS, double& dg_dpsi) {    
    dg_dS = law_.dg_dS(psi, dV_dS);
    dg_dpsi = law_.dg_dpsi(delta, psi, dV_dpsi);
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
