#include <cmath>
#include <functional>
#include <iostream>
#include <ostream>

#include "Problem.h"
#include "Zero.h"

int main(int argc, char* argv[]) {
    // constexpr double tau = 26.54612236050893147442;
    // constexpr double tau = 32.9;
    constexpr double tau = 30;
    // constexpr double tau = 2;
    constexpr std::size_t n = 10000;
    constexpr double psi0 = 0.2;
    constexpr double psi1 = 0.8;
    double step = (psi1 - psi0) / (n - 1);

    std::cout<<"start sample.cpp"<<std::endl;

    std::cout << "psi,V,strength,G,dGdpsi" << std::endl;
    for (std::size_t i = 0; i < n; ++i) {
        double psi = i * step + psi0;
        auto F = [&tau, &psi](double V) { return fl(tau, V, psi); };
        double V = tndm::zeroIn(-tau / eta, tau / eta, F);
        double str = strength(V, psi);
        // double G = 1.0 - V * psi / L;
        double G = (b * V0 / L) * (exp((f0 - psi) / b) - V / V0);
        // F(V,psi) = 0
        // DF/Dpsi = dF/dV dV/dpsi + dF/dpsi = 0
        // dV/dpsi = - (dF/dV)^{-1} * dF/dpsi
        double dVdpsi = -dfldpsi(V, psi) / dfldV(V, psi);
        // double dGdpsi = -(dVdpsi * psi + V) / L;
        double dGdpsi = (b * V0 / L) * (-exp((f0 - psi) / b) / b - dVdpsi / V0);
        std::cout << psi << "," << V << "," << str << "," << G << "," << dGdpsi
                  << std::endl;
    }

    return 0;
}
