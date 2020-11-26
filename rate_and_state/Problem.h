#ifndef PROBLEM_20200824_H
#define PROBLEM_20200824_H

#define _USE_MATH_DEFINES
#include <cassert>
#include <cmath>

constexpr double V0 = 1e-6;
constexpr double Vinit = 1e-9;
constexpr double L = 0.008;
constexpr double a = 0.014;
// constexpr double a = 0.0247;
constexpr double b = 0.015;
constexpr double f0 = 0.6;
// constexpr double s_n = 50e6;
constexpr double s_n = 50;
// constexpr double eta = 2600.0 * 3464.0 / 2.0;
constexpr double eta = 2.600 * 3.464 / 2.0;
constexpr double abstol = 1e-8;
constexpr double reltol = 1e-8;
constexpr std::size_t maxits = 100;

// MMS
constexpr double t_e = 2.5;
constexpr double t_w = 0.1;
// double tau_star(double t) { return 5.0e6; }
double tau_star(double t) { return 29.0; }

double V_star(double t) { return (atan((t - t_e) / t_w) + M_PI / 2.0) / M_PI; }
double dV_stardt(double t) {
    double x = (t - t_e) / t_w;
    return 1.0 / (t_w * M_PI * (x * x + 1.0));
}
double psi_star(double t) {
    //return a * log((2.0 * V0 / V_star(t)) *
    //               sinh((tau_star(t) - eta * V_star(t)) / (a * s_n)));
    return (L / V0) * exp((a / b) * log((2.0 * V0 / V_star(t)) *
    sinh((tau_star(t) - eta * V_star(t)) /
    (a * s_n))) - f0 / b);
}
double dpsi_stardt(double t) {
    double V = V_star(t);
    double dVdt = dV_stardt(t);
    double diff = (tau_star(t) - eta * V) / (a * s_n);
    //return a *
    //       (cosh(diff) * (-eta * dVdt) * 2.0 * V0 / (V * a * s_n) -
    //        2.0 * V0 * dVdt / (V * V) * sinh(diff)) /
    //       (2.0 * V0 / V * sinh(diff));
    return psi_star(t) * a * V / (b * sinh(diff)) *
    (1.0 / (V * a * s_n) * cosh(diff) * (-eta * dVdt) -
     dVdt / (V * V) * sinh(diff));
}

double strength(double V, double psi) {
    return s_n * a * asinh(V / (2.0 * V0) * exp(psi / a));
}

double fl(double tau, double V, double psi) {
    if (psi<0) psi = -psi;
    // std::cout<<"V = "<<V<<", psi = "<<psi<<", asinh term = "<<asinh(V / (2.0 * V0) * exp((f0 + b * log(V0 * psi / L)) / a))<<std::endl;
    // return tau - s_n * a * asinh(V / (2.0 * V0) * exp(psi / a)) - eta * V;
    return tau -
    s_n * a *
     asinh(V / (2.0 * V0) * exp((f0 + b * log(V0 * psi / L)) / a)) -
     eta * V;
}

double dfldV(double V, double psi) {
    //double e = exp(psi / a);
    double e = exp((f0 + b * log(V0 * psi / L)) / a);
    return -eta -
           a * s_n * e / (2.0 * V0 * sqrt(1 + e * e * V * V / (4.0 * V0 * V0)));
}

double dfldpsi(double V, double psi) {
    //double e = exp(psi / a);
    //return -s_n * e * V /
    //       (2.0 * V0 * sqrt(1 + e * e * V * V / (4.0 * V0 * V0)));
    double e = exp((f0 + b * log(V0 * psi / L)) / a);
    return -V * b * s_n * e /
           (2.0 * V0 * psi * sqrt(1 + e * e * V * V / (4.0 * V0 * V0)));
}

#endif  // PROBLEM_20200824_H
