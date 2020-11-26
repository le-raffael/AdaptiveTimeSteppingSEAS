#ifndef MATHFUNCTIONS_20201103_H
#define MATHFUNCTIONS_20201103_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

template<typename T>
void RK2(std::vector<T>& X, std::function<T(T,T)> F, double t, double dt, size_t step){    
    T k1 = dt * F(X[step], t);
    T k2 = dt * F(X[step] + 0.5 * k1, t + 0.5 * dt);
    if (X.size() == step + 1) {
        X.push_back(X[step] + k2);
    } else {
        X[step+1] = X[step] + k2;
    }
}

template<typename T>
void RK4(std::vector<T>& X, std::function<T(T,T)> F, double t, double dt, size_t step){    
    T k1 = dt * F(X[step], t);
    T k2 = dt * F(X[step] + 0.5 * k1, t + 0.5 * dt);
    T k3 = dt * F(X[step] + 0.5 * k2, t + 0.5 * dt);
    T k4 = dt * F(X[step] + k3, t + dt);

    T X_new = X[step] + 1./6 * k1 + 1./3 * k2 + 1./3 * k3 + 1./6 * k4;
    if (X.size() == step + 1) {
        X.push_back(X_new);
    } else {
        X[step+1] = X_new;
    }
}

template<typename T>
void RKF45(std::vector<T>& X, T& error, std::function<T(T,T)> F, double t, double dt, size_t step){    

    // from Butcher tableau of RKF45
    T c2 = 1./4; 
    T c3 = 3./8;
    T c4 = 12./13;
    T c5 = 1.;
    T c6 = 1./2;

    T a21 = 1./4;
    T a31 = 3./32;      T a32 = 9./32;
    T a41 = 1932./2197; T a42 =-7200./2197; T a43 = 7296./2197;
    T a51 = 439/216;    T a52 = -8.;        T a53 = 3680./513;  T a54 = -845./4104;
    T a61 = -8./27;     T a62 = 2.;         T a63 =-3544./2565; T a64 = 1859./4104; T a65 = -11./40;   

    T b1 = 16./135;  T b2 = 0;  T b3 = 6656./12825; T b4 = 28561./56430; T b5 = -9./50; T b6 = 2./55;
    T bs1 = 25./216; T bs2 = 0; T bs3 = 1408./2565; T bs4 = 2197./4104;  T bs5 = -1./5;


    // perform the RK scheme    
    T k1 = dt * F(X[step], t);
    T k2 = dt * F(X[step] + a21 * k1, t + c2 * dt);
    T k3 = dt * F(X[step] + a31 * k1 + a32 * k2, t + c3 * dt);
    T k4 = dt * F(X[step] + a41 * k1 + a42 * k2 + a43 * k3, t + c4 * dt);
    T k5 = dt * F(X[step] + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, t + c5 * dt);
    T k6 = dt * F(X[step] + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5, t + c6 * dt);

    T X_new5 = X[step] + b1  * k1 + b2  * k2 + b3  * k3 + b4  * k4 + b5  * k5 + b6 * k6;
    T X_new4 = X[step] + bs1 * k1 + bs2 * k2 + bs3 * k3 + bs4 * k4 + bs5 * k5;

    // get the prediction of the relative error
    error = fabs(X_new5 - X_new4);

    if (X.size() == step + 1) {
        X.push_back(X_new4);
    } else {
        X[step+1] = X_new4;
    }
}


template<typename T>
void RKDP45(std::vector<T>& X, T& error, std::function<T(T,T)> F, double t, double dt, size_t step){    

    // from Butcher tableau of RKF45
    T c2 = 1./5; 
    T c3 = 3./10;
    T c4 = 4./5;
    T c5 = 8./9;
    T c6 = 1.;
    T c7 = 1.;

    T a21 = 1./5;
    T a31 = 3./40;      T a32 = 9./40;
    T a41 = 44./45;     T a42= -56./15;     T a43 = 32./9;
    T a51 =19372./6561; T a52 =-25360./2187;T a53 =64448./6561; T a54 = -212./729;
    T a61 =-9017./3168; T a62 = -355./33;   T a63 =46732./5247; T a64 = 49./176;    T a65 =-5103./18656;   
    T a71 = 35./384;    T a72 = 0;          T a73 =500./1113;   T a74 = 125./192;   T a75 =-2187./6784;  T a76 = 11./84;  

    T b1 = 35./384;     T b2 = 0;  T b3 =500./1113;    T b4 = 125./192;     T b5 =-2187./6784;     T b6 = 11./84;  
    T bs1 =5179./57600; T bs2 = 0; T bs3 =7571./16695; T bs4 = 393./640;    T bs5 =-92097./339200; T bs6 = 187./2100; T bs7 = 1./40;


    // perform the RK scheme    
    T k1 = dt * F(X[step], t);
    T k2 = dt * F(X[step] + a21 * k1, t + c2 * dt);
    T k3 = dt * F(X[step] + a31 * k1 + a32 * k2, t + c3 * dt);
    T k4 = dt * F(X[step] + a41 * k1 + a42 * k2 + a43 * k3, t + c4 * dt);
    T k5 = dt * F(X[step] + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, t + c5 * dt);
    T k6 = dt * F(X[step] + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5, t + c6 * dt);
    T k7 = dt * F(X[step] + a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6, t + c7 * dt);

    T X_new5 = X[step] + b1  * k1 + b2  * k2 + b3  * k3 + b4  * k4 + b5  * k5 + b6 * k6;
    T X_new4 = X[step] + bs1 * k1 + bs2 * k2 + bs3 * k3 + bs4 * k4 + bs5 * k5 + bs6 * k6 + bs7 * k7;

    // get the prediction of the relative error
    error = fabs(X_new5 - X_new4);

    if (X.size() == step + 1) {
        X.push_back(X_new4);
    } else {
        X[step+1] = X_new4;
    }
}


template<typename T>
void NewtonMethod(T x_old, T& x, std::function<T(T)> f, double tolerance){

    T f_x = f(x);  
    T f_x_old = f(x_old);

    //evaluate derivative of f (secant method)
    T f_x_deriv = (f_x - f_x_old) / (x - x_old);

    T x_new = x - f_x / f_x_deriv;

    size_t count = 0;

    while (fabs(x_new - x) > tolerance){
        x_old = x;
        f_x_old = f_x;
        x = x_new;

        f_x = f(x);
        f_x_deriv = (f_x - f_x_old) / (x - x_old);

        x_new =  x - f_x / f_x_deriv;

        count++;
        if (count > 100) break;
    }
    x = x_new;
}

template<typename T> 
T ImplicitEulerMethod(std::vector<T>& X, T& error, std::function<T(T,T)> F, double t, double dt, size_t step) {
    T x_n = X[step];

    std::function<T(T)> f = [&F, &dt, &t, &x_n](T x){
        return x_n - x + dt * F(x,t+dt);
    };

    // initial guess
    T x_new = x_n + dt * F(x_n, t);

    NewtonMethod(x_n, x_new, f, 1e-7);

    return x_new;
}

template<typename T> 
T ImplicitBDF2Method(std::vector<T>& X, T& error, std::function<T(T,T)> F, double t, double dt, double dt_old, size_t step) {
    T x_n = X[step];
    T x_n_1 = X[step-1];

    T frac = dt_old / dt;
    T a = -frac*frac - 2*frac - 1;

    std::function<T(T)> f = [&F, &dt, &dt_old, &a, &t, &x_n, &x_n_1](T x){
        return -x_n_1 - a*x_n + (a+1)*x - (dt_old + (a+1) * dt) * F(x,t+dt);
    };

    // initial guess
    T x_new = x_n + dt * F(x_n, t);

    NewtonMethod(x_n, x_new, f, 1e-7);
 
    return x_new;
}

template<typename T> 
T ImplicitBDF3Method(std::vector<T>& X, T& error, std::function<T(T,T)> F, double t, double dt, double dt_old, double dt_old_old, size_t step) {
    T x_n = X[step];
    T x_n_1 = X[step-1];
    T x_n_2 = X[step-2];

    T u = dt;
    T v = dt + dt_old;
    T w = dt + dt_old + dt_old_old;

    T a = w*w * (w - u) / (v*v * (u - v));
    T b = w*w * (v - w) / (u*u * (u - v));

    std::function<T(T)> f = [&F, &dt, &dt_old, &dt_old_old, &a, &b, &t, &x_n, &x_n_1, &x_n_2](T x){
        return -x_n_2 -a*x_n_1 - b*x_n + (1+a+b)*x - (dt_old_old + (a+1) * dt_old + (1+a+b) * dt) * F(x,t+dt);
    };

    // initial guess
    T x_new = x_n + dt * F(x_n, t);

    NewtonMethod(x_n, x_new, f, 1e-7);
 
    return x_new;
}


template<typename T> 
void timeAdaptiveBDF12Method(std::vector<T>& X, T& error, std::function<T(T,T)> F, double t, double dt, double dt_old, size_t step) {
    

    // Backwards Euler solution
    T firstOrderSolution = ImplicitEulerMethod(X, error, F, t, dt, step);

    T secondOrderSolution;


    // BDF2 for the error term
    if (X.size() >= 2) secondOrderSolution = ImplicitBDF2Method(X, error, F, t, dt, dt_old, step);

    error = fabs(firstOrderSolution - secondOrderSolution);


    if (X.size() == step + 1) {
        X.push_back(firstOrderSolution);
    } else {
        X[step+1] = firstOrderSolution;
    }
}


template<typename T> 
void timeAdaptiveBDF23Method(std::vector<T>& X, T& error, std::function<T(T,T)> F, double t, double dt, double dt_old, double dt_old_old, size_t step) {
    

    // Backwards Euler solution
    T firstOrderSolution = ImplicitBDF2Method(X, error, F, t, dt, dt_old, step);

    T secondOrderSolution;


    // BDF2 for the error term
    if (X.size() >= 2) secondOrderSolution = ImplicitBDF3Method(X, error, F, t, dt, dt_old, dt_old_old, step);

    error = fabs(firstOrderSolution - secondOrderSolution);


    if (X.size() == step + 1) {
        X.push_back(firstOrderSolution);
    } else {
        X[step+1] = firstOrderSolution;
    }
}

#endif