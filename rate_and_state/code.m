
V0 = 1e-6;
Vinit = 1e-9;
L = 0.008;
a = 0.014;

b = 0.015;
f0 = 0.6;
s_n = 50;
eta = 2.600 * 3.464 / 2.0;
abstol = 1e-8;
reltol = 1e-8;

t_e = 1.0;
t_w = 0.1;






function r = V_star(t)
    r = (atan((t - t_e) / t_w) + 0.5 * pi) / (2 * pi);
end

function r = tau_star(t)
    r = 29;
end

function r = psi_star(t)
    r = a * log(2 * V0 / V_star(t) * sinh((tau_star(t) - eta * V_sta(t)) / (a * s_n)));
end

function r = fl(tau, V, psi)
    r = tau - s_n * a * asinh(V / (2 * V0) * exp(psi / a)) - eta * V
end

