#!/usr/bin/env Rscript

require(ggplot2)

fun <- function(V, a) {
    return(a * asinh(V/(2.0 * 1e-6) * exp((0.6 + 0.015 * log(1.0e-6 * 8000 / 0.008))/a)))
}

fun1 <- function(V) {
    return(fun(V, 0.010))
}

fun2 <- function(V) {
    return(fun(V, 0.015))
}

fun3 <- function(V) {
    return(fun(V, 0.025))
}

p <- ggplot(data.frame(x=0), aes(x=x)) +
        stat_function(fun = fun1, colour='red', n=10001) +
        stat_function(fun = fun2, colour='blue', n=10001) +
        stat_function(fun = fun3, n=10001) +
        xlim(-4e-8, 4e-8)

ggsave('F.pdf', p, device=cairo_pdf)

tau0 <- function(a) {
    e <- exp((0.6 + 0.010 * log(1e-6 / 1e-9)) / a);
    return (50e6 * a * asinh((1e-9 / (2.0 * 1e-6)) * e) + 2670*3464/2 * 1e-9)
}

p <- ggplot(data.frame(x=0), aes(x=x)) +
        stat_function(fun = tau0, n=10001) +
        xlim(0.010, 0.025)

ggsave('tau0.pdf', p, device=cairo_pdf)


