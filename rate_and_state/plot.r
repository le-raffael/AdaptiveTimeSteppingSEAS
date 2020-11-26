#!/usr/bin/env Rscript

require(ggplot2)
require(reshape2)
require(ggthemes)

d <- read.csv('result.csv')

d <- melt(d, id.vars='time')

p <- ggplot(d, aes(x=time, y=value)) +
        geom_line() +
        geom_point(size=0.8) +
        facet_grid(variable ~ ., scales='free_y') +
        theme_tufte()
ggsave('result.pdf', p, height=14, width=20, units='cm', device=cairo_pdf)
