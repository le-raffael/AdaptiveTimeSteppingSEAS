#!/usr/bin/env Rscript

require(ggplot2)
require(reshape2)
require(ggthemes)

fontfam <- 'LM Roman 10'
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

d <- read.csv('convergence.csv')
d <- melt(d, id.vars=c('type', 'dt'))
dts <- sort(unique(d$dt))
for (row in 1:nrow(d)) {
    index <- which(d[row,'dt'] == dts)
    if (index <= length(dts)) {
        otherdt <- dts[index+1]
        otherrow <- which(d$type == d[row,'type'] &
                          d$variable == d[row, 'variable'] &
                          d$dt == otherdt)
        if (length(otherrow) == 1) {
            d[row,'CO'] <- log(d[otherrow,'value'] / d[row,'value']) / log(otherdt / d[row,'dt'])
        }
    }
}

formatCO <- function(x) {
  y <- x
  tmp <- formatC(y, format='g', digits=2)
  tmp[is.na(y)] <- ''
  return(tmp)
}

p <- ggplot(d, aes(x=dt, y=value, label=formatCO(CO), colour=factor(type))) +
        geom_line() +
        geom_point() +
        geom_text(vjust=0, hjust=1) +
        scale_x_continuous(trans='log2', breaks=unique(d$dt)) +
        scale_y_continuous(trans='log10', breaks=10^seq(-15,0,3)) +
        scale_colour_manual(name='Type', values=cbbPalette, guide=guide_legend(nrow=1)) +
        facet_grid(variable ~ .) +
        theme_tufte() +
        theme(legend.position='bottom',
              legend.box.spacing=unit(0,'pt'),
              panel.spacing=unit(12,'pt'),
              text=element_text(family=fontfam),
              axis.text=element_text(family=fontfam, colour='black'))
ggsave('convergence.pdf', p, height=14, width=20, units='cm', device=cairo_pdf)

