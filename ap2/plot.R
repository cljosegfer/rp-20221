rm(list = ls())

data <- read.csv(file = 'data/trans.csv', header = FALSE)

X <- data[, -ncol(data)]
Y <- data[, ncol(data)]

png(file = 'fig/plot-trans.png', width = 600, height = 400)
plot(X, type = 'p', col = Y, xlim = c(-0.5, 1.5), ylim = c(5, 15))
dev.off()
