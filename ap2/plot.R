rm(list = ls())

data <- read.csv(file = 'data/trans.csv', header = FALSE)

X <- as.matrix(data[, -ncol(data)])
Y <- data[, ncol(data)]

png(file = 'fig/plot-trans.png', width = 600, height = 400)
plot(X, type = 'p', col = Y, xlim = c(min(X[, 1]), max(X[, 1])), ylim = c(min(X[, 2]), max(X[, 2])))
dev.off()
