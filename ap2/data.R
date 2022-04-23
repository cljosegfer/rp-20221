rm(list = ls())

library(mvtnorm)

n1 <- 140
n2 <- 60

xc1 <- rmvnorm(n1, mean = c(3, 3), sigma = matrix(c(5, 0, 0, 0.2), nrow = 2, byrow = TRUE))
xc2 <- rmvnorm(n2, mean = c(3, 4), sigma = matrix(c(5, 0, 0, 0.2), nrow = 2, byrow = TRUE))

X <- rbind(xc1, xc2)
Y <- c(rep(1, n1), rep(2, n2))

data <- cbind(X, Y)
write.table(data, file = 'data/raw.csv', sep = ',', row.names = FALSE, col.names = FALSE)

png(file = 'fig/plot.png', width = 600, height = 400)
plot(X, type = 'p', col = Y, xlim = c(-4, 10), ylim = c(1, 7))
dev.off()
