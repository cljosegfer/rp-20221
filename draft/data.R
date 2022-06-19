rm(list = ls())

library(mvtnorm)

n1 <- 140
n2 <- 60

s = 2
sigma = matrix(c(s, 0, 0, s), nrow = 2, byrow = TRUE)
xc1 <- rmvnorm(n1, mean = c(2, 2), sigma = sigma)
xc2 <- rmvnorm(n2, mean = c(4, 4), sigma = sigma)

X <- rbind(xc1, xc2)
Y <- c(rep(1, n1), rep(-1, n2))

data <- cbind(X, Y)
write.table(data, file = 'data/raw.csv', sep = ',', row.names = FALSE, col.names = FALSE)

# png(file = 'fig/plot.png', width = 600, height = 400)
# plot(X, type = 'p', col = Y)
# dev.off()
