rm(list = ls())

# library("spdep")
library("MASS")
# library("prettyGraphs")
# library("igraph")
# library("cccd")
# library(mvtnorm)
library(Rcpp)
sourceCpp('GGClassification_gabriel_graph.cpp')
library(mlbench)

gglocal = function(distance_array){
  n = nrow(distance_array)
  adjacency = matrix(data = 0, ncol = n, nrow = n)
  min_sum = 0
  for (i in 1:(n-1)){
    for (j in (i+1):n){
      min_sum = min(distance_array[i, ] + distance_array[j, ])
      if (distance_array[i, j] <= min_sum){
        adjacency[i, j] = 1
        adjacency[j, i] = 1
      }
    }
  }
  
  return(adjacency)
}

# gen data
d <- 2
npontos <- 30

m1 <- 3
sd1 <- 0.5
cov1 <- diag(sd1, d)
X1 <- mvrnorm(n = npontos, rep(m1, d), cov1)

m2 <- 6
sd2 <- 0.5
cov2 <- diag(sd2, d)
X2 <- mvrnorm(n = npontos, rep(m2, d), cov2)

# # plot
# par(pty = 's', bty = 'n')
# plot(X1, xlim = c(0, 9), ylim = c(0, 9), xlab = 'X1', ylab = 'X2', col = 'blue', pch = 20)
# points(X2, col = 'red', pch = 20)

# gg
X <- rbind(X1, X2)
D <- c(rep(1, nrow(X1)), rep(-1, nrow(X2)))
gg <- GabrielGraph(X)

distance <- as.matrix(dist(x = X, method = 'minkowski', p = 2,
                   diag = TRUE, upper = TRUE))
distance <- distance ^ 4
diag(distance) <- Inf
localgg <- gglocal(distance)

# write
data <- cbind(X, D)
write.table(data, file = 'data/raw.csv', sep = ',', row.names = FALSE, col.names = FALSE)
write.table(gg, file = 'data/raw-gg.csv', sep = ',', row.names = FALSE, col.names = FALSE)

# gen overlap data
npontos <- 50

m1 <- 4
sd1 <- 0.9
cov1 <- diag(sd1, d)
X1 <- mvrnorm(n = npontos, rep(m1, d), cov1)

m2 <- 6
sd2 <- 0.9
cov2 <- diag(sd2, d)
X2 <- mvrnorm(n = npontos, rep(m2, d), cov2)

# # plot
# par(pty = 's', bty = 'n')
# plot(X1, xlim = c(0, 9), ylim = c(0, 9), xlab = 'X1', ylab = 'X2', col = 'blue', pch = 20)
# points(X2, col = 'red', pch = 20)

# gg
X <- rbind(X1, X2)
D <- c(rep(1, nrow(X1)), rep(-1, nrow(X2)))
gg <- GabrielGraph(X)

distance <- as.matrix(dist(x = X, method = 'minkowski', p = 2,
                           diag = TRUE, upper = TRUE))
distance <- distance ^ 4
diag(distance) <- Inf
localgg <- gglocal(distance)

# write
data <- cbind(X, D)
write.table(data, file = 'data/overlap.csv', sep = ',', row.names = FALSE, col.names = FALSE)
write.table(gg, file = 'data/overlap-gg.csv', sep = ',', row.names = FALSE, col.names = FALSE)

# spirals
data <- mlbench.spirals(n = 1000, cycles = 1, sd = 0.05)
X <- as.data.frame(data['x'])
y <- as.data.frame(data['classes'])

# write
csv <- cbind(X, y)
write.table(csv, file = 'data/spirals.csv', sep = ',', row.names = FALSE, col.names = FALSE)
