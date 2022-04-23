rm(list = ls())

options = c('raw', 'trans')
for (option in options){
  data <- read.csv(file = paste('data/', option, '.csv', sep = ''), header = FALSE)
  
  X <- data[, -ncol(data)]
  Y <- data[, ncol(data)]
  
  dist_euclidean <- as.matrix(dist(X))
  png(file = paste('fig/dist-', option, '.png', sep = ''), width = 600, height = 400)
  image(dist_euclidean)
  dev.off()
}

