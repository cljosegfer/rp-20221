rm(list=ls())

# params
dir_path <- 'output'

metodos <- c('base', 
             'lmnn', 
             'lfda')

classificadores <- c('svm', 
             'clas')

# indices <- c('calinski_harabasz', 
#              'davies_bouldin', 
#              'silhouette', 
#              'q', 
#              'discard')

# indices <- c('q', 
#              'discard')

indices <- c('discard')

# processo
primeiro = TRUE
img_path <- sprintf('%s/fig/reg_%s.png', dir_path, indices)
png(file = img_path, width = 612, height = 525)
for (classificador in classificadores){
  # report
  report <- matrix(0, nrow = length(indices) + 1, ncol = 2*length(metodos))
  
  filename <- sprintf('%s/%s.csv', dir_path, classificador)
  acc <- read.csv(filename, row.names = 1)
  
  for (mt in seq_along(metodos)){
    metodo <- metodos[mt]
    
    # data
    X <- matrix(0, nrow = nrow(acc), ncol = length(indices))
    j <- 1
    for (indice in indices){
      index <- read.csv(sprintf('%s/indices/%s.csv', dir_path, indice), row.names = 1)
      X[, j] <- index[, mt]
      j <- j + 1
    }
    y <- acc[, 1]
    
    # lm
    dt <- data.frame(y, X)
    modelo <- lm(y ~ X, data = dt)
    
    coef <- modelo$coefficients
    pvalor <- summary(modelo)$coefficients[, 4]
    
    # report
    report[, 2*mt - 1] <- coef
    report[, 2*mt] <- pvalor
    
    if (primeiro == TRUE) {
      plot(X, y, type = 'p', xlim = c(0, 1), ylim = c(0, 1), 
           xlab = 'q', ylab = 'AUC')
      primeiro = FALSE
    }
    else {
      points(X, y, xlim = c(0, 1), ylim = c(0, 1))
    }
    abline(modelo, col = which(classificadores == classificador) + 1)
    # par(new=TRUE)
  }
  
  # export
  coletivo <- data.frame(report)
  rownames(coletivo) <- c('intercept', indices)
  columns <- c()
  for (metodo in metodos){
    columns <- append(columns, sprintf('%s_coef', metodo))
    columns <- append(columns, sprintf('%s_pvalor', metodo))
  }
  colnames(coletivo) <- columns
  
  # write
  filename <- sprintf('output/lm/%s_composto.csv', classificador)
  write.csv(coletivo, filename)
}
dev.off()
