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

indices <- c('q')

# processo
for (classificador in classificadores){
  # report
  report <- matrix(0, nrow = 2*length(indices), ncol = 2*length(metodos))
  
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
    
    # cor
    modelo <- cor.test(X, y)
    modelo2 <- cor.test(X, y, method = 'spearman', exact = FALSE)
    
    coef <- modelo$estimate
    pvalor <- modelo$p.value
    
    # report
    report[1, 2*mt - 1] <- coef
    report[1, 2*mt] <- pvalor
    
    report[2, 2*mt - 1] <- modelo2$estimate
    report[2, 2*mt] <- modelo2$p.value
  }
  
  # export
  coletivo <- data.frame(report)
  rownames(coletivo) <- c('Pearson', 'Spearman')
  columns <- c()
  for (metodo in metodos){
    columns <- append(columns, sprintf('%s_coef', metodo))
    columns <- append(columns, sprintf('%s_pvalor', metodo))
  }
  colnames(coletivo) <- columns
  
  # write
  filename <- sprintf('output/cor/%s_composto.csv', classificador)
  write.csv(coletivo, filename)
}
