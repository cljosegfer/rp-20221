rm(list=ls())

library(Rcpp)

sourceCpp('GGClassification_gabriel_graph.cpp')

# params
dir_matlab_file <- 'data/temp_bases/temp'
dir_gg <- 'data/gg'
datasets <- c('australian', 
              'banknote', 
              'breastcancer', 
              'breastHess', 
              'bupa', 
              'climate', 
              'diabetes', 
              'fertility', 
              'german', 
              'golub', 
              'haberman', 
              'heart', 
              'ILPD', 
              'parkinsons', 
              'sonar')
K <- 10

# processo
# i <- 1
# fold_n <- 1
for (i in seq(length(datasets))){
  for (fold_n in seq(K)){
    # read
    filename <- sprintf('%s/exportBase_%s_folds_10_exec_%s.mat',
                        dir_matlab_file, datasets[i], fold_n)
    data_mat <- R.matlab::readMat(filename)
    
    # train / test
    train <- data_mat$data[[1]]
    class_train <- data_mat$data[[2]]
    test <- data_mat$data[[3]]
    class_test <- data_mat$data[[4]]
    
    # gg
    X <- train
    gg <- GabrielGraph(X)
    
    # editing
    edited <- c()
    for (k in 1:nrow(gg)){
      cluster <- class_train == class_train[k]
      intrusos <- gg[k, !cluster]
      if (sum(intrusos) == 0){
        edited <- append(edited, k)
      }
    }
    
    # write
    path <- sprintf('%s/ggBase_%s_folds_10_exec_%s.csv',
                    dir_gg, datasets[i], fold_n)
    write.csv(edited, path, row.names = FALSE)
    
    # log
    log <- sprintf('dataset: %s %s / %s | fold : %s / %s', 
                   datasets[i], i, length(datasets), 
                   fold_n, K)
    print(log)
  }
}
