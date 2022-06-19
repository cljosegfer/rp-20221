rm(list=ls())

library(Rcpp)
library(pROC)

sourceCpp('GGClassification_model.cpp')
sourceCpp('GGClassification_predict.cpp')

# params
dir_path <- 'data'
out_path <- 'output'
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

metodos <- c('base', 
             'lmnn', 
             'lfda')

# report
report <- matrix(0, nrow = length(datasets), ncol = 2*length(metodos))

metodo <- 'base'
mt <- 1

# processo
for (mt in seq_along(metodos)){
  metodo <- metodos[mt]
  for (i in seq(length(datasets))){
    relatorio <- matrix(0, ncol = K, nrow = 1)
    for (fold_n in seq(K)){
      # read
      filename <- sprintf('%s/%s/exportBase_%s_folds_10_exec_%s.mat',
                          dir_path, metodo, datasets[i], fold_n)
      data_mat <- R.matlab::readMat(filename)
      
      # train / test
      train <- data_mat$data[[1]]
      class_train <- data_mat$data[[2]]
      test <- data_mat$data[[3]]
      class_test <- data_mat$data[[4]]
      
      # clas
      modelo <- model(train, class_train)
      y_hat <- predict(modelo, test)
      
      # auc
      roc_obj <- roc(c(class_test), c(y_hat), quiet = TRUE)
      
      # log
      log <- sprintf('metodo: %s dataset: %s %s / %s | fold : %s / %s',
                     metodo, datasets[i], i, length(datasets),
                     fold_n, K)
      print(log)
      
      # report
      relatorio[fold_n] <- auc(roc_obj)
    }
    report[i, 2*mt - 1] <- round(mean(relatorio), digits = 2)
    report[i, 2*mt] <- round(sd(relatorio), digits = 2)
  }
}

# export
coletivo <- data.frame(report)
rownames(coletivo) <- datasets
columns <- c()
for (metodo in metodos){
  columns <- append(columns, sprintf('%s_mean', metodo))
  columns <- append(columns, sprintf('%s_std', metodo))
}
colnames(coletivo) <- columns

# write
write.csv(coletivo, 'output/clas.csv')
