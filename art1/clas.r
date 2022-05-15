rm(list=ls())

library(Rcpp)
library(pROC)

sourceCpp('GGClassification_model.cpp')
sourceCpp('GGClassification_predict.cpp')

# params
# dir_path = 'data/temp_bases/temp'
# metodo <- 'base'
metodo <- 'silh'
dir_path <- 'data/trans'
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

# report
relatorio <- matrix(0, ncol = length(datasets), nrow = K)
discard <- matrix(0, ncol = length(datasets), nrow = K)

# processo
for (i in seq(length(datasets))){
  for (fold_n in seq(K)){
    # read
    filename <- sprintf('%s/exportBase_%s_folds_10_exec_%s.mat',
                        dir_path, datasets[i], fold_n)
    data_mat <- R.matlab::readMat(filename)

    # train / test
    train <- data_mat$data[[1]]
    class_train <- data_mat$data[[2]]
    
    test <- data_mat$data[[3]]
    class_test <- data_mat$data[[4]]

    # clas
    modelo <- model(train, class_train)
    y_hat <- predict(modelo, test)

    # loss
    acuracia <- 1 - sum(abs(class_test - y_hat)) / length(class_test) / 2

    # auc
    roc_obj <- roc(class_test, y_hat)
    
    # log
    log <- sprintf('dataset: %s %s / %s | fold : %s / %s',
                   datasets[i], i, length(datasets),
                   fold_n, K)
    print(log)
    
    # discard
    d <- 1 - length(modelo$Filtrado[, 1]) / length(train[, 1])
    discard[fold_n, i] <- d

    # report
    relatorio[fold_n, i] <- auc(roc_obj)
  }
}

# report
individual <- data.frame(relatorio)
colnames(individual) <- datasets
coletivo <- data.frame(apply(X = individual, MARGIN = 2, mean),
                       apply(X = individual, MARGIN = 2, sd))
colnames(coletivo) <- c('mean', 'sd')

d_individual <- data.frame(discard)
colnames(d_individual) <- datasets
d_coletivo <- data.frame(apply(X = d_individual, MARGIN = 2, mean),
                       apply(X = d_individual, MARGIN = 2, sd))
colnames(d_coletivo) <- c('mean', 'sd')

# write
write.csv(individual, paste('output', metodo, 'individual.csv', sep = '/'))
write.csv(coletivo, paste('output', metodo, 'coletivo.csv', sep = '/'))
write.csv(d_individual, paste('output', metodo, 'd_individual.csv', sep = '/'))
write.csv(d_coletivo, paste('output', metodo, 'd_coletivo.csv', sep = '/'))
