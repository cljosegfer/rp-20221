rm(list=ls())

# read
metodo <- 'base'
auc_d <- read.csv(file = sprintf('output/%s/coletivo.csv', metodo))
dis_d <- read.csv(file = sprintf('output/%s/d_coletivo.csv', metodo))
silh <- read.csv(file = 'output/otm-log.csv', header = FALSE)
metodo <- 'silh'
auc_s <- read.csv(file = sprintf('output/%s/coletivo.csv', metodo))
dis_s <- read.csv(file = sprintf('output/%s/d_coletivo.csv', metodo))

# lm
auc <- auc_s[, 2] - auc_d[, 2]
silh <- silh[, 3] - silh[, 2]
dis <- dis_s[, 2] - dis_d[, 2]

# plot
png(file = 'fig/auc_silh.png',
    width=600, height=425)
plot(silh, auc)
dev.off()
png(file = 'fig/auc_dis.png',
    width=600, height=425)
plot(dis, auc)
dev.off()
png(file = 'fig/dis_silh.png',
    width=600, height=425)
plot(silh, dis)
dev.off()
