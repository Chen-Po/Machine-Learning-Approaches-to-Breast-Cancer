mydata <- read.csv("/Users/liaochenpo/Desktop/Desktop/Course/Machine Learning/Statistical learning/Final_Project/BreastCancer.csv", header = TRUE)
mydata$Classification <- as.factor(mydata$Classification)
dim(mydata)
dim(na.omit((mydata)))
head(mydata)
str(mydata)
summary(mydata)

#### Exploratory Data analysis
library(GGally)
library(reshape)
library(ggplot2)
library(ggcorrplot)
library(scales)
ggpairs(data = mydata, aes(colour = Classification), 
        lower=list(combo=wrap("facethist", binwidth=3))) + theme_bw()

redata <- melt(mydata, id.var = "Classification")
pp <- ggplot(data = redata, aes(x=value)) +  
  geom_histogram(aes(y = ..density..), bins=30) + 
  geom_density(col = "#1b98e0", size = 1) + 
  facet_wrap(~ variable, scales = "free")
pp

p <- ggplot(data = redata, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=Classification)) + 
  facet_wrap( ~ variable, scales="free")
p

ggcorrplot(cor(mydata[, -10]), hc.order = TRUE, 
           type = "lower",
           ggtheme = theme_gray,
           colors = c("#6D9EC1", "white", "#E46726"), lab = TRUE, tl.cex = 12)

aa <- data.frame(round(prop.table(table(mydata$Classification)), 2))
colnames(aa) <- c("Classification", "Freq")

ggplot(aa, aes(x = "", y = Freq, fill = Classification)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) + 
  geom_text(aes(y = Freq / 1.82 + c(0, cumsum(Freq)[-length(Freq)]), 
                label = percent(1 - Freq)), size = 5) + theme_minimal()

# Remove outliers of each class (mean +/- 3sd)
mydata$ID <- c(1:nrow(mydata))
Class1 <- subset(mydata, Classification == "1")
Class2 <- subset(mydata, Classification == "2")

Class1$ID[which(Class1$Resistin > mean(Class1$Resistin) + 3 * sd(Class1$Resistin))]
Class1$ID[which(Class1$Adiponectin > mean(Class1$Adiponectin) + 3 * sd(Class1$Adiponectin))]
Class2$ID[which(Class2$Resistin > mean(Class2$Resistin) + 3 * sd(Class2$Resistin))]
Class2$ID[which(Class2$Adiponectin > mean(Class2$Adiponectin) + 3 * sd(Class2$Adiponectin))]

mydata <- mydata[-c(14, 17, 38, 88, 115), -11]

#### Logistic Regression
library(glmnet)
library(nnet)
set.seed (20221217)
cv.out <- cv.glmnet(as.matrix(mydata[, 1:9]), 
                    as.matrix(mydata[, 10]), alpha=1, nfolds = 5, 
                    family="binomial", type.measure="auc")
plot(cv.out)
bestlam = cv.out$lambda.min
bestlam
lasso.coef = predict(cv.out, type = "coefficients", s = bestlam)[,1]
abline(v = log(bestlam), col = "red")

coef.names = names(lasso.coef[lasso.coef != 0])[-1]
coef.names

lasso.coef = predict(cv.out, type = "coefficients", s = bestlam)[,1]
coef <- data.frame(round(lasso.coef, 2))
colnames(coef) <- "Coef"
coef
knitr::kable(coef)

max(cv.out$cvm)
which.max(cv.out$cvm)
cv.out$cvup[which.max(cv.out$cvm)]
cv.out$cvlo[which.max(cv.out$cvm)]

set.seed (20221217)
cv.out.acc <- cv.glmnet(as.matrix(mydata[, 1:9]), 
                        as.matrix(mydata[, 10]), alpha = 1, nfolds = 5, 
                        family="binomial", type.measure = "class")
logit.acc <- 1 - cv.out.acc$cvm[which(cv.out.acc$lambda == bestlam)]


logit.auc <- max(cv.out$cvm)

#### Random Forest
# Mtry: Number of variables randomly sampled as candidates at each split.
library (randomForest)
library("ROCR")
k = 5
set.seed(20221217)
folds = sample(1:k, nrow(mydata), replace=TRUE)
rf.acc = matrix(NA, k, 9)
rf.auc = matrix(NA, k, 9)
for(j in 1:k){
  for(i in 1:9){
    rf.fit = randomForest(Classification ~ ., data = mydata[folds != j,], mtry = i, importance = TRUE, ntree = 5000)
    pred.class = predict(rf.fit, mydata[folds == j,], type = "class")
    pred.prob = predict(rf.fit, mydata[folds == j,], type = "prob")[, 2]
    rf.table = table(pred.class, mydata$Classification[folds == j])
    rf.acc[j, i] = sum(diag(rf.table)) / sum(rf.table)
    ROC.pred <- prediction(pred.prob, mydata[folds == j, 10])
    ROC.perf <- performance(ROC.pred, measure = "tpr", x.measure = "fpr")
    AUC <- performance(ROC.pred, "auc")
    rf.auc[j, i] <- AUC@y.values[[1]]
  }
}
rf.acc
rf.auc 

matplot(cbind(apply(rf.auc, 2, mean),
              apply(rf.auc, 2, max) ,
              apply(rf.auc, 2, min)), type = "l", lty = c(1, 2, 2), col = "black",
        xlab = "mtry", ylab = "AUC")
abline(v = which.max(apply(rf.auc, 2, mean)), col = "red")
legend("topleft", c("mean", "min & max"), lty = 1:2, bty="n",  pt.cex = 0)

rf.auc[, which.max(apply(rf.auc, 2, mean))]
mean(rf.auc[, which.max(apply(rf.auc, 2, mean))])
sd(rf.auc[, which.max(apply(rf.auc, 2, mean))])

mean(rf.acc[, which.max(apply(rf.acc, 2, mean))])

set.seed(20221217)
rf.model <- randomForest(Classification ~ ., data = mydata, 
                         mtry = which.max(apply(rf.auc, 2, mean)), 
                         importance = TRUE, ntree = 5000)
rf.model
varImpPlot(rf.model)


#### SVM(radial)
library(tidyverse)
library(kernlab)
library(parsnip)
library(tidymodels)
library(rsample)
library(doParallel)
library(tidyquant)

tuning_svm <- svm_rbf(cost = 1, rbf_sigma = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_grid <- expand.grid(rbf_sigma = seq(0.1, 3, by = 0.1))
svm_grid

set.seed(20221217)
cv_folds <- vfold_cv(data = mydata, v = 5)
cv_folds

svm_wf <- workflow() %>%
  add_model(tuning_svm) %>%
  add_formula(as.factor(Classification) ~ .)

all_cores <- parallel::detectCores(logical = FALSE)
cl <- makeCluster(all_cores)
registerDoParallel(cl)

svm_results <-  svm_wf %>% 
  tune_grid(resamples = cv_folds,
            grid = svm_grid)

results <- svm_results %>% 
  collect_metrics() %>%
  arrange(desc(mean))
results[which(results[, 1] == c(results[1, 1])), ]
results[1, 1][[1]]

##### Tune Cost #####
tuning_svm <- svm_rbf(cost = tune(), rbf_sigma = 0.2) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

tuning_svm

svm_grid <- expand.grid(cost = c(0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2))
svm_grid

set.seed(20221217)
cv_folds <- vfold_cv(data = mydata, v = 5)
cv_folds

svm_wf <- workflow() %>%
  add_model(tuning_svm) %>%
  add_formula(as.factor(Classification) ~ .)

svm_results <-  svm_wf %>% 
  tune_grid(resamples = cv_folds,
            grid = svm_grid)

results <- svm_results %>% 
  collect_metrics() %>%
  arrange(desc(mean))
results[which(results[, 1] == c(results[1, 1])), ]
svm.auc <- results[which(results[, 1] == c(results[1, 1])), ][1, 4]
svm.acc <- results[which(results[, 1] == c(results[1, 1])), ][2, 4]

svm_results %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(rbf_sigma = factor(cost)) %>%
  mutate(rbf_sigma = factor(cost)) %>%
  ggplot(aes(cost, mean)) +
  geom_point() +
  labs(y = "AUC") +
  tidyquant::theme_tq()

######Comparison####
AUC.1 <- data.frame(logit.auc, mean(rf.auc[, which.max(apply(rf.auc, 2, mean))]), svm.auc)
perfr <- rbind(AUC.1, c(logit.acc, mean(rf.acc[, which.max(apply(rf.acc, 2, mean))]), svm.acc[[1]]))
colnames(perfr) <- c("Logistic Regression","Random Forest","SVM")
row.names(perfr) <- c("AUC", "Accuracy")
round(perfr, 4)
knitr::kable(t(round(perfr, 4)))

#### Basis expansion
library(splines)

Age <- c(2, 3, 4, 5, 6)
BMI <- c(2, 3, 4, 5, 6)
Glucose <- c(2, 3, 4, 5, 6)
Resistin <- c(2, 3, 4, 5, 6)
Insulin <- c(2, 3, 4, 5, 6)
Leptin <- c(2, 3, 4, 5, 6)
Adiponectin <- c(2, 3, 4, 5, 6)
ns.grid <- expand.grid(Age, BMI, Glucose, Resistin, Insulin, Leptin, Adiponectin)

mydata$Classification <- ifelse(mydata$Classification == 2, 1, 0)

basis.mean.auc <- NULL
basis.max.auc <- NULL
basis.min.auc <- NULL
basis.mean.acc <- NULL
y <- mydata$Classification

for(i in 1:5**5){
  x <- model.matrix(Classification ~ ns(Age, ns.grid$Var1[i])  + 
                    ns(BMI, ns.grid$Var2[i]) + 
                    ns(Glucose, ns.grid$Var3[i]) + 
                    ns(Resistin, ns.grid$Var4[i]) + 
                    ns(Insulin, ns.grid$Var5[i]) + 
                    Leptin + Adiponectin, data = mydata)[, -1]
  set.seed(20221217)
  basis.cv.out <- cv.glmnet(as.matrix(x), y, alpha = 1, nfolds = 5, 
                            family = "binomial", type.measure = "auc")
  
  bestlam = basis.cv.out$lambda.min
  
  set.seed(20221217)
  basis.cv.out.acc <- cv.glmnet(as.matrix(x), y, alpha = 1, nfolds = 5, 
                                family="binomial", type.measure = "class")
  basis.mean.acc[i] <- 1 - basis.cv.out.acc$cvm[which(basis.cv.out.acc$lambda == bestlam)]
  
  basis.mean.auc[i] <- max(basis.cv.out$cvm)
  basis.max.auc[i] <- basis.cv.out$cvup[which.max(basis.cv.out$cvm)]
  basis.min.auc[i] <- basis.cv.out$cvlo[which.max(basis.cv.out$cvm)]
}

max(basis.mean.auc)
which.max(basis.mean.auc)
ns.grid[which.max(basis.mean.auc ),]
ns.grid.result <- ns.grid[which.max(basis.mean.auc ),]
colnames(ns.grid.result) <- c("Age", "BMI", "Glucose", "Insulin", "Resistin")
knitr::kable(ns.grid.result)

basis.auc <- rbind(basis.mean.auc, basis.max.auc, basis.min.auc)


matplot(t(basis.auc)[, 1], type = "l", lty = c(1, 2, 2), col = c("darkgreen", "gray", "gray"),
        xlab = "ns.grid", ylab = "AUC")
abline(v = which.max(basis.mean.auc), col = "red", lwd = 0.7, lty = 2)
max(basis.mean.auc)


basis.mean.acc[which.max(basis.mean.auc)]
max(basis.mean.auc)
basis.perf <- data.frame(max(basis.mean.auc), basis.mean.acc[which.max(basis.mean.auc)])
basis.perf <- round(basis.perf, 4)
colnames(basis.perf) <- c("AUC", "Accuracy")
rownames(basis.perf) <- "GAM"
basis.perf <- t(basis.perf)
basis.perf
knitr::kable(basis.perf)

