library(readr)
library(xgboost)
library(gbm)
library(caret)
library(DMwR)
library(nnet)
library(randomForest)
library(e1071)

############ gbm ######################
train.gbm<-function(traindata){
  X=traindata[,-1]
  y=traindata[,1]
  fit_gbm = gbm.fit(X, y,
                    distribution = "multinomial",
                    n.trees = 10,
                    interaction.depth = 3, 
                    shrinkage = 0.1,
                    bag.fraction = 0.5,
                    verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)
  return(list(fit=fit_gbm, iter=best_iter))
}


############ BP network ######################
train.bp<- function(traindata) {
  traindata$y<- as.factor(traindata$y)
  model.nnet <- nnet(y ~ ., data = traindata, linout = F,
                     size = 1, decay = 0.01, maxit = 200,
                     trace = F, MaxNWts=6000)
  return(model.nnet)
}

############ Random Forest ######################
# First tune random forest model, tune parameter 'mtry'
train.rf<- function(traindata) {
  
  traindata$y<- as.factor(traindata$y)
  y.index<- which(colnames(traindata)=="y")
  bestmtry <- tuneRF(y= traindata$y, x= traindata[,-y.index], stepFactor=1.5, improve=1e-5, ntree=600)
  best.mtry <- bestmtry[,1][which.min(bestmtry[,2])]
  
  
  model.rf <- randomForest(y ~ ., data = traindata, ntree=600, mtry=best.mtry, importance=T)
  return(model.rf)
}

############ SVM ######################
train.svm<- function(traindata) {
  traindata$y<- as.factor(traindata$y)
  model.svm<- svm(y~., data = traindata,cost=100, gamma=0.01)
  return(model.svm)
}

############ Logistic ######################
train.log <- function(train_data){
  model.log = multinom(y~., data=train_data,trace = F)
  
  return(model.log)
}
#############xgboost########################
train.xgboost = function(training){
  trainnn<-as.matrix(training)
  dtrain=xgb.DMatrix(data=trainnn[,-1],label=trainnn[,1])
  param <- list("objective" = "multi:softmax",
                "eval_metric" = "mlogloss",
                "num_class" = 3, 'eta' = 0.3, 'max_depth' = 4)
  bst <- xgb.train(data = dtrain,  param = param, nrounds = 1000)
  return(bst)
}
#############Final Model:
