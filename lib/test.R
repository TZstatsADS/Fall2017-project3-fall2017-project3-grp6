library(gbm)
library(caret)
library(DMwR)
library(nnet)
library(randomForest)
library(e1071)

############ gbm ######################

test.gbm <- function(model, test.data)
{
  pred<- predict(model$fit, newdata = test.data, n.trees = model$iter, type="response")
  pred<-data.frame(pred)
  return(apply(pred,1,which.max)-1)
}

############ BP network ######################
test.bp=function(model,test.data)
{
  pre.nnet <- predict(model, test.data, type = "class")
  return(pre.nnet)
}

############ Random Forest ######################
test.rf <- function(model,test.data) {
  return(predict(model, test.data, type = "class"))
}


############ SVM ######################
test.svm <- function(model,test.data)
{
  return(predict(model,test.data,type="class"))
}

############ Logistic ######################
test.log <- function(model, test_data) 
{
  glm.pred<-predict(model, test_data, type = "class")
  
#  result<- predict(model,newdata =test_data,type = 'response')
#  fitted.results <- ifelse(result>0.5,1,0)
#  return(fitted.results)
}
##############xgboost########################
test.xgboost = function(model,test_data){
  testtt<-as.matrix(test_data)
  test_pred <- predict(model, newdata = testtt[,-1])
  return(test_pred)
}
