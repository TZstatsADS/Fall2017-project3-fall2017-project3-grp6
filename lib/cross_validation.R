### Cross Validation ###
########################

cv.function <- function(data, K){
  
  n <- nrow(data)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))
  cv.error.BP <- rep(NA, K)
  cv.error.rf <- rep(NA, K)
  cv.error.svm <- rep(NA, K)
  cv.error.log <- rep(NA, K)
  cv.error.xgboost<- rep(NA, K)
  
  for (i in 1:K){
    train.data <- data[s != i,]
    test.data <- data[s == i,]

    fit.BP <- train.bp(train.data)
    pred.BP <- test.bp(fit.BP, test.data)  
    cv.error.BP[i] <- mean(pred.BP != test.data$y)
    
    fit.rf <- train.rf(train.data)
    pred.rf <- test.rf(fit.rf, test.data)  
    cv.error.rf[i] <- mean(pred.rf != test.data$y)
    
    fit.svm <- train.svm(train.data)
    pred.svm <- test.svm(fit.svm, test.data)  
    cv.error.svm[i] <- mean(pred.svm != test.data$y)
    
    fit.log <- train.log(train.data)
    pred.log <- test.log(fit.log, test.data)
    cv.error.log[i] <- mean(pred.log != test.data$y)
    
    fit.xgboost<-train.xgboost(train.data)
    pred.xgboost<-test.xgboost(fit.xgboost,test.data)
    cv.error.xgboost[i]<-mean(pred.xgboost != test.data$y)
  
  }			
  cv.error<- data.frame(
        bp = mean(cv.error.BP), 
        rf = mean(cv.error.rf), svm = mean(cv.error.svm), 
        logistic= mean(cv.error.log),xgboost= mean(cv.error.xgboost))
  return(cv.error)
}
