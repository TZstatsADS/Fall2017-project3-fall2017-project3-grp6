### Import matrix of features X and create vector of labels y

train<-function(X, y, depth, shrinkage){
  fit_gbm = gbm.fit(X, y,
                    distribution = "multinomial",
                    n.trees = 10,
                    interaction.depth = depth, 
                    shrinkage = shrinkage,
                    bag.fraction = 0.5,
                    verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)
  return(list(fit=fit_gbm, iter=best_iter))
}
###############
test = function(fit_train, dat_test){
  pred<- predict(fit_train$fit, newdata = dat_test, 
                             n.trees = fit_train$iter, 
                             type="response")
  pred<-data.frame(pred)
  colnames(pred)<-c(0,1,2)
  return(apply(pred,1,which.max)-1)
}

#######################
cv.function<-function(X, y, depth, shrinkage, K=3){
  n = length(y)
  n.fold = floor(n/K)
  s = sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error = rep(NA, K)
  
  for (i in 1:K){
    train.data = X[s != i,]
    train.label = y[s != i]
    test.data = X[s == i,]
    test.label = y[s == i]
    
    fit = train(train.data, train.label, depth, shrinkage)
    pred = test(fit, test.data)  
    cv.error[i] = mean(pred != test.label) 
  }			
  return(c(mean(cv.error),sd(cv.error)))
}

# Cross-validation: choosing between different values of depth and shrinkage for GBM

depths = c(3, 5, 7, 9, 11) #3
shrinkages = 0.1


err_cv = array(dim=c(length(depths),2))

for(k in 1:length(depths)){
  cat("k=", k, "\n")
  err_cv[k,] <- cv.function(X, y, depths[k], shrinkage=shrinkages, K=5)  #K=5
}

err_cv