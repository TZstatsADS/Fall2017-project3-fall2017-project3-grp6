library("data.table")

train.baseline<-function(X, y, depth, shrinkage){
  library('gbm')
  fit_gbm = gbm.fit(X, y,
                    distribution = "multinomial",
                    n.trees = 250,
                    interaction.depth = depth, 
                    shrinkage = shrinkage,
                    bag.fraction = 0.5,
                    verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)
  return(list(fit=fit_gbm, iter=best_iter))
}

test.baseline = function(fit_train, dat_test){
  library("gbm")
  pred <- predict(fit_train$fit, newdata = dat_test, 
                             n.trees = fit_train$iter, 
                             type="response")
  
  return(as.numeric(pred> 0.5))
}


cv.function<-function(X, y, depth, shrinkage, K=5){
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

depths = c(3, 5, 7, 9, 11)
shrinkages = 0.1

err_cv = array(dim=c(length(depths),2))