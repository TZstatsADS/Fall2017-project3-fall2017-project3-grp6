library(data.table)
source("./lib/train.R")
source("./lib/test.R")
source("./lib/cross_validation.R")


### Import matrix of features X and create vector of labels y
#####read in SIFT feature and labels
features <- read.csv('~/Desktop/Data Science/project3/training_set/sift_train.csv')
dim(features)
label_train<-read.csv('~/Desktop/Data Science/project3/training_set/label_train.csv')
dim(label_train)
y<-label_train[,2]
X<-features[,-1]

train<-function(X, y, depth, shrinkage){
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

test = function(fit_train, dat_test){
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

# Cross-validation: choosing between different values of depth and shrinkage for GBM

#depths = c(1, 3, 5, 7, 9)
depths = c(3, 5, 7, 9, 11)
#shrinkages = c(.0001, .001, .01, .1, .5)
shrinkages = 0.1

#cv_output = array(dim=c(length(depths), length(shrinkages), 4))
err_cv = array(dim=c(length(depths),2))

#for(i in 1:length(depths)){
  #for(j in 1:length(shrinkages)){
   # cat("i=", i,", j=", j, "\n")
   # cv_output[i,j,] <- cv.function(X, y, depth=depths[i], shrinkage=shrinkages[j], K=5)
 # }
#}

for(k in 1:length(depths)){
  cat("k=", k, "\n")
  err_cv[k,] <- cv.function(X, y, depths[k], shrinkage=shrinkages, K=5)  #K=5
}

################Choose the best parameter value
#depth_best <- depths[which.min(cv_output[,1])]
#par_best <- list(depth=depth_best)

################train the model with the entire training set
#tm_train <- system.time(fit_train <- train(dat_train, label_train, par_best))
#save(fit_train, file="./output/fit_train.RData")
