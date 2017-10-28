#####read in hog feature and labels

features_hog <- read.csv('~/Desktop/Data Science/project3/training_set/hog.csv',header=TRUE)
dim(features_hog)
label_train<-read.csv('~/Desktop/Data Science/project3/training_set/label_train.csv')
dim(label_train)
y<-label_train[,2]
traindata<-cbind(y,features_hog)

train.rf<- function(traindata) {
  
  traindata$y<- as.factor(traindata$y)
  y.index<- which(colnames(traindata)=="y")
  bestmtry <- tuneRF(y= traindata$y, x= traindata[,-y.index], stepFactor=1.5, improve=1e-5, ntree=600)
  best.mtry <- bestmtry[,1][which.min(bestmtry[,2])]
  
  model.rf <- randomForest(y ~ ., data = traindata, ntree=600, mtry=best.mtry, importance=T)
  return(model.rf)
}

test.rf <- function(model,test.data) {
  pred <- predict(model, test.data, type = "class")
  return(pred)
}

cv.function<-function(traindata, K=5){
  n = dim(traindata)[1]
  n.fold = floor(n/K)
  s = sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error = rep(NA, K)
  
  for (i in 1:K){
    train.data = traindata[s != i,]
    test.data = traindata[s == i,]
    test.label = traindata[s == i,1]
    
    rf.model = train.rf(train.data)
    pred = test.rf(rf.model,test.data)
    
    cv.error[i] = mean(pred != test.label) 
  }			
  return(c(mean(cv.error),sd(cv.error)))
}

cv.function(traindata,K=5)  #K=5

