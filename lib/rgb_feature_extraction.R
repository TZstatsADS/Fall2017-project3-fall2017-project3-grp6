#############RGB
############# We are actually not using this method finally, however I keep it to show the work.
library("EBImage")

extract_rgb_feature <- function(img){
  # INPUT: an EBImage image object. Use EBImage::readImage to create from jpg
  # OUTPUT: a 1000-dimensional vector of color frequencies in img
  
  mat <- imageData(img)
  
  rBin = seq(0, 1, length.out=10)
  gBin = seq(0, 1, length.out=10)
  bBin = seq(0, 1, length.out=10)
  
  freq_rgb = as.data.frame(table(factor(findInterval(mat[,,1], rBin), levels=1:10), 
                                 factor(findInterval(mat[,,2], gBin), levels=1:10), 
                                 factor(findInterval(mat[,,3], bBin), levels=1:10)))
  
  rgb_feature = as.numeric(freq_rgb$Freq)/(ncol(mat)*nrow(mat)) # normalization
  
  return(rgb_feature)
}

################ Store RGB feature extraction as a n*1000 data frame
rgb <- vector()

for (i in 1:3000){
  n<-nchar(as.character(i))
  path<-paste0("D:/Fall 2017/ADS/training_set/images/img_",paste(rep(0,4-n),collapse=""),i,".jpg")
  a <- readImage(path)
  rgb <- rbind(rgb,extract_rgb_feature(a))
}

write.csv(rgb,file = "rgb.csv")
