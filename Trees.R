###Regression Tree

####Setup####

library(AppliedPredictiveModeling)
data(concrete)

concrete.raw <- concrete #create a backup

str(concrete) #quick look at data

head(concrete) #look at top
tail(concrete) #look at bottom

colSums(is.na(concrete)) #check for missing values

concrete.dups <- concrete[!duplicated(concrete), ]
dim(concrete)-dim(concrete.dups) #duplicate rows
concrete <- concrete[!duplicated(concrete), ] #remove duplicates

#Split Dataset
train <- concrete[which(concrete$Age<=28),]
train <- train[,-8]
str(train)
test <- concrete[which(concrete$Age>28),]
ytest <- test[,9]
test <- test[,-c(8,9)]
str(test)


#### Fitting Regression Tree ####
library(tree)
#original tree
tree.concrete <- tree(CompressiveStrength~.,train)
summary(tree.concrete)
plot(tree.concrete)
text(tree.concrete,pretty=0)

#cross validation to determine tree size
cv.concrete <- cv.tree(tree.concrete)
plot(cv.concrete$size,cv.concrete$dev,type='b')

#prune tree at 7 nodes
prune.concrete <- prune.tree(tree.concrete,best=7)
plot(prune.concrete)
text(prune.concrete,pretty=0)

#prediction plot - original tree (full tree per cv results)
yhat.orig=predict(tree.concrete,newdata=test)
plot(yhat.orig,ytest)
abline(lm(ytest~yhat.orig), col="red")
orig.mse <- mean((yhat.orig-ytest)^2)
(orig.rmse <- sqrt(orig.mse))

AllTestResults$FullTree <- yhat.orig
TestRMSE$FullTree <- orig.rmse

#prediction plot - pruned tree
yhat.prune=predict(prune.concrete,newdata=test)
plot(yhat.prune,ytest)
abline(lm(ytest~yhat.prune), col="red")
prune.mse <- mean((yhat.prune-ytest)^2)
(prune.rmse <- sqrt(prune.mse))

AllTestResults$PruneTree <- yhat.prune
TestRMSE$PruneTree <- prune.rmse

#### Bagging and Random Forests ####
library(randomForest)
set.seed(1)

#Bagging (Note: bagging is a special case of a random forest with m = p)
bag.concrete <- randomForest(CompressiveStrength~.,data=train,mtry=8,importance=TRUE)
bag.concrete

#prediction plot - bagged tree
yhat.bag <- predict(bag.concrete,newdata=test)
plot(yhat.bag, ytest)
abline(lm(ytest~yhat.bag), col="red")
bag.mse <- mean((yhat.bag-ytest)^2)
(bag.rmse <- sqrt(bag.mse))

round(importance(bag.concrete), 2)
varImpPlot(bag.concrete)
#getTree(bag.concrete, k=1, labelVar=TRUE)

AllTestResults$BagTree <- yhat.bag
TestRMSE$BagTree <- bag.rmse

#Random Forest (Note: by default RF library uses p/3 variables for building tree)
#(Also note: Use a small value of m when we have a large number of correlated predictors)
set.seed(1)
rf.concrete <- randomForest(CompressiveStrength~.,data=train,mtry=3,importance=TRUE)
rf.concrete

#prediction plot - random forest
yhat.rf <- predict(rf.concrete,newdata=test)
plot(yhat.rf, ytest)
abline(lm(ytest~yhat.rf), col="red")
rf.mse <- mean((yhat.rf-ytest)^2)
(rf.rmse <- sqrt(rf.mse))

round(importance(rf.concrete), 2)
varImpPlot(rf.concrete)

AllTestResults$RandomForest <- yhat.rf
TestRMSE$RandomForest <- rf.rmse

#### Boosting ####

library(gbm)
set.seed(1)
boost.concrete <- gbm(CompressiveStrength~.,data=train,distribution="gaussian",n.trees=10000,interaction.depth=1, shrinkage=0.001,cv.folds=10,verbose=F)

gbm.perf(boost.concrete, method="cv") #10,000 trees is appropriate

summary(boost.concrete)

#partial dependence plots (marginal effect of the selected variables on the response after integrating out the other variables)
plot(boost.concrete,i="Cement")
plot(boost.concrete,i="Water")
plot(boost.concrete,i="Superplasticizer")


#prediction plot - boosted tree
yhat.boost <- predict(boost.concrete, newdata=test, n.trees=10000)
plot(yhat.boost, ytest)
abline(lm(ytest~yhat.boost), col="red")
boost.mse <- mean((yhat.boost-ytest)^2)
(boost.rmse <- sqrt(boost.mse))

AllTestResults$BoostTree <- yhat.boost
TestRMSE$BoostTree <- boost.rmse

#### Results ####
head(AllTestResults)
TestRMSE

#scatterplot of observations vs predicted values
featurePlot(AllTestResults[, -c(1:4)], AllTestResults$obs,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth"),
            main = "Observed vs. Predicted")


