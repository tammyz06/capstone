#### Prepare Data  ####
#Split Dataset
train <- concrete[which(concrete$Age<=28),]
train <- train[,-8]
str(train)
test <- concrete[which(concrete$Age>28),]
ytest <- test[,9]
test <- test[,-c(8,9)]
str(test)

# Create a control function for cross-fold validation 
library(caret)
set.seed(1)
indx <- createFolds(train$CompressiveStrength, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

####  MLM  ####
library(car)
mlm <- lm(CompressiveStrength~., train)
summary(mlm)
par(mfrow=c(2,2))
plot(mlm) 
ncvTest(mlm) #may have unequal error variances
vif(mlm) #no multicollinearity all values less than 5

#MLM without Superplasticizer
mlm2 <- lm(CompressiveStrength~.-Superplasticizer, train)
summary(mlm2)
par(mfrow=c(2,2))
plot(mlm2) 
ncvTest(mlm2)

mlm2pred <- predict(mlm2, test)
mlm2testResults <- data.frame(obs = ytest, pred = mlm2pred )
defaultSummary(mlm2testResults)
AllTestResults$MLMwoSP <- mlm2pred

#MLM without FlyAsh
mlm3 <- lm(CompressiveStrength~.-FlyAsh, train)
summary(mlm3)
par(mfrow=c(2,2))
plot(mlm3) 
ncvTest(mlm3)

mlm3pred <- predict(mlm3, test)
mlm3testResults <- data.frame(obs = ytest, pred = mlm3pred )
defaultSummary(mlm3testResults)
AllTestResults$MLMwoFA <- mlm3pred

# CV Tuned linear regression model with all of the predictors.
set.seed(1)
lmTune0 <- train(x = train[,-8], y = train[,8],
                 method = "lm",
                 trControl = ctrl)
lmTune0
summary(lmTune0)

lmtunepred <- predict(lmTune0, test)
lmtunetestResults <- data.frame(obs = ytest, pred = lmtunepred )
defaultSummary(lmtunetestResults) #same results as mlm


####  BSS  ####
#BSS to explore interaction terms
library(leaps)
regfit.full <- regsubsets(CompressiveStrength~.^2,data=train,nvmax=28)
(reg.summary=summary(regfit.full))

par(mfrow=c(2,2))
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(17,reg.summary$adjr2[17], col="red",cex=2,pch=20)
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(13,reg.summary$cp[13],col="red",cex=2,pch=20)
which.min(reg.summary$bic)
plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
points(10,reg.summary$bic[10],col="red",cex=2,pch=20)

coef(regfit.full, 4)
coef(regfit.full, 6)
coef(regfit.full,10)

bss10 <- lm(CompressiveStrength~Cement+
              BlastFurnaceSlag+
              Water+
              Superplasticizer+
              Cement:BlastFurnaceSlag+
              BlastFurnaceSlag:FlyAsh+
              BlastFurnaceSlag:CoarseAggregate+
              BlastFurnaceSlag:FineAggregate+
              Water:Superplasticizer+
              Water:CoarseAggregate, train)
summary(bss10)
par(mfrow=c(2,2))
plot(bss10) #diagnostics are good
ncvTest(bss10) #homoskedastic


#prediction plot - BSS
par(mfrow=c(1,1))
yhat.bss <- predict(bss10,newdata=test)
plot(yhat.bss,ytest)
abline(lm(ytest~yhat.bss), col="red")
bss.mse <- mean((yhat.bss-ytest)^2)
(bss.rmse <- sqrt(bss.mse))

# bsstestResults <- data.frame(obs = ytest, pred = bsspred )
# defaultSummary(bsstestResults)

# Save the test set results in a data frame                 
AllTestResults <- data.frame(obs = ytest, BSS = yhat.bss)
TestRMSE <- data.frame(Test = "RMSE", BSS = bss.rmse)



####  WLS  ####
#WLS to rectify heteroskedacity
mlm.weights <- 1 / lm(abs(mlm$residuals) ~ mlm$fitted.values)$fitted.values^2
wls <- lm(CompressiveStrength~., data=train, weights=mlm.weights)
summary(wls)
par(mfrow=c(2,2))
plot(wls) 
ncvTest(wls) #constant variance

#prediction plot - WLS
par(mfrow=c(1,1))
yhat.wls <- predict(wls,newdata=test)
plot(yhat.wls,ytest)
abline(lm(ytest~yhat.wls), col="red")
wls.mse <- mean((yhat.wls-ytest)^2)
(wls.rmse <- sqrt(wls.mse))

AllTestResults$WLS <- yhat.wls
TestRMSE$WLS <- wls.rmse


####  PLS  ####
#PLS to reduce the model
library(caret)
set.seed(1)
pls <- train(x = train[,-8], y = train[,8],
             method = "pls",
             tuneGrid = expand.grid(ncomp = 1:8),
             trControl = ctrl)
pls
plot(pls)
plsImp <- varImp(pls, scale = FALSE)
plot(plsImp, top = 7, scales = list(y = list(cex = .95)))


#prediction plot - PLS
par(mfrow=c(1,1))
yhat.pls <- predict(pls,newdata=test)
plot(yhat.pls,ytest)
abline(lm(ytest~yhat.pls), col="red")
pls.mse <- mean((yhat.pls-ytest)^2)
(pls.rmse <- sqrt(pls.mse))

AllTestResults$PLS <- yhat.pls
TestRMSE$PLS <- pls.rmse

####Regression Results ####
head(AllTestResults)
TestRMSE


#scatterplot of observations vs predicted values
featurePlot(AllTestResults[, -1], AllTestResults$obs,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth"),
            main = "Observed vs. Predicted")
