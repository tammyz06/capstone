#### EDA ####

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


#raw data histograms
library(ggpubr)
gghistogram(concrete, x=colnames(concrete),
            fill = "#0073C2FF", 
            color = "#0073C2FF",
            add = "mean", 
            rug = TRUE) 

#summary table
library(summarytools)
dfs <- dfSummary(concrete)
dfs$Valid <- NULL
dfs$Missing <- NULL
view(dfs)

#skewness statistic
library(e1071)
(skewValues <- apply(concrete, 2, skewness)) #BlastFurnaceSlag and Superplasticizer are highly right skewed

#FlyAsh appears skewed because of the high number of zero values (zero variance predictor - consider removing? APM pg.44)
summary(concrete$FlyAsh)
FAzeros <- concrete$FlyAsh ==0
summary(FAzeros) #more than half are zeroes

library(caret)
nearZeroVar(concrete)
# To filter for near-zero variance predictors, the caret package function nearZeroVar will return the column numbers of any predictors that fulfill the conditions outlined in APM Sect. 3.5.


####Use caret's preProcess function to transform for skewness and standardize (APM pg 54)####

#add constant +1 for non-infinity log results
concrete$BlastFurnaceSlag <- concrete$BlastFurnaceSlag+1
concrete$FlyAsh <- concrete$FlyAsh+1
concrete$Superplasticizer <- concrete$Superplasticizer+1

library(caret)
(conPP <- preProcess(concrete[,-8], method = c("BoxCox","center", "scale")))
concrete <- predict(conPP, concrete)  #Apply the transformations
head(concrete)

#transformed data histograms and skewness stat
gghistogram(concrete, x=colnames(concrete),
            fill = "#0073C2FF", 
            color = "#0073C2FF",
            add = "mean", 
            rug = TRUE) 

(skewValues <- apply(concrete, 2, skewness))



####correlations  (APM pg. 56) ####
library(corrplot)
(correlations <- cor(concrete))
corrplot(correlations, order = "hclust")
findCorrelation(correlations, cutoff = .60) #Superplasticizer is correlated with FlyAsh & Water - consider for deletion


#scatterplot of each variable by Compressive Strength
featurePlot(concrete[, -9], concrete$CompressiveStrength,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth"),
            main = "Scatterplots by Compressive Strength")
