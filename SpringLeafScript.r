setwd ("C:/Users/Aroon/Documents/Kaggle/Springleaf Marketing Response")


data.train <- read.csv("train.csv")
data.test <- read.csv("test.csv")

data.train <- as.data.frame(data.train[,c(-1)])  ## remove "ID" field
data.test <- as.data.frame(data.test[,c(-1)])

feature.names <- names(data.train)
for (f in feature.names){
	data.train[,f] <- as.vector(data.train[,f])
	}

feature.names <- names(data.test)	
for (f in feature.names){
	data.test[,f] <- as.vector(data.test[,f])
	}
	
## descibe the data. Evaluate the output for feature engineering opportunities

### remove variables with zero mean and zero sd

# first means
feature.names <- names(data.train)
key <- c()
meanvalue <- c()
sdvalue <- c()

for (f in feature.names) {
	if (class(data.train[,f]) == "integer") {
	key <- append(key,as.vector(f))
	meanvalue <- append(meanvalue, as.vector(mean(data.train[,f], na.rm=T)))
	sdvalue <- append(sdvalue, as.vector (sd(data.train[,f], na.rm=T)))
	}
	}

kvdf <- as.data.frame(cbind(key,meanvalue,sdvalue))
kvdfsub <- as.data.frame(subset(kvdf, meanvalue == 0))
kvdfsub <- as.vector(kvdfsub[,c(-2,-3)])
	
data.train <- data.train[, !names(data.train) %in% kvdfsub]	
data.test <- data.test[, !names(data.test) %in% kvdfsub]

# now standard deviation
feature.names <- names(data.train)
key <- c()
meanvalue <- c()
sdvalue <- c()
	
for (f in feature.names) {
	if (class(data.train[,f]) == "integer") {
	key <- append(key,as.vector(f))
	meanvalue <- append(meanvalue, as.vector(mean(data.train[,f], na.rm=T)))
	sdvalue <- append(sdvalue, as.vector (sd(data.train[,f], na.rm=T)))
	}
	}

kvdf <- as.data.frame(cbind(key,meanvalue,sdvalue))
kvdfsub <- as.data.frame(subset(kvdf, sdvalue == 0))
kvdfsub <- as.vector(kvdfsub[,c(-2,-3)])
	
data.train <- data.train[, !names(data.train) %in% kvdfsub]	
data.test <- data.test[, !names(data.test) %in% kvdfsub]

# remove "character" variables (after inspection) that have just one factor

feature.names <- names(data.train)
kv <- list()	
for (f in feature.names) {
	if (class(data.train[,f]) == "character" & length(unique(data.train[,f])) < 3) {
	key <- f
	value <- unique(data.train[,f])
	kv[[key]] <- value
	}
	}

remov <- names(kv)
data.train <- data.train[, !names(data.train) %in% remov]
data.test <- data.test[, !names(data.test) %in% remov]
# lets evaluate the remaining "character variables"	

# VAR_0073, VAR_0075, VAR_0156, VAR_0157, VAR_0158, VAR_0159, VAR_0166, VAR_0167, VAR_0168, VAR_0169, VAR_0176, VAR_0177, VAR_0178, VAR_0179, VAR_0217, 
# VAR_0204 has interesting date data
# VAR_0237, VAR_0200, VAR_0274,  has geographic data

data.train$VAR_0073 <- difftime(strptime(data.train$VAR_0073, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0075 <- difftime(strptime(data.train$VAR_0075, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0156 <- difftime(strptime(data.train$VAR_0156, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0157 <- difftime(strptime(data.train$VAR_0157, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0158 <- difftime(strptime(data.train$VAR_0158, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0159 <- difftime(strptime(data.train$VAR_0159, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0166 <- difftime(strptime(data.train$VAR_0166, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0167 <- difftime(strptime(data.train$VAR_0167, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0168 <- difftime(strptime(data.train$VAR_0168, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0169 <- difftime(strptime(data.train$VAR_0169, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0176 <- difftime(strptime(data.train$VAR_0176, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0177 <- difftime(strptime(data.train$VAR_0177, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0178 <- difftime(strptime(data.train$VAR_0178, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0179 <- difftime(strptime(data.train$VAR_0179, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0217 <- difftime(strptime(data.train$VAR_0217, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.train$VAR_0204 <- difftime(strptime(data.train$VAR_0204, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))

data.test$VAR_0073 <- difftime(strptime(data.test$VAR_0073, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0075 <- difftime(strptime(data.test$VAR_0075, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0156 <- difftime(strptime(data.test$VAR_0156, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0157 <- difftime(strptime(data.test$VAR_0157, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0158 <- difftime(strptime(data.test$VAR_0158, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0159 <- difftime(strptime(data.test$VAR_0159, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0166 <- difftime(strptime(data.test$VAR_0166, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0167 <- difftime(strptime(data.test$VAR_0167, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0168 <- difftime(strptime(data.test$VAR_0168, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0169 <- difftime(strptime(data.test$VAR_0169, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0176 <- difftime(strptime(data.test$VAR_0176, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0177 <- difftime(strptime(data.test$VAR_0177, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0178 <- difftime(strptime(data.test$VAR_0178, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0179 <- difftime(strptime(data.test$VAR_0179, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0217 <- difftime(strptime(data.test$VAR_0217, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))
data.test$VAR_0204 <- difftime(strptime(data.test$VAR_0204, format='%d%b%H:%M:%S') , as.POSIXlt("2015-01-01"), units = c("days"))

## convert dates to numerics
feature.names <- names(data.train)

for (f in feature.names) {
	if (class(data.train[,f]) == "difftime") {
	data.train[,f] <- as.numeric(data.train[,f])
	data.test[,f] <- as.numeric(data.test[,f])

	}
	}	

for (f in feature.names) {
	if (class(data.train[,f]) == "character") {
	data.train[,f] <- as.factor(data.train[,f])
	data.test[,f] <- as.factor(data.test[,f])

	}
	}	

# we will remove VAR200 and VAR404 because the number of factors for these exceed 1024, and these cannot be handled by most models. And the others because they have only NAs in them
remov = c("VAR_0404", "VAR_0200", "VAR_0207", "VAR_0213", "VAR_0840")
data.train <- data.train[, !names(data.train) %in% remov]
data.test <- data.test[, !names(data.test) %in% remov]	
# lets do some clean up
rm(kvdf,f,feature.names,key,kv,kvdfsub,meanvalue,remov,sdvalue,value, maxima, minima, sumna, structure,NAdf)
gc()

# replacing all NAs with -9999999999 on integer or numeric variables

feature.names <- names(data.train)
for (f in feature.names) {
	if(sum(is.na(data.train[,f]) > 0)) {
	data.train[,f][is.na(data.train[,f])] <- -9999999999
	}
	}

feature.names <- names(data.test)
for (f in feature.names) {
	if(sum(is.na(data.test[,f]) > 0)) {
	data.test[,f][is.na(data.test[,f])] <- -9999999999
	}
	}

rm(kvdf,f,feature.names,key,kv,kvdfsub,meanvalue,remov,sdvalue,value, maxima, minima, sumna, structure,NAdf)
gc()	
################################## let the initial analysis begin

#separate test and train
set.seed(123)
train= sample(c(TRUE,TRUE,TRUE,FALSE), nrow(data.train),rep=TRUE)
test = (!train)

data.train.train <- as.data.frame(data.train[train,])
data.train.test <- as.data.frame(data.train[test,])

#lets try boost first
# 
boost.fit <- gbm(target ~ ., data = data.train.train, distribution= "bernoulli",n.trees =100 , shrinkage =0.1, interaction.depth = 1, n.cores=8, verbose = T)

boost.probs=predict(boost.fit,data.train.test, n.trees = 100, type = "response")	
boost.ROCR = prediction(boost.probs, data.train.test$target)
boost.perfROCR = performance(boost.ROCR, "tpr", "fpr")
plot(boost.perfROCR, colorize=TRUE)
performance(boost.ROCR, "auc")@y.values[[1]]
	
boost.fit <- gbm(target ~ ., data = data.train.train, distribution= "bernoulli",n.trees =500 , shrinkage =0.02, interaction.depth = 1, n.cores=8, verbose = T)

boost.fit <- gbm(target ~ ., data = data.train.train, distribution= "bernoulli",n.trees =100 , shrinkage =0.1, interaction.depth = 2, n.cores=8, verbose = T)	
	
boost.probs2=predict(boost.fit,data.train.test, n.trees = 100, type = "response")		
boost.ROCR = prediction(boost.probs2, data.train.test$target)
boost.perfROCR = performance(boost.ROCR, "tpr", "fpr")
plot(boost.perfROCR, colorize=TRUE)
performance(boost.ROCR, "auc")@y.values[[1]]	
	
boost.fit <- gbm(target ~ ., data = data.train.train, distribution= "bernoulli",n.trees =100 , shrinkage =0.1, interaction.depth = 4, n.cores=8, verbose = T)	
	
boost.probs3=predict(boost.fit,data.train.test, n.trees = 100, type = "response")		
boost.ROCR = prediction(boost.probs3, data.train.test$target)
boost.perfROCR = performance(boost.ROCR, "tpr", "fpr")
plot(boost.perfROCR, colorize=TRUE)
performance(boost.ROCR, "auc")@y.values[[1]]		
###########################################################################
	
## we will now reduce the data to its principal components	

# separate label from predictors	
remov <- c("target")
data.train.label <- data.train[,names(data.train) %in% remov]
data.train <- data.train[,!names(data.train) %in% remov]
# before merging add markers for train and test
data.train$train <- 1
data.test$train <- 0
# merge test and train data together
mergedata <- rbind(data.train,data.test)
# create dummy variables
library(caret)
dmy <- dummyVars(~., data = mergedata, fullRank = T)
transformeddata <- data.frame(predict(dmy, newdata = mergedata))
rm(mergedata, remov)
rm(dmy)
gc()
# rebuild test and train data
data.train <- subset(transformeddata, train ==1)
data.test <- subset(transformeddata, train ==0)
data.train$train <- NULL
data.test$train <- NULL
rm(transformeddata)
gc()
# lets look at near zero variances and remove variables that have variances less than 0.01%
nzv <- nearZeroVar(data.train, saveMetrics = TRUE)
nzv.train <- data.train[c(rownames(nzv[nzv$percentUnique > 0.01,])) ]
nzv.test <- data.test[c(rownames(nzv[nzv$percentUnique > 0.01,])) ]
#nzv.train$X <- NULL ### this has to be done if we load the newly created data at line 240
#nzv.test$X <- NULL


#lets try taking this smaller dataset through xgboost

set.seed(123)
train = sample(c(TRUE,TRUE,TRUE,FALSE), nrow(nzv.train),rep=TRUE)
test = (!train)
data.train.label <- as.data.frame(data.train.label)
colnames(data.train.label) <- c("target")

data.x.train <- as.matrix(nzv.train[train,])
data.y.train <- as.matrix(data.train.label[train,])

data.x.test <- as.matrix(nzv.train[test,])
data.y.test <- as.matrix(data.train.label[test,])

library(xgboost)

cv.mod <- xgb.cv(data = data.x.train, nfold = 5, verbose = T, label = data.y.train, nrounds = 200, 
                 objective = 'binary:logistic', eval_metric = 'auc', eta = 0.1, max_depth = 10,
                 gamma = 0.5, min_child_weight = 0.8)

plot(cv.mod$test.auc.mean)					 
gc()
## build model and predict
					
xgbmod <- xgboost(data = data.x.train, label = data.y.train, nrounds = 200, 
					objective = 'binary:logistic', eval_metric = 'auc', eta = 0.1, max_depth = 10,
					gamma = 0.5, min_child_weight = 0.8)

testpred <- predict(xgbmod, data.x.test)			
library(ROCR)
# calculate auc for the model

xgb.ROCR = prediction(testpred, data.y.test)
xgb.perfROCR = performance(xgb.ROCR, "tpr", "fpr")
plot(xgb.perfROCR, colorize=TRUE)
performance(xgb.ROCR, "auc")@y.values


#################### miscellaneous code below this useful kv implementation in R	


feature.names <- names(data.train)

key <- c()
sumna <- c()
minima <- c()
maxima <- c()

for (f in feature.names) {

	key <- append(key,f)
	sumna <- append(sumna, sum(is.na(data.train[,f])))

	}
NAdf <- cbind(key,sumna, minima, maxima)
write.csv(NAdf, "NAdf.csv")

'''
library(psych)
datadetail <- describe(data.train)
write.csv(datadetail, "datadetail.csv")
structure <- capture.output(str(data.train, list.len=2000))
write.csv(structure, "datastr.csv")
var200 <- as.data.frame(table(data.train$VAR_0200))
write.csv(var200, "var200.csv")
allNA <- as.data.frame(sapply(data.train, function(x)all(is.na(x))))
write.csv(allNA, "allNA.csv")

# evaluate the csv output for feature engineering opportunity

# as an example VAR159 seems to have time data
data.train$VAR_0159 <- as.vector(data.train$VAR_0159)
r <- regexpr("[A-Z][A-Z][A-Z]",data.train$VAR_0159)
data.train$month <- ifelse(grepl("[A-Z][A-Z][A-Z]",data.train$VAR_0159)==TRUE,regmatches(data.train$VAR_0159, r),"")

data.test$VAR_0159 <- as.vector(data.test$VAR_0159)
r <- regexpr("[A-Z][A-Z][A-Z]",data.test$VAR_0159)
data.test$month <- ifelse(grepl("[A-Z][A-Z][A-Z]",data.test$VAR_0159)==TRUE,regmatches(data.test$VAR_0159, r),"")
'''

''' n.trees 500 not recommended as the gain from the previous boost was limited
boost.probs1=predict(boost.fit,data.train.test, n.trees = 500, type = "response")		
boost.ROCR = prediction(boost.probs1, data.train.test$target)
boost.perfROCR = performance(boost.ROCR, "tpr", "fpr")
plot(boost.perfROCR, colorize=TRUE)
performance(boost.ROCR, "auc")@y.values[[1]]	
'''
'''
nzv.train <- data.train[,names(data.train) %in% keep]
nzv.test <- data.test[,names(data.test) %in% keep]
'''
'''
feature.names <- names(data.train)
key <- c()
for (f in feature.names) {
	if (class(data.train[,f]) == "character") {
	key <- append(key,f)

	}
	}	

key
'''

#lets count the number of NAs in each of the remaining columns
feature.names <- names(nzv.train)

key <- c()
sumna <- c()
minima <- c()
maxima <- c()

for (f in feature.names) {
	key <- append(key,f)
	sumna <- append(sumna, sum(is.na(nzv.train[,f])))
	minima <- append(minima, min(nzv.train[,f], na.rm=T))
	maxima <- append(maxima, max(nzv.train[,f], na.rm=T))
	}
NAdf <- cbind(key,sumna, minima, maxima)
write.csv(NAdf, "NAdf.csv")



# we will center and scale the training data
segPP <- preProcess(nzv.train, c("center","scale"))
nzv.train.tr <- predict(segPP, nzv.train)

'''
# and now for the principal comps for train data
nzv.train.pc <- prcomp(~ ., data = nzv.train, scale = T)
'''