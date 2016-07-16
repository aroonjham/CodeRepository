library("twitteR")
library("wordcloud")
library("tm")

api_key <- "YOUR API KEY"
api_secret <- "YOUR API SECRET"
access_token <- "YOUR ACCESS TOKEN"
access_token_secret <- "YOUR ACCESS TOKEN SECRET"
setup_twitter_oauth(api_key,api_secret,access_token,access_token_secret

rem <- "marketinganalytics"
hash <- "#"
keyword <- paste0(hash,rem)

r_stats <- searchTwitter(keyword, n=1000)
r_stats_text <- sapply(r_stats, function(x) x$getText())
#r_stats_text <- unlist(strsplit(r_stats_text, split=", ")) ## optional step r_stats_text <- unlist(strsplit(r_stats_text, split=" "))
r_stats_text <- iconv(r_stats_text, "latin1", "ASCII", sub=" ")
#r_stats_text_ind <- grep("r_stats_text", iconv(r_stats_text, "latin1", "ASCII", sub=" "))
#r_stats_text <- r_stats_text[-r_stats_text_ind]

r_stats_text_corpus <- Corpus(VectorSource(r_stats_text))
r_stats_text_corpus <- tm_map(r_stats_text_corpus, content_transformer(tolower)) 
r_stats_text_corpus <- tm_map(r_stats_text_corpus, PlainTextDocument)
r_stats_text_corpus <- tm_map(r_stats_text_corpus, removePunctuation)
r_stats_text_corpus <- tm_map(r_stats_text_corpus, removeWords, c(rem, stopwords("english")))
wordcloud(r_stats_text_corpus,min.freq=2,max.words=100)


###############################################################
###															###
###															###
###The following code is for kaggle competion What's cooking###
###															###
###															###
###############################################################

################## 

setwd("C:/Users/Aroon/Documents/Kaggle/Whats_Cooking")

library(jsonlite)

train <- fromJSON("train.json")
combi <- train

x <- length(combi$ingredients)
for (i in 1:x) {
combi$ingredients[[i]] <- sub(" ","",combi$ingredients[[i]])
}

library(tm)
#library(SnowballC)
#create corpus
corpus <- Corpus(VectorSource(combi$ingredients))
corpus <- tm_map(corpus, tolower) # Convert text to lowercase
corpus <- tm_map(corpus, removePunctuation) # Remove Punctuation
corpus <- tm_map(corpus, removeWords, c(stopwords('english'))) # Remove Stopwords
corpus <- tm_map(corpus, stripWhitespace) # Remove Whitespaces
#corpus <- tm_map(corpus, stemDocument) # Perform Stemming
corpus <- tm_map(corpus, PlainTextDocument) # Convert the text into plain text document

# create a document matrix where the rows are document IDs, and the columns are recurring words and the frequency of their occurrence
frequencies <- DocumentTermMatrix(corpus)
frequencies

# lets explore the data
freq <- colSums(as.matrix(frequencies)) # sum of columns
ord <- order(freq)
freq[tail(ord)] # most common ingredients
head(table(freq),20) # you will notice that there are 494 items that appear only once as an ingredient

#create a data frame for visualization
wf <- data.frame(word = names(freq), freq = freq)
head(wf)

library(ggplot2)
chart <- ggplot(subset(wf, freq >10000), aes(x = word, y = freq))
chart <- chart + geom_bar(stat = 'identity', color = 'black', fill = 'white')
chart <- chart + theme(axis.text.x=element_text(angle=45, hjust=1))
chart

# find the level of correlation between two ingredients. Here we select the ingredients with highest frequencies
findAssocs(frequencies, c('salt','oil', 'fresh','ground', 'garlic'), corlimit=0.30)

#create wordcloud
library(wordcloud)
set.seed(142)
#plot word cloud
wordcloud(names(freq), freq, min.freq = 2500, colors = brewer.pal(8, "Spectral"))

# lets remove some sparse terms (this operation is focused on column items that occur very infrequently)
dim(frequencies) # dimension of original matrix - notice 2655 items in the columns

sparse <- removeSparseTerms(frequencies, 0.9995)
dim(sparse) # dimension of new sparse matrix - notice 1738 items in the columns

# final data wrangling
newsparse <- as.data.frame(as.matrix(sparse))
dim(newsparse)
colnames(newsparse) <- make.names(colnames(newsparse)) # make.names ensures Syntactically Valid Names
newsparse$cuisine <- as.factor(c(train$cuisine)) # add target variable to newsparse

mytrain <- newsparse

### let the modeling begin

library(xgboost)


set.seed(123)
train= sample(c(TRUE,TRUE,TRUE,FALSE), nrow(mytrain),rep=TRUE)
test = (!train)

#split data between test and train
mytrain <- newsparse[train,]
mytest <- newsparse[test,]

# create matrix that can be used with xgboost model
ctrain <- xgb.DMatrix(as.matrix(mytrain[,!colnames(mytrain) %in% c('cuisine')]), label = as.numeric(mytrain$cuisine)-1)

# lets do some cross validation
cv.mod <- xgb.cv(data = ctrain, nfold = 5, verbose = T, nrounds = 300, 
					objective = "multi:softmax", eval_metric = 'merror', eta = 0.1, max_depth = 50, num_class = 20)

					
cv.mod1 <- xgb.cv(data = ctrain, nfold = 5, verbose = T, nrounds = 300, 
					objective = "multi:softmax", eval_metric = 'merror', eta = 0.3, max_depth = 100, num_class = 20)

					
plot(cv.mod$test.merror.mean)					
lines(cv.mod1$test.merror.mean)

# create first xgboost model and predit

xgbmodel <- xgboost(data = ctrain, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 20, verbose = 1)

xgbmodel.predict <- predict(xgbmodel, newdata = as.matrix(mytest[, !colnames(mytest) %in% c('cuisine')]))
xgbmodel.predict.text <- levels(mytrain$cuisine)[xgbmodel.predict + 1] # convert numeric predicted labels into text
table(mytest$cuisine, xgbmodel.predict) # compare and contrast actual versus prediction. The diagonal elements are the ones the model predicted accurately

sum(diag(table(mytest$cuisine, xgbmodel.predict)))/nrow(mytest) # 79% accuracy

# create second xgboost model and predit

xgbmodel2 <- xgboost(data = ctrain, max.depth = 20, eta = 0.2, nrounds = 250, objective = "multi:softmax", num_class = 20, verbose = 1)

xgbmodel.predict2 <- predict(xgbmodel2, newdata = as.matrix(mytest[, !colnames(mytest) %in% c('cuisine')]))
xgbmodel.predict2.text <- levels(mytrain$cuisine)[xgbmodel.predict2 + 1] # convert numeric predicted labels into text
table(mytest$cuisine, xgbmodel.predict2) # compare and contrast actual versus prediction. The diagonal elements are the ones the model predicted accurately

sum(diag(table(mytest$cuisine, xgbmodel.predict2)))/nrow(mytest) # 79% accuracy as well

# create third xgboost model and predit

xgbmodel3 <- xgboost(data = ctrain, max.depth = 25, gamma = 2, min_child_weight = 2, eta = 0.1, nround = 250, objective = "multi:softmax", num_class = 20, verbose = 1)

xgbmodel.predict3 <- predict(xgbmodel3, newdata = as.matrix(mytest[, !colnames(mytest) %in% c('cuisine')]))
xgbmodel.predict3.text <- levels(mytrain$cuisine)[xgbmodel.predict3 + 1] # convert numeric predicted labels into text
table(mytest$cuisine, xgbmodel.predict3) # compare and contrast actual versus prediction. The diagonal elements are the ones the model predicted accurately

sum(diag(table(mytest$cuisine, xgbmodel.predict3)))/nrow(mytest) # 78% accuracy

# final step

resultmatrix <- cbind(xgbmodel.predict.text, xgbmodel.predict2.text, xgbmodel.predict3.text)

# function that extracts most common occurrence (Mode)

Mode <- function(x) {
u <- unique(x)
u[which.max(tabulate(match(x, u)))]
}

finalpred <- apply(resultmatrix,1,Mode)
sum(diag(table(mytest$cuisine, finalpred)))/nrow(mytest) # lifts the predicted value

# where do we go wrong??

library(sqldf)
results <- as.data.frame(table(mytest$cuisine, finalpred))
colnames(results) = c('originalpred','finalpred','count')

Correctdf <- sqldf("
					SELECT originalpred AS 'cuisine', count AS 'correct_count'
					FROM results
					WHERE originalpred = finalpred
                     " )
					 
Totaldf <- as.data.frame(table(mytest$cuisine))
colnames(Totaldf) = c('cuisine','Total_count')

CompleteDf = merge(Totaldf, Correctdf, all.x = TRUE)

CompleteDf$accuracy <- CompleteDf$correct_count/CompleteDf$Total_count

#### very cool way of replacing words in a large string. Something extra for text mining

x <- "I´m performing a sentiment analysis project. At this moment, I need to filter my database with several stopwords. The tm package has the function to do this, but I have to turn a list or data frame of words into a corpus object., and I don´t want to do this."

stopwords <- c("I", "to", "the", "a", "with", "object", "do", "or", "of", "my", "this", "in", "into")

# "\<" is another escape sequence for the beginning of a word, and "\>" is the end

stopwords <- paste0("\\<", stopwords, "\\>")

# | means or

gsub(paste(stopwords, collapse = "|"), "", x, ignore.case = TRUE)

###############################################################
###															###
###															###
###   						Forecasting						###
###															###
###															###
###############################################################

####################### some notes on forecast ##################

data.train <- as.ts(data.train$Units)

tsdisplay(data.train) 
tsdisplay(diff(data.train,2))

f.model <- Arima(data.train, order = c(2,1,0))
f.cast <- forecast(f.model, h = 100)
f.acc <- accuracy(f.cast, data.test$Units)

f.model.auto <- auto.arima(data.train, stepwise = F, approximation = F)
f.cast.auto <- forecast(f.model.auto, h = 100)
f.acc.auto <- accuracy(f.cast.auto, data.test$Units)

###############################################################
###															###
###															###
###   					Widening the data					###
###															###
###															###
###############################################################


####################### some notes on widening the data ##################

library(reshape2)

text.data <- read.csv("challenge.csv", stringsAsFactors = F)
text.data$Category <- as.factor(text.data$Category)
text.data <- data.frame(text.data, value=TRUE)

wide.data <- reshape(text.data, idvar=c("Text"), timevar="Category", direction="wide")


#################### caravan data classification using various techniques ####################

### clean up and separate test and train
data <- Caravan
set.seed(123)
train= sample(c(TRUE ,FALSE), nrow(data),rep=TRUE)
train= sample(c(TRUE ,FALSE), nrow(data),rep=TRUE)
test = (!train)
data$Purchase1 <- ifelse(data$Purchase == "Yes", 1, 0)

rem <- c("Purchase")
data <- data[,!(names(data) %in% rem)]

data.test <- data[test,]
data.train <- data[train,]

# Number of folds
tr.control = trainControl(method = "cv", number = 10)

### first ... decision tree

cp.grid = expand.grid(cp = (0:10)*0.001) ## set grod for cp values
tree.tr = train(Purchase1 ~.,data = data.train,method = "rpart",trControl = tr.control, tuneGrid = cp.grid)

best.tree = tree.tr$finalModel
best.tree$tuneValue #returns selected cp value
prp(best.tree)

best.tree.pred = predict(best.tree, newdata=data.test)
tree.predROCR = prediction(best.tree.pred, data.test$Purchase1)
tree.perfROCR = performance(tree.predROCR, "tpr", "fpr")
plot(tree.perfROCR, colorize=TRUE)
performance(tree.predROCR, "auc")@y.values ####0.6996917

### second ... boost

shrinkage = (1:10)*0.001
n.trees =1000
interaction.depth = 1
boost.grid = expand.grid(shrinkage = shrinkage, n.trees = n.trees, interaction.depth=interaction.depth)

boost.tr = train(Purchase1 ~.,data = data.train,method = "gbm",trControl = tr.control, tuneGrid = boost.grid)

best.boost = boost.tr$finalModel

best.boost.pred = predict(best.boost, newdata=data.test, n.trees = 1000, type="response")
boost.predROCR = prediction(best.boost.pred, data.test$Purchase1)
boost.perfROCR = performance(boost.predROCR, "tpr", "fpr")
plot(boost.perfROCR, colorize=TRUE)
performance(boost.predROCR, "auc")@y.values ####0.7542671

### third ... naivebayes
data <- Caravan
set.seed(123)
train= sample(c(TRUE ,FALSE), nrow(data),rep=TRUE)
train= sample(c(TRUE ,FALSE), nrow(data),rep=TRUE)
test = (!train)

data.test <- data[test,]
data.train <- data[train,]

library(e1071)
nb = naiveBayes(Purchase ~., data = data.train)
nb.pred <- predict(nb,data.test,type = c("class"))

table(nb.pred,data.test$Purchase)

nb.prob <- predict(nb,data.test,type = c("raw"))

#################