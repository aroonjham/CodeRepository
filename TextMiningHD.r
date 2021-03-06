# set working directory
setwd ("C:/Users/Aroon/Downloads/HomeDepotTest")

# import data
data1 <- read.csv("train.csv")
data2 <- read.csv("product_descriptions.csv")

# merge datasets
dataInt <- merge(data1, data2, by.x = "product_uid", by.y = "product_uid")

# remove unnecessary data from memory
rm(data1)
rm(data2)

# create corpusProdDesc from product description
library(tm)
library(proxy)
library(RWeka) #needed for n-gram tokenizing

# the following are the steps we use in building the solution. We take this approach to overcome memory restriction of R
# make individual vectors of search term (st), product description (pd) and product title (pt)
# combine to a single vector
# create a corpus
# conduct corpus cleansing
# make a dtm with tdidf
# convert to matrix
# rename matrix rows
# run dist command
# convert dist to matrix
# extract the relevant items

#initiate some blank vectors and a dataframe
cp_st <- as.vector(0)
cp_pd <- as.vector(0)
cp_pt <- as.vector(0)
df_distances <- data.frame(st_pd_euc = double(),
							st_pt_euc = double(),
							st_pd_man = double(),
							st_pt_man = double(),
							st_pd_cos = double(),
							st_pt_cos = double())

i = nrow(dataInt)

# here's the loop. It will take a while to run.
for (count in 1:i) {

cp_st <- as.vector(dataInt$search_term[count])
cp_pd <- as.vector(dataInt$product_description[count])
cp_pt <- as.vector(dataInt$product_title[count])

cp <- c(cp_st, cp_pd, cp_pt)

cp <- Corpus(VectorSource(cp))
cp <- tm_map(cp, tolower)
cp <- tm_map(cp, removePunctuation)
cp <- tm_map(cp, removeWords, c(stopwords('english')))
cp <- tm_map(cp, stripWhitespace)
#cp <- tm_map(cp, stemDocument)
cp <- tm_map(cp, PlainTextDocument)
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))
dtm <- as.matrix(DocumentTermMatrix(cp, control = list(tokenize = BigramTokenizer, weighting = function(x) weightTfIdf(x, normalize = T))))
rownames(dtm) <- c("st","pd","pt")

euc_dist <- as.matrix(dist(dtm))

st_pd_euc <- euc_dist[2,1]
st_pt_euc <- euc_dist[3,1]

man_dist <- as.matrix(dist(dtm, "manhattan"))

st_pd_man <- man_dist[2,1]
st_pt_man <- man_dist[3,1]

cos_dist <- as.matrix(dist(dtm, "cosine"))

st_pd_cos <- euc_dist[2,1]
st_pt_cos <- euc_dist[3,1]

df_row <- as.vector(c(st_pd_euc,st_pt_euc,st_pd_man,st_pt_man,st_pd_cos,st_pt_cos))

df_distances <- rbind(df_distances,df_row)
}

colnames(df_distances) <- c("st_pd_euc","st_pt_euc","st_pd_man", "st_pt_man","st_pd_cos","st_pt_cos")
df_distances$relevance <- dataInt$relevance

write.csv(df_distances, file = "df_distances.csv")
#df_distances <- read.csv("df_distances.csv")
df_distances$relevance <- as.factor(df_distances$relevance)

#separate test and train
set.seed(123)
train= sample(c(TRUE,TRUE,TRUE,FALSE), nrow(df_distances),rep=TRUE)
test = (!train)

df.train <- as.data.frame(df_distances[train,])
df.test <- as.data.frame(df_distances[test,])

library(nnet)

mod <- multinom(relevance ~ ., df.train)

# we will use a fancy prediction function http://www.r-bloggers.com/how-to-multinomial-regression-models-in-r/

predictMNL <- function(model, newdata) {
    
    # Only works for neural network models
    if (is.element("nnet",class(model))) {
        # Calculate the individual and cumulative probabilities
        probs <- predict(model,newdata,"probs")
        cum.probs <- t(apply(probs,1,cumsum))
        
        # Draw random values
        vals <- runif(nrow(newdata))
        
        # Join cumulative probabilities and random draws
        tmp <- cbind(cum.probs,vals)
        
        # For each row, get choice index.
        k <- ncol(probs)
        ids <- 1 + apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
        
        # Return the values
        return(ids)
    }
}

y2 <- predictMNL(mod,df.test)

df2 <- cbind(df.test,y=y2)


####################################

convert <- function(x){
switch(x,
"1",
"1.25",
"1.33",
"1.5",
"1.67",
"1.75",
"2",
"2.25",
"2.33",
"2.5",
"2.67",
"2.75",
"3")
}


