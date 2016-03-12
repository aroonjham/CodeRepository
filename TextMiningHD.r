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
library(RWeka)


# solution is a loop
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
