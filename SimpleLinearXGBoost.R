library(Metrics)
library(data.table)
library(SnowballC)
library(xgboost)


set.seed(123)

cat("Reading data\n")
train <- fread('input/train.csv')
test <- fread('input/test.csv')
desc <- fread('input/product_descriptions.csv')

cat("Merge description with train and test data \n")
train <- merge(train,desc, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)
test <- merge(test,desc, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)


word_match <- function(words,title,desc,type=1){
  n_title <- 0
  n_desc <- 0
  words <- unlist(strsplit(words," "))
  nwords <- length(words)
  for(i in 1:length(words)){
    pattern <- ifelse(type==1,paste("(^| )",words[i],"($| )",sep=""),
                ifelse(type==2,words[i],
                             paste("[0-","9]",sep="") ) )
  
    n_title <- n_title + ifelse(type==2,grepl(pattern,title,ignore.case=TRUE),grepl(pattern,title,perl=TRUE,ignore.case=TRUE))
    n_desc <- n_desc + ifelse(type==2,grepl(pattern,desc,ignore.case=TRUE),grepl(pattern,desc,perl=TRUE,ignore.case=TRUE))
  }
  return(c(n_title,nwords,n_desc))
}
word_stem <- function(words){
  
  i <- 1
  words <- unlist(strsplit(words," "))
  nwords <- length(words)
  pattern <- wordStem(words[i], language = "porter")
  for(i in 2:length(words)){
    pattern <- paste(pattern,wordStem(words[i], language = "porter"),sep=" ")
  }
  return(pattern)
}



cat("Get number of words and word matching title in train\n")
train_words <- as.data.frame(t(mapply(word_match,train$search_term,train$product_title,train$product_description)))
train$nmatch_title <- train_words[,1]
train$nwords <- train_words[,2]
train$nmatch_desc <- train_words[,3]


cat("Get number of words and word matching title in test\n")
test_words <- as.data.frame(t(mapply(word_match,test$search_term,test$product_title,test$product_description)))
test$nmatch_title <- test_words[,1]
test$nwords <- test_words[,2]
test$nmatch_desc <- test_words[,3]

rm(train_words,test_words)


cat("Get number of words and word matching title in train with porter stem\n")
train$search_term2 <- sapply(train$search_term,word_stem)
train_words <- as.data.frame(t(mapply(word_match,train$search_term2,train$product_title,train$product_description,2)))
train$nmatch_title2 <- train_words[,1]
train$nmatch_desc2 <- train_words[,3]



cat("Get number of words and word matching title in test with porter stem\n")
test$search_term2 <- sapply(test$search_term,word_stem)
test_words <- as.data.frame(t(mapply(word_match,test$search_term2,test$product_title,test$product_description,2)))
test$nmatch_title2 <- test_words[,1]
test$nmatch_desc2 <- test_words[,3]



rm(train_words,test_words)


cat("A simple linear model on number of words and number of words that match\n")

#Define sample for validation dataset
h<-sample(nrow(train),5000)
#Define X
X <- c("nmatch_title","nwords","nmatch_desc", "nmatch_title2","nmatch_desc2")
#Define y
y <- "relevance"


dval<-xgb.DMatrix(data=data.matrix(train[h,X,with=F]),label=as.matrix(train[h,y,with=F]))
dtrain<-xgb.DMatrix(data=data.matrix(train[-h,X,with=F]),label=as.matrix(train[-h,y,with=F]))
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster             = "gbtree",
                eta                 = 0.025, 
                max_depth           = 6, 
                subsample           = 0.7, 
                colsample_bytree    = 0.9, 
                eval_metric         = "rmse",
                min_child_weight    = 6
)



clf <- xgb.train(data = dtrain, 
                 params                = param, 
                 nrounds               = 500,
                 verbose               = 1,
                 watchlist             = watchlist,
                 early_stopping_rounds = 50, 
                 print_every_n         = 1
)
clf$best_score

# Impo
var_imp <- xgb.importance(X,model=clf)

xgb.plot.importance(var_imp)

train$pred_relevance <- predict(clf,data.matrix(train[,X,with=F]))


cat("Submit file\n")
test_relevance <- predict(clf,data.matrix(test[,X,with=F]),ntreelimit =clf$best_iteration)
summary(test_relevance)
test_relevance <- ifelse(test_relevance>3,3,test_relevance)
test_relevance <- ifelse(test_relevance<1,1,test_relevance)

submission <- data.frame(id=test$id,relevance=test_relevance)

### Outputs
write.csv(train,"train_final.csv")

# write_csv(submission,"xgb_submission.csv")





