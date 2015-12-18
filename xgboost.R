setwd("~/Desktop/E/kaggle data/Rossman_store")

library(readr)
library(magrittr)
library(data.table)
library(zoo)
library(forecast)
library(xgboost)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(gbm)
library(dplyr)
#library()

train=read_csv("train.csv", col_types=list(
  Store = col_integer(),
  DayOfWeek= col_integer(),
  Date = col_date(),
  Sales = col_integer(),
  Customers = col_integer(),
  Open = col_integer(),
  Promo = col_integer(),
  StateHoliday = col_character(),
  SchoolHoliday = col_integer()))

test=read.csv("test.csv")
store=read.csv("store.csv")


#train$Date=NULL
#test$Date=NULL

train=merge(train,store, by="Store")
test=merge(test,store, by="Store")


test$Open[is.na(test$Open)] = 1

train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

train_all=train

train <- train[ -which(train$Open=='1' & train$Sales==0),]
train <- train[ which(train$Sales!='0'),]
train = train[which(train$Open==1),]

train$Date=as.Date(train$Date)
test$Date=as.Date(test$Date)

train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))
train$day <- as.integer(format(train$Date, "%d"))

test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$day <- as.integer(format(test$Date, "%d"))

train <- train[,-c(3,8)]

test <- test[,-c(4,7)]
###########exploratory data analysis#####################

table(train_all$Open, train_all$DayOfWeek)

unique(train_all$Store[train_all$Open==0 & train_all$DayOfWeek==7])

unique(train_all$Store[train_all$Open==0 & train_all$DayOfWeek==7 & train_all$Sales>0])



d=rpart(Sales~factor(DayOfWeek), data=train)
fancyRpartPlot(d)

train$DayOfWeek_bin=0
train$DayOfWeek_bin[train$DayOfWeek==6]=1
train$DayOfWeek_bin[train$DayOfWeek==2 | train$DayOfWeek==3 | train$DayOfWeek==4 | train$DayOfWeek==5]=2
train$DayOfWeek_bin[train$DayOfWeek==1 | train$DayOfWeek==7]=3

test$DayOfWeek_bin=0
test$DayOfWeek_bin[test$DayOfWeek==6]=1
test$DayOfWeek_bin[test$DayOfWeek==2 | test$DayOfWeek==3 | test$DayOfWeek==4 | test$DayOfWeek==5]=2
test$DayOfWeek_bin[test$DayOfWeek==1 | test$DayOfWeek==7]=3

################################


feature.names <- names(train)[c(1,2,6,8:12,14:18)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}
nrow(train)
h<-sample(nrow(train),100000)

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log1p(train$Sales)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log1p(train$Sales)[-h])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.25, # 0.06, #0.01,
                max_depth           = 15, #changed from default of 8
                subsample           = 0.7, # 0.7
                colsample_bytree    = 0.7, # 0.7
                min_child_weight    = 0.2
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 800, #300, #280, #125, #250, # changed from 300
                    verbose             = 1,
                    early.stop.round    = 30,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)

#test1=test[test$Open==1,]
#test2=test[test$Open==0,]
pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1

submission <- data.frame(ID=test$Id, Sales=pred1)
mean(submission$Sales)

write.csv(submission, "xg_ensem5.csv", row.names = F)

#############ensemble of csv's######################
c1=read.csv("xg1.csv")
c2=read.csv("xg2.csv")
c3=read.csv("xg3.csv")
c4=read.csv("xg4.csv")
c6=read.csv("xg6.csv")
c7=read.csv("xg7.csv")
c9=read.csv("xg9.csv")
c10=read.csv("xg10.csv")
c11=read.csv("xg11.csv")
c15=read.csv("xg15.csv")


pred1=(c1$Sales+c2$Sales+c3$Sales+c4$Sales+c6$Sales+c7$Sales+c9$Sales+c10$Sales+c11$Sales+c15$Sales)/10

submission <- data.frame(ID=test$Id, Sales=pred1)
mean(submission$Sales)

write.csv(submission, "xg_ensem5.csv", row.names = F)