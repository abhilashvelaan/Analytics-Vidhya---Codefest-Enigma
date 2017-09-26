library(tidyverse)
library(lubridate)
library(caret)
library(h2o)

#reading csv files
train_submission = read.csv(file.choose(), header = T,stringsAsFactors = T,
                            na.strings = c("",NA))
user_data = read.csv(file.choose(), header = T, stringsAsFactors = T,
                     na.strings = c("",NA))
problem_data = read.csv(file.choose(), header = T, stringsAsFactors = T,
                        na.strings = c("",NA))

test_submission = read.csv(file.choose(), header = T, stringsAsFactors = T,
                           na.strings = c("",NA))
#summary statistics 
summary(train_submission)
summary(user_data)
summary(problem_data)

#NAs (Missing values)
colSums(is.na(train_submission)) #no NAs
colSums(is.na(test_submission)) #no NAs
colSums(is.na(problem_data)) # points => 3971 out of 6544 obs,
                             # level_types => 133 out of 6544 obs
                             # tags => 3484 out of 6544 obs
colSums(is.na(user_data)) # country => 1153 out of 3571 obs

#joining the user_data and problem_data with train and test data
model_train = left_join(train_submission,problem_data,by = "problem_id")
model_train = left_join(model_train,user_data,by = "user_id")

model_test = left_join(test_submission,problem_data,by = "problem_id")
model_test = left_join(model_test,user_data,by = "user_id")

summary(model_train)
summary(model_test)

#creating the variable ID
model_train$ID = as.factor(paste(model_train$user_id,"_",model_train$problem_id,sep = ""))

#calculating acceptance rate using problem_solved and submission count
model_train$acceptance = model_train$problem_solved/model_train$submission_count
model_test$acceptance = model_test$problem_solved/model_test$submission_count

#creating a group based on rating and max rating
model_train$rating_group = as.factor(ifelse(model_train$max_rating == model_train$rating, "same","decrease"))
model_test$rating_group = as.factor(ifelse(model_test$max_rating == model_test$rating, "same","decrease"))

#calculating total active days from last_online_time_seconds and registration_time_seconds
class(model_train$last_online_time_seconds) = c("POSIXt","POSIXct")
model_train$last_online_time_seconds = as.POSIXlt(model_train$last_online_time_seconds)
class(model_test$last_online_time_seconds) = c("POSIXt","POSIXct")
model_test$last_online_time_seconds = as.POSIXlt(model_test$last_online_time_seconds)


class(model_train$registration_time_seconds) = c("POSIXt", "POSIXct")
model_train$registration_time_seconds = as.POSIXlt(model_train$registration_time_seconds)
class(model_test$registration_time_seconds) = c("POSIXt", "POSIXct")
model_test$registration_time_seconds = as.POSIXlt(model_test$registration_time_seconds)

model_train$approx_active_days = round(as.numeric(model_train$last_online_time_seconds - 
                                                    model_train$registration_time_seconds))
model_test$approx_active_days = round(as.numeric(model_test$last_online_time_seconds - 
                                                   model_test$registration_time_seconds))

#selecting and modifying data for Random_forest
train_1 = model_train %>% 
  select(level_type,points,tags,submission_count,problem_solved,contribution,country,follower_count,
         max_rating,rating,rank,acceptance,rating_group,approx_active_days,attempts_range)

test_1 = model_test %>% 
  select(level_type,points,tags,submission_count,problem_solved,contribution,country,follower_count,
         max_rating,rating,rank,acceptance,rating_group,approx_active_days)

train_1$attempts_range = as.factor(train_1$attempts_range)

localh2o = h2o.init(nthreads = 3)
h2o.init()

#data to h2o cluster 
train.h2o = as.h2o(train_1)
test.h20 = as.h2o(test_1)

splits = h2o.splitFrame(train.h2o, 0.7, seed = 4321)

train_h2o = h2o.assign(splits[[1]], "train.hex")
valid_h2o = h2o.assign(splits[[2]], "valid.hex")
test_h2o = h2o.assign(test.h20, "test.hex")

#check col index
colnames(train_h2o)

y.dep = 15
x.indep = c(1:14)

rf2 = h2o.randomForest(training_frame = train_h2o,
                       validation_frame = valid_h2o,
                       x = x.indep, y = y.dep,
                       model_id = "rf_v2",
                       ntrees = 500,
                       max_depth = 30,
                       stopping_rounds = 2,
                       stopping_tolerance = 1e-2,
                       score_each_iteration = T,
                       seed = 4321 )
summary(rf2)
h2o.performance(rf2)
rf2@model$validation_metrics

predict_rf2 = as.data.frame(h2o.predict(rf2, test_h2o))
sub_rf2 = data.frame(ID = test_submission$ID, attempts_range = predict_rf2$predict)
write.csv(sub_rf2, file = "E:\\EXTRA\\AV\\CODEFEST-ENIGMA -2017\\SUBMISSIONS\\submission_h2o_rf2_4.csv",row.names = F)