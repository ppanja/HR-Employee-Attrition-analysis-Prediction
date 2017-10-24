################## Library Load ###########################

library(readr)
library(partykit)
library(rpart)
library(caret)
library(ggplot2)
library(scales)
library(randomForest)
library(pacman)
library(ROCR)
library(data.table)
library(ineq)
library(neuralnet)
library(e1071)

################## deciling code function ###########################
decile <- function(x){
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i, na.rm=T)
  }
  return (
  ifelse(x<deciles[1], 1,
  ifelse(x<deciles[2], 2,
  ifelse(x<deciles[3], 3,
  ifelse(x<deciles[4], 4,
  ifelse(x<deciles[5], 5,
  ifelse(x<deciles[6], 6,
  ifelse(x<deciles[7], 7,
  ifelse(x<deciles[8], 8,
  ifelse(x<deciles[9], 9, 10
  ))))))))))
}

################## Outlier Analysis - Varaiable ###########################

outlier_upper=function(x){
  q = quantile(x)
  names(q) = NULL
  q1 = q[2]
  q3 = q[4]
  QR = q3-q1
  return(q3+1.5*QR);
}

outlier_lower=function(x){
  q = quantile(x)
  names(q) = NULL
  q1 = q[2]
  q3 = q[4]
  QR = q3-q1
  return(q1-1.5*QR);
}

################## DATA SUMMARY ###########################
HR_Attrn_Data <- read_csv("C:/Users/Prasanta/Downloads/PGPBABI/DataMining/HR_Employee_Attrition_Data.csv")
dim(HR_Attrn_Data)
sapply(HR_Attrn_Data,class)
summary(HR_Attrn_Data)

################## Exploratory Data Analysis ###########################
> ggplot(HR_Attrn_Data, aes(x = WorkLifeBalance, fill = Attrition)) +
+     stat_count(width = 0.5) +
+     xlab("Work Life Balance") + 
+     ylab("Count") +
+     labs(fill = "Attrition")

> ggplot(HR_Attrn_Data, aes(x = YearsInCurrentRole, fill = Attrition)) +
+     stat_count(width = 0.5) +
+     xlab("YearsInCurrentRole") + 
+     ylab("Count") +
+     labs(fill = "Attrition")

> ggplot(HR_Attrn_Data, aes(x = JobSatisfaction, fill = Attrition)) +
+     stat_count(width = 0.5) +
+     xlab("Job Satisfaction") + 
+     ylab("Count") +
+     labs(fill = "Attrition")

d = HR_Attrn_Data[,-c(10,22,27)]

par(mfrow=c(2,5))

boxplot( HR_Attrn_Data$MonthlyIncome, main = "Monthly Income", col = "blue")
boxplot( HR_Attrn_Data$NumCompaniesWorked, main = "Comp Worked", col = "blue")
boxplot( HR_Attrn_Data$PercentSalaryHike, main = "% Sal hike", col = "blue")
boxplot( HR_Attrn_Data$StockOptionLevel, main = "Stock Opt level", col = "blue")
boxplot( HR_Attrn_Data$TotalWorkingYears, main = "Total Wk yrs", col = "blue")
boxplot( HR_Attrn_Data$TrainingTimesLastYear, main = "Training count", col = "blue")
boxplot( HR_Attrn_Data$YearsAtCompany, main = "Yrs at Comp", col = "blue")
boxplot( HR_Attrn_Data$YearsInCurrentRole, main = "Yrs in Curr Role", col = "blue")
boxplot( HR_Attrn_Data$YearsSinceLastPromotion, main = "Yrs frm last prom", col = "blue")
boxplot( HR_Attrn_Data$YearsWithCurrManager, main = "Yrs with Curr Mgr", col = "blue")

################## Convert some categorical variables to factors ###########################
HR_Attrn_Data$BusinessTravel = as.factor(HR_Attrn_Data$BusinessTravel)
HR_Attrn_Data$Attrition = as.factor(HR_Attrn_Data$Attrition)
HR_Attrn_Data$Department = as.factor(HR_Attrn_Data$Department)
HR_Attrn_Data$EducationField = as.factor(HR_Attrn_Data$EducationField)
HR_Attrn_Data$Gender = as.factor(HR_Attrn_Data$Gender)
HR_Attrn_Data$JobRole = as.factor(HR_Attrn_Data$JobRole)
HR_Attrn_Data$MaritalStatus = as.factor(HR_Attrn_Data$MaritalStatus)
HR_Attrn_Data$Over18 = as.factor(HR_Attrn_Data$Over18)
HR_Attrn_Data$OverTime = as.factor(HR_Attrn_Data$OverTime)

## Hypothesis: Attrition seems to be more when opportunity to travel is rare
 ggplot(HR_Attrn_Data, aes(x = BusinessTravel, fill = Attrition)) +
     stat_count(width = 0.5) +
     xlab("Business Travel") + 
     ylab("Count") +
     labs(fill = "Attrition")
#--YES

## Hypothesis: Some JobRole have high attrition rates
ggplot(HR_Attrn_Data, aes(x = JobRole, fill = Attrition)) +
     stat_count(width = 0.5) +
     xlab("Job Role") + 
     ylab("Count") +
     labs(fill = "Attrition")
#--YES
 
## Hypothesis: Male has higher attrition rates compared to Female
ggplot(HR_Attrn_Data, aes(x = Gender, fill = Attrition)) +
     stat_count(width = 0.5) +
     xlab("Gender") + 
     ylab("Count") +
     labs(fill = "Attrition")
#--YES

## Hypothesis: Married people have less attrition rates compared to single
 ggplot(HR_Attrn_Data, aes(x = MaritalStatus, fill = Attrition)) +
     stat_count(width = 0.5) +
     xlab("Marital Status") + 
     ylab("Count") +
     labs(fill = "Attrition")
#-- YES 
ggplot(train, aes(x = JobLevel)) + geom_bar(aes(fill = Attrition), position = 'fill')
  

################## remove outlier data from the data set ###########################
d = subset (d, d$NumCompaniesWorked> outlier_lower(d$NumCompaniesWorked) & d$NumCompaniesWorked < outlier_upper(d$NumCompaniesWorked))
d = subset (d, d$StockOptionLevel> outlier_lower(d$StockOptionLevel) & d$StockOptionLevel < outlier_upper(d$StockOptionLevel))
d = subset (d, d$TotalWorkingYears> outlier_lower(d$TotalWorkingYears) & d$TotalWorkingYears < outlier_upper(d$TotalWorkingYears))
d = subset (d, d$TrainingTimesLastYear> outlier_lower(d$TrainingTimesLastYear) & d$TrainingTimesLastYear < outlier_upper(d$TrainingTimesLastYear))
d = subset (d, d$YearsAtCompany> outlier_lower(d$YearsAtCompany) & d$YearsAtCompany < outlier_upper(d$YearsAtCompany))
d = subset (d, d$YearsInCurrentRole> outlier_lower(d$YearsInCurrentRole) & d$YearsInCurrentRole < outlier_upper(d$YearsInCurrentRole))
d = subset (d, d$YearsSinceLastPromotion> outlier_lower(d$YearsSinceLastPromotion) & d$YearsSinceLastPromotion < outlier_upper(d$YearsSinceLastPromotion))
d = subset (d, d$YearsWithCurrManager> outlier_lower(d$YearsWithCurrManager) & d$YearsWithCurrManager < outlier_upper(d$YearsWithCurrManager))

################## Training and Test data creation ###########################
hrtrain<-head(HR_Attrn_Data,round(nrow(HR_Attrn_Data)*0.7))
dim(hrtrain)
hrtest<-tail(HR_Attrn_Data,round(nrow(HR_Attrn_Data)*0.3))
dim(hrtest)
rf_hrtrain <- hrtrain
rf_hrtest <- hrtest

n_hrtrain <- hrtrain
n_hrtest <- hrtest

################## Build CART model using hrtrain data ###########################
cart.model = rpart(Attrition ~., data = hrtrain, method = "class", control = rpart.control(minsplit = 60, minbucket = 30, depth=6))
plot(as.party(cart.model))
print(cart.model)

summary(cart.model)

hrtrain$predict.class <- predict(cart.model, hrtrain, type="class")
hrtrain$predict.score <- predict(cart.model, hrtrain, type="prob")
View(hrtrain)
head(hrtrain)

## deciling
hrtrain$deciles <- decile(hrtrain$predict.score[,2])

## Ranking code
tmp_DT <- data.table(hrtrain)
rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_yes = sum(Attrition == "Yes"), 
  cnt_no = sum(Attrition == "No")) , 
  by=deciles][order(-deciles)]
rank$rrate <- round(rank$cnt_yes * 100 / rank$cnt,2);
rank$cum_yes <- cumsum(rank$cnt_yes)
rank$cum_no <- cumsum(rank$cnt_no)
rank$cum_rel_yes <- round(rank$cum_yes / sum(rank$cnt_yes),2);
rank$cum_rel_no <- round(rank$cum_no / sum(rank$cnt_no),2);
rank$ks <- abs(rank$cum_yes - rank$cum_rel_no);
View(rank)

pred <- prediction(hrtrain$predict.score[,2], hrtrain$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
gini = ineq(hrtrain$predict.score[,2], type="Gini")
with(hrtrain, table(Attrition, predict.class))
auc
KS
gini

ptree<- prune(cart.model, cp=cart.model$cptable[which.min(cart.model$cptable[,"xerror"]),"CP"])

## Scoring another dataset using the CART Model Object

hrtest$predict.class <- predict(ptree, hrtest, type="class")
hrtest$predict.score <- predict(ptree, hrtest, type="prob")
View(hrtest)
head(hrtest)

## deciling
hrtest$deciles <- decile(hrtest$predict.score[,2])

## Ranking code
tmp_DT <- data.table(hrtest)
rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_yes = sum(Attrition == "Yes"), 
  cnt_no = sum(Attrition == "No")) , 
  by=deciles][order(-deciles)]
rank$rrate <- round(rank$cnt_yes * 100 / rank$cnt,2);
rank$cum_yes <- cumsum(rank$cnt_yes)
rank$cum_no <- cumsum(rank$cnt_no)
rank$cum_rel_yes <- round(rank$cum_yes / sum(rank$cnt_yes),2);
rank$cum_rel_no <- round(rank$cum_no / sum(rank$cnt_no),2);
rank$ks <- abs(rank$cum_yes - rank$cum_rel_no);
View(rank)

with(hrtest, table(Attrition, predict.class))


################## Build the random forest using rf_hrtrain data ###########################

rf = randomForest(Attrition ~ Age + BusinessTravel + DailyRate + Department + DistanceFromHome + Education + EducationField + EnvironmentSatisfaction + Gender + HourlyRate + JobInvolvement + JobLevel + JobRole + JobSatisfaction + MaritalStatus + MonthlyIncome + MonthlyRate + NumCompaniesWorked + Over18 + OverTime + PercentSalaryHike + PerformanceRating + RelationshipSatisfaction + StandardHours + StockOptionLevel + TotalWorkingYears + TrainingTimesLastYear + WorkLifeBalance + YearsAtCompany + YearsInCurrentRole + YearsSinceLastPromotion + YearsWithCurrManager, data=hrtrain, ntree=501, mtry=3, nodesize = 20, importance=TRUE)

## Check the error rate
plot(rf, main="")
legend("topright", c("OOB", "No", "Yes"), text.col=1:6, lty=1:3, col=1:3)
title(main="Error Rates Random Forest")

### Tune the random forest
tRF <- tuneRF(x = rf_hrtrain[,-2], 
               y=rf_hrtrain$Attrition,
               mtryStart = 3, 
               ntreeTry=20, 
               stepFactor = 1.5, 
               improve = 0.001, 
               trace=TRUE, 
               plot = TRUE,
               doBest = TRUE,
               nodesize = 20, 
               importance=TRUE
)


rf_hrtrain$predict.class = predict(tRF, rf_hrtrain, type="class")
rf_hrtrain$predict.score = predict(tRF, rf_hrtrain, type="prob")
head(rf_hrtrain)

## deciling
rf_hrtrain$deciles = decile(rf_hrtrain$predict.score[,2])


## Ranking code
tmp_DT = data.table(rf_hrtrain)
rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_yes = sum(Attrition == "Yes"), 
  cnt_no = sum(Attrition == "No")) , 
  by=deciles][order(-deciles)]
rank$rrate <- round(rank$cnt_yes * 100 / rank$cnt,2);
rank$cum_yes <- cumsum(rank$cnt_yes)
rank$cum_no <- cumsum(rank$cnt_no)
rank$cum_rel_yes <- round(rank$cum_yes / sum(rank$cnt_yes),2);
rank$cum_rel_no <- round(rank$cum_no / sum(rank$cnt_no),2);
rank$ks <- abs(rank$cum_yes - rank$cum_rel_no);
View(rank)

pred = prediction(rf_hrtrain$predict.score[,2], rf_hrtrain$Attrition)
perf = performance(pred, "tpr", "fpr")
plot(perf)
KS = max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc = performance(pred,"auc"); 
auc = as.numeric(auc@y.values)

library(ineq)
gini = ineq(rf_hrtrain$predict.score[,2], type="Gini")

with(rf_hrtrain, table(Attrition, predict.class))
auc
KS
gini


## Scoring another dataset using the Random Forest Model Object
rf_hrtest$predict.class <- predict(tRF, rf_hrtest, type="class")
rf_hrtest$predict.score <- predict(tRF, rf_hrtest, type="prob")
with(rf_hrtest, table(Attrition, predict.class))

rf_hrtest$deciles <- decile(rf_hrtest$predict.score[,2])
tmp_DT = data.table(rf_hrtest)
h_rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_yes = sum(Attrition == "Yes"), 
  cnt_no = sum(Attrition == "No")) , 
  by=deciles][order(-deciles)]
h_rank$rrate <- round(h_rank$cnt_yes * 100 / h_rank$cnt,2);
h_rank$cum_yes <- cumsum(h_rank$cnt_yes)
h_rank$cum_no <- cumsum(h_rank$cnt_no)
h_rank$cum_rel_yes <- round(h_rank$cum_yes / sum(h_rank$cnt_yes),2);
h_rank$cum_rel_no <- round(h_rank$cum_no / sum(h_rank$cnt_no),2);
h_rank$ks <- abs(h_rank$cum_yes - h_rank$cum_rel_no);
View(h_rank)



################## Build the Neural Nets model using n_hrtrain data ###########################


n_hrtrain <- head(HR_Attrn_Data,round(nrow(HR_Attrn_Data)*0.7))
dim(n_hrtrain)
[1] 2058   35
n_hrtrain$Attrition <- as.numeric(n_hrtrain$Attrition)-1
View(n_hrtrain)

m1 <- model.matrix(~ Attrition+Age+BusinessTravel+DailyRate+Department+DistanceFromHome+EducationField+EnvironmentSatisfaction+MonthlyIncome+NumCompaniesWorked+OverTime+TotalWorkingYears+TrainingTimesLastYear+WorkLifeBalance+YearsInCurrentRole+YearsSinceLastPromotion+YearsWithCurrManager,data=n_hrtrain)
View(m1)
colnames(m1)[7] <- 'DepartmentResearch_Development'
colnames(m1)[10] <- 'EducationFieldLife_Sciences'
colnames(m1)[14] <- 'EducationFieldTechnical_Degree'
View(m1)
m1scaled <- scale(m1)
m1scaled <- scale(m1[,-2])
View(m1scaled)
m1scaled <- cbind(n_hrtrain[2],m1scaled)
View(m1scaled)
mod.net <- neuralnet(Attrition ~Age+BusinessTravelTravel_Frequently+BusinessTravelTravel_Rarely+DailyRate+DepartmentResearch_Development+DepartmentSales+DistanceFromHome+EducationFieldLife_Sciences+EducationFieldMarketing+EducationFieldMedical+EducationFieldOther+EducationFieldTechnical_Degree +EnvironmentSatisfaction+MonthlyIncome+NumCompaniesWorked+OverTimeYes+TotalWorkingYears+TrainingTimesLastYear+WorkLifeBalance+YearsInCurrentRole+YearsSinceLastPromotion+YearsWithCurrManager, data=m1scaled[,-2],hidden = 4,err.fct = "sse",lifesign = 'minimal',linear.output = F,threshold = .1)
##hidden: 4    thresh: 0.1    rep: 1/1    steps:    4068	error: 83.20291	time: 4.27 secs
plot(mod.net)
n_hrtrain$Prob = mod.net$net.result[[1]]
quantile(n_hrtrain$Prob, c(0,1,5,10,25,50,75,90,95,99,100)/100))
Error: unexpected ')' in "quantile(n_hrtrain$Prob, c(0,1,5,10,25,50,75,90,95,99,100)/100))"
quantile(n_hrtrain$Prob, c(0,1,5,10,25,50,75,90,95,99,100)/100)
##            0%             1%             5%            10%            25%            50%            75% 
##-0.21399557987 -0.17276166405 -0.10087681622  0.03747340901  0.03897463727  0.08447675364  0.16631056428 
           90%            95%            99%           100% 
## 0.46873861708  0.73822948459  1.05550828660  1.26850862040 
hist(n_hrtrain$Prob)

n_hrtrain$deciles <- decile(n_hrtrain$Prob)
library(data.table)
tmp_DT = data.table(n_hrtrain)
rank <- tmp_DT[, list(
     cnt = length(Attrition), 
     cnt_resp = sum(Attrition), 
     cnt_non_resp = sum(Attrition == 0)) , 
     by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);
library(scales)
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)
## Assgining 0 / 1 class based on certain threshold
n_hrtrain$Class = ifelse(n_hrtrain$Prob>0.21,1,0)
with( n_hrtrain, table(Attrition, as.factor(Class)  ))
## We can use the confusionMatrix function of the caret package 
##install.packages("caret","e1071")
library(caret)
confusionMatrix(n_hrtrain$Attrition, n_hrtrain$Class)

## Scoring another dataset using the Neural Net Model Object

n_hrtest <- head(HR_Attrn_Data,round(nrow(HR_Attrn_Data)*0.7))
dim(n_hrtest)
##[1] 2058   35
n_hrtest$Attrition <- as.numeric(n_hrtest$Attrition)-1
View(n_hrtest)

m2 <- model.matrix(~ Attrition+Age+BusinessTravel+DailyRate+Department+DistanceFromHome+EducationField+EnvironmentSatisfaction+MonthlyIncome+NumCompaniesWorked+OverTime+TotalWorkingYears+TrainingTimesLastYear+WorkLifeBalance+YearsInCurrentRole+YearsSinceLastPromotion+YearsWithCurrManager,data=n_hrtest)
View(m2)
colnames(m2)[7] <- 'DepartmentResearch_Development'
colnames(m2)[10] <- 'EducationFieldLife_Sciences'
colnames(m2)[14] <- 'EducationFieldTechnical_Degree'
View(m2)
m2scaled <- scale(m2[,-c(1:2)])
View(m2scaled)
compute.output = compute(mod.net, m2scaled)
n_hrtest$Predict.score = compute.output$net.result
View(n_hrtest)
quantile(n_hrtest$Predict.score, c(0,1,5,10,25,50,75,90,95,99,100)/100)
##            0%             1%             5%            10%            25%            50%            75% 
##-0.21388692351 -0.16810823587 -0.06646280610  0.03779839471  0.03962155151  0.09297336147  0.19840127378 
##           90%            95%            99%           100% 
## 0.47226490506  0.72273461658  1.04827898309  1.27788222149
n_hrtest$deciles <- decile(n_hrtest$Prob)
tmp_DT = data.table(n_hrtest)
h_rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_resp = sum(Attrition), 
  cnt_non_resp = sum(Attrition == 0)) , 
  by=deciles][order(-deciles)]
h_rank$rrate <- round (h_rank$cnt_resp / h_rank$cnt,2);
h_rank$cum_resp <- cumsum(h_rank$cnt_resp)
h_rank$cum_non_resp <- cumsum(h_rank$cnt_non_resp)
h_rank$cum_rel_resp <- round(h_rank$cum_resp / sum(h_rank$cnt_resp),2);
h_rank$cum_rel_non_resp <- round(h_rank$cum_non_resp / sum(h_rank$cnt_non_resp),2);
h_rank$ks <- abs(h_rank$cum_rel_resp - h_rank$cum_rel_non_resp);
h_rank$rrate <- percent(h_rank$rrate)
h_rank$cum_rel_resp <- percent(h_rank$cum_rel_resp)
h_rank$cum_rel_non_resp <- percent(h_rank$cum_rel_non_resp)
View(rank)

