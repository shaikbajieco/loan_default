library(ggplot2)
library(dplyr)
library(tidyr)
library(tidymodels)
library(car)


### Importing data

loan_data_train = read.csv("E:\\Data Analytics\\R Language\\Data\\Additional Datasets\\loan_data_train.csv")
loan_data_test = read.csv("E:\\Data Analytics\\R Language\\Data\\Additional Datasets\\loan_data_test.csv")

glimpse(loan_data_train)
glimpse(loan_data_test)

names(loan_data_train)
names(loan_data_test)

# 'Interest.Rate' is the dependent variable. We need predict the Interest.Rate for
# test data by analysing (building regression model on) train data set.

### Data Preparation (Feature Engineering)

# For data manipulation and maintain consistency for both data sets, two data 
# sets must be combined first. To do so add Interest.Rate column to test data 
# set to make columns of both data sets match. And introducing data column to 
# separate separate both data sets letter on.

loan_data_test$Interest.Rate = NA
loan_data_test$Data = 'test'
loan_data_train$Data = 'train'

loan_data = rbind(loan_data_train, loan_data_test)

glimpse(loan_data)

# Here observe that, data type of these features namely ['Amount.Requested', 
# 'Amount.Funded.By.Investors', 'Interest.Rate', 'Debt.To.Income.Ratio', 
# 'Open.CREDIT.Lines' and 'Revolving.CREDIT.Balance'] recorded as Character data
# type. Converting those column's data type from Character to Numeric

loan_data = loan_data %>% 
  mutate(Amount.Requested = as.numeric(Amount.Requested),
         Amount.Funded.By.Investors = as.numeric(Amount.Funded.By.Investors),
         Interest.Rate = as.numeric(gsub("%", "", Interest.Rate)),
         Debt.To.Income.Ratio = as.numeric(gsub("%", "", Debt.To.Income.Ratio)),
         Open.CREDIT.Lines = as.numeric(Open.CREDIT.Lines),
         Revolving.CREDIT.Balance = as.numeric(Revolving.CREDIT.Balance)
         )

# The variable 'Amount.Funded.By,Investors' not valid to consider for model 
# building. Because it is recorded after the decision making, so drop it.

loan_data$Amount.Funded.By.Investors = NULL

# Now move on to some complected tasks. The 'FICO.Range' variable represents 
# credit score, but data mentioned in ranges will not be suitable for machine 
# learning modelling. For that reason converting the variable into a convenient 
# format, that is by calculating average of two values.

table(loan_data$FICO.Range)

loan_data = loan_data %>% 
  mutate(f1 = as.numeric(substr(FICO.Range, 1, 3)),
         f2 = as.numeric(substr(FICO.Range, 5, 7)),
         FICO = (f1 + f2)/2
         ) %>% 
  select(-FICO.Range, -f1, -f2)

# The variable 'Employment.Length' also need to trim character portion to make it 
# convenient for machine learning modelling.

table(loan_data$Employment.Length)

loan_data = loan_data %>% 
  mutate(Employment = ifelse(substr(Employment.Length, 1, 2) == "10", 10, Employment.Length),
         Employment = ifelse(substr(Employment.Length, 1, 1) == "<", 0, Employment),
         Employment = gsub("years", "", Employment),
         Employment = gsub("year", "", Employment),
         Employment = as.numeric(Employment)
         ) %>% 
  select(-Employment.Length)


# 'Loan.Purpose', here taking a very subjecting move, one can make dummy variable
# for all possible categories. But in this modelling process considering similar
# and combining into one group.Creating dummy variables for those combined groups,
# this helps to bring down number of categories.

table(loan_data$Loan.Purpose)

round(tapply(loan_data$Interest.Rate, loan_data$Loan.Purpose, mean, na.rm = T))

loan_data = loan_data %>%
  mutate(Loan.Purpose.10 = as.numeric(Loan.Purpose == 'educational'),
         Loan.Purpose.11 = as.numeric(Loan.Purpose %in% c("major_purchase","medical","car")),
         Loan.Purpose.12 = as.numeric(Loan.Purpose %in% c("vacation","wedding","home_improvement")),
         Loan.Purpose.13 = as.numeric(Loan.Purpose %in% c("other","small_business","credit_card")),
         Loan.Purpose.14 = as.numeric(Loan.Purpose %in% c("debt_consolidation","house","moving"))) %>%
  select(-Loan.Purpose)

# Creating dummy variables for other categorical variables namely ["Loan.Length", 
# "State", "Home.Ownership"], But creating dummy variables for very small frequency
# categories will lead to perfect multicollinearity problem. For that here considering
# categories with minimum of 100 observations.

glimpse(loan_data)

cat.vars = c("Loan.Length", "State", "Home.Ownership")

# Writing function for creating dummy variables

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    data[,name]=as.numeric(data[,var]==cat)
  }
  data[,var]=NULL
  return(data)
}

# Creating dummies for variables in 'cat.vars' with frequency greater than 100.

for(col in cat.vars){
  loan_data = CreateDummies(loan_data, col, 100)
}

# Impute missing values with column's mean value of train data

lapply(loan_data, function(x) sum(is.na(x)))

loan_data = loan_data[!(is.na(loan_data$ID)),] # Not a valid row 

for(col in names(loan_data)){
  if(sum(is.na(loan_data[,col])) > 0 & !(col %in% c("ID","Data","Interest.Rate"))){
    loan_data[is.na(loan_data[,col]),col] = mean(loan_data[loan_data$Data=="train",col],na.rm=T)
  }
}

### Data preparation completed, separate train & test data.

loan_data_train_new = loan_data %>% filter(Data == "train") %>% select(-Data)
loan_data_test_new = loan_data %>% filter(Data == "test") %>% select(-Data, -Interest.Rate)

# Divide train data into two part (in 70:30 ratio) to build regression model
# on one data set and check its performance of the other data set. Because
# directly predicting Interest.Rate on test data set will not help us to 
# know about model accuracy.

set.seed(21)
s = sample(1:nrow(loan_data_train_new), 0.7*nrow(loan_data_train_new))

sample_train = loan_data_train_new[s,]
sample_test = loan_data_train_new[-s,]

# Before running running regression model it is important to check whether
# there any linearly dependent variables present in the data set, they can creat
# multicollinearity problem in model. It will be checked by VIF value.

fit_lm = lm(Interest.Rate~ . -ID, data = sample_train)
sort(vif(fit_lm), decreasing = T)[1:3]

fit_lm = lm(Interest.Rate~ . -ID -Loan.Purpose.14, data = sample_train)
sort(vif(fit_lm), decreasing = T)[1:3]

# Considering ID column in the model is meaningless, because it is for referential 
# purpose. The variable 'Loan.Purpose.14' is linearly dependent on other column in
# the data set.

# Next, using step function removing variables those not statistically significant.

fit_lm = lm(Interest.Rate~ . -ID -Loan.Purpose.14, data = sample_train)
fit_lm = stats::step(fit_lm)

formula(fit_lm)

fit_lm = lm(Interest.Rate ~ Amount.Requested + Open.CREDIT.Lines + 
              Inquiries.in.the.Last.6.Months + FICO + Loan.Length_36months + 
              State_TX + Home.Ownership_MORTGAGE, data = sample_train)
summary(fit_lm) 

# In summary, observe that Adjusted R-squared is ~0.76. It tells that 76% variability
# in outcome (that is Interest.Rate) has been explained by this model.

# Below figure will visualize the how close this model predicts Interest.Rate

sample_train %>%
  mutate(pred_Interest.Rate = predict(fit_lm, newdata = sample_train)) %>%
  ggplot(aes(x = Interest.Rate, y = pred_Interest.Rate)) + geom_point(alpha = 0.6)

# Check RMSE

rmse = mean((sample_test$Interest.Rate - predict(fit_lm, newdata = sample_test)) ^ 2) %>% 
  sqrt()
rmse

# This rmse value (which is ~2.167) can be compare with other model's rmse value
# to select best model. 

### Final model: Using entire training data

