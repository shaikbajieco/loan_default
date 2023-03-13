library(ggplot2)
library(dplyr)
library(tidyr)
library(tidymodels)


### Importing data

loan_data_train = read.csv("E:\\Data Analytics\\R Language\\Data\\Additional Datasets\\loan_data_train.csv")
loan_data_test = read.csv("E:\\Data Analytics\\R Language\\Data\\Additional Datasets\\loan_data_test.csv")

glimpse(loan_data_train)
glimpse(loan_data_test)

names(loan_data_train)
names(loan_data_test)

# 'Interest.Rate' is the dependent variable. We need predict the Interest.Rate for
# test data by analysing train data set.

### Data Preparation (Feature Engineering)

# For data manipulation and consistency, two data sets must be combined first. 
loan_data_test$Interest.Rate = NA
loan_data_test$Data = 'test'
loan_data_train$Data = 'train'

loan_data = rbind(loan_data_train, loan_data_test)

glimpse(loan_data)

# Covert data type of these features namely ['Amount.Requested', 'Amount.Funded.By.Investors',
# 'Interest.Rte', 'Debt.To.Income.Ratio', 'Open.CREDIT.Lines' and 'Revolving.CREDIT.Balance']
# from Character to Numeric

loan_data = loan_data %>% 
  mutate(Amount.Requested = as.numeric(Amount.Requested),
         Amount.Funded.By.Investors = as.numeric(Amount.Funded.By.Investors),
         Interest.Rate = as.numeric(gsub("%", "", Interest.Rate)),
         Debt.To.Income.Ratio = as.numeric(gsub("%", "", Debt.To.Income.Ratio)),
         Open.CREDIT.Lines = as.numeric(Open.CREDIT.Lines),
         Revolving.CREDIT.Balance = as.numeric(Revolving.CREDIT.Balance)
         )

# Variable 'Amount.Funded.By,Investors' is recorded after the event, so drop it.

loan_data$Amount.Funded.By.Investors = NULL

# 'FICO.Range' variable represents credit score, but data mentioned in ranges will
# not be suitable for machine learning. For that reason manipulation the variable
# by calculating average of two values.

loan_data = loan_data %>% 
  mutate(f1 = as.numeric(substr(FICO.Range, 1, 3)),
         f2 = as.numeric(substr(FICO.Range, 5, 7)),
         FICO = (f1 + f2)/2
         ) %>% 
  select(-FICO.Range, -f1, -f2)







