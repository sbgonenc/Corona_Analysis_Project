# Title     : TODO
# Objective : TODO
# Created by: BERK
# Created on: 19/05/2020
require(dplyr)
library(ggplot2)
library(broom)

x<-c(5,4,3,2,1)
y<-c(1,2,3,4,5)

corona_df <- read.delim('dataappended.txt')
corona_all <- read.csv('datasets/ourworld_corona_data.csv')
summary(corona_df)
summary(corona_all)

plot(corona_df)
plot(corona_all)

multiple_regression_test <- lm(total_deaths~total_cases + aged_65_older + diabetes_prevalence +
  female_smokers + male_smokers + population + population_density + hospital_beds_per_100k + gdp_per_capita + extreme_poverty, data = corona_all)

summary(multiple_regression_test)

car::vif(multiple_regression_test)
