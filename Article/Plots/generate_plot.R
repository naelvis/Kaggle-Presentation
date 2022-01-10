# Libraries
library(tidyverse)
library(magrittr)
library(vroom)
library(janitor)
library(ggplot2)

# Input
setwd("/Users/nelvis/Documents/R/Kaggle ALP Presentation/Kaggle-Presentation/Article/Plots")

input <- "./data" %>% 
  list.files(full.names = TRUE) %>% 
  map_dfr(~ .x %>% 
            vroom() %>% 
            mutate(Source = str_extract(.x,
                                        pattern = "(?<=data/).*(?=.csv)"))
          ) %>% 
  clean_names()

data_plot <- input %>% 
  select(test_rmse_mean,
         test_rmse_std,
         source) %>% 
  group_by(source) %>%
  filter(test_rmse_mean == min(test_rmse_mean)) %>% 
  ungroup() %>% 
  arrange(desc(test_rmse_mean)) %>% 
  separate(source,
           c("Loss_Function",
             "Bagging",
             "NLP",
             "Boosting")) %>% 
  mutate(Loss_Function = factor(Loss_Function,
                                levels = c("rmse",
                                           "gamma",
                                           "tweedie")))
  
ggplot(data_plot, aes(x = 1, fill = source, y = test_rmse_mean)) +
  geom_col(position = "identity") +
  coord_cartesian(ylim = c(0, 23292.20))

ggplot(data_plot, aes(shape = Bagging,
                      color = NLP, 
                      x = Loss_Function,
                      y = test_rmse_mean)) +
  geom_point(position = position_dodge(width = 0.3)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  facet_grid(factor(Boosting, 
                    levels= c("noboost", "boost"))~., 
             scales = "free")

+
  coord_cartesian(ylim = c(23000, 24000))







