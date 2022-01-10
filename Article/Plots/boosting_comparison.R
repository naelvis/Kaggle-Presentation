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
  str_subset(pattern = "csv") %>% 
  map_dfr(~ .x %>% 
            vroom() %>% 
            mutate(Source = str_extract(.x,
                                        pattern = "(?<=data/).*(?=.csv)"))
  ) %>% 
  clean_names()

boost_plot <- input %>% 
  select(x1,
         test_rmse_mean,
         test_rmse_std,
         source) %>% 
  filter(str_detect(source, "forest_nlp_boost"),
         str_detect(source, "gamma", negate = TRUE)) %>% 
  mutate(source = ifelse(str_detect(source, "rmse"),
                                    "Squared Error",
                                    "Tweedie\nLikelihood"))

  ggplot(boost_plot, aes(x = x1, y = test_rmse_mean, color = source)) +
    geom_line() +
    xlim(c(0, 200)) +
    annotate("rect",
             xmin = 15,
             xmax = 75,
             ymin = min(boost_plot$test_rmse_mean),
             ymax = max(boost_plot$test_rmse_mean),
             alpha = .1,
             fill = "blue") +
    labs(color = "Loss Function",
         x = "Boosting round",
         y = "RMSE",
         title = "Boosting for different loss functions")
  