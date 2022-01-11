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
                                           "tweedie"))) %>% 
  mutate(NLP = ifelse(NLP == "nlp",
                      "With NLP",
                      "Without NLP"),
         Bagging = ifelse(Bagging == "forest",
                          "RF (50)",
                          "Single tree"),
         Boosting = ifelse(Boosting == "boost",
                           "300 boosting rounds",
                           "Without boosting"),
         Loss_Function = ifelse(Loss_Function == "rmse",
                                "Squared\nError",
                                ifelse(Loss_Function == "gamma",
                                       "Gamma\nLikelihood",
                                       "Tweedie\nLikelihood"))) %>% 
  mutate(Loss_Function = factor(Loss_Function,
                                levels = c("Squared\nError",
                                           "Gamma\nLikelihood",
                                           "Tweedie\nLikelihood")))


ggplot(data_plot, aes(shape = Bagging,
                      color = NLP, 
                      x = Loss_Function,
                      y = test_rmse_mean)) +
  geom_point(position = position_dodge(width = 0.3)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  facet_grid(factor(Boosting, 
                    levels= c("Without boosting", "300 boosting rounds"))~., 
             scales = "free") +
  labs(x="Loss Function",
       y ="RMSE",
       title = "Model comparison")+
  theme(plot.title = element_text(hjust = 0.5))

ppi <- 300
png("ModelComparison.png",
    width = 4*ppi,
    height = 4*ppi,
    res = ppi)
ggplot(data_plot, aes(shape = Bagging,
                      color = NLP, 
                      x = Loss_Function,
                      y = test_rmse_mean)) +
  geom_point(position = position_dodge(width = 0.3)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  facet_grid(factor(Boosting, 
                    levels= c("Without boosting", "300 boosting rounds"))~., 
             scales = "free") +
  labs(x="Loss Function",
       y ="RMSE",
       title = "Model comparison")+
  theme(plot.title = element_text(hjust = 0.5))
dev.off()


+
  coord_cartesian(ylim = c(23000, 24000))

ggplot(data_plot, aes(x = 1, fill = source, y = test_rmse_mean)) +
  geom_col(position = "identity") +
  coord_cartesian(ylim = c(0, 23292.20)) 





