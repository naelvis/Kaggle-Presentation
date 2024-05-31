# Libraries
library(tidyverse)
library(magrittr)
library(vroom)
library(janitor)
library(ggplot2)
library(lubridate)
library(viridis)
library(scales)

# Input
 setwd("/Users/nelvis/Documents/R/Kaggle ALP Presentation/Kaggle-Presentation/Article/Plots")
#setwd("D:/Attila/DataScience/kaggle/actuarial_loss_prediction/presentation/Kaggle-Presentation/Article/Plots/")

predictions <- read_csv("./Backtesting/submission_20211230T113132.csv") %>% 
  rename(Prediction = UltimateIncurredClaimCost)
original <- read_csv("./Backtesting/data_20211230T113132.csv")
original_ultimates <- read_csv("./Backtesting/train.csv") %>% 
  right_join(predictions)

ultimates <- original_ultimates %>% 
  select(UltimateIncurredClaimCost, Prediction) %>% 
  mutate(error = abs(UltimateIncurredClaimCost-Prediction),
         error_perc = error/UltimateIncurredClaimCost) %>% 
  pivot_longer(c(Prediction, error),
               names_to = "Type",
               values_to = "Value")

ggplot(ultimates, aes(x = UltimateIncurredClaimCost,
                      y = Value)) +
  geom_point() +
  facet_grid(Type~ .)

ultimates_v2 <- original_ultimates %>% 
  mutate(error = abs(UltimateIncurredClaimCost-Prediction)) %>% 
  arrange(UltimateIncurredClaimCost) %>% 
  mutate(error_cum = cumsum(error)) %>% 
  mutate(error_percent = error_cum/sum(error)) %>% 
  pivot_longer(c(Prediction, error_percent),
               names_to = "Type",
               values_to = "Value")

ggplot(ultimates_v2, aes(x = UltimateIncurredClaimCost, y = Value)) +
  geom_point() +
  facet_grid(Type~ ., scales = "free")

ultimates_v2a <- original_ultimates %>% 
  mutate(error = abs(UltimateIncurredClaimCost-Prediction)) %>% 
  arrange(InitialIncurredCalimsCost) %>% 
  mutate(error_cum = cumsum(error),
         ultimate_cum = cumsum(UltimateIncurredClaimCost),
         pred_cum = cumsum(Prediction),
         initial_cum = cumsum(InitialIncurredCalimsCost)) %>% 
  mutate(error_percent = error_cum/sum(error),
         ultimate_percent = ultimate_cum/sum(UltimateIncurredClaimCost),
         pred_percent = pred_cum/sum(Prediction),
         initial_percent = initial_cum/sum(InitialIncurredCalimsCost)) %>% 
  pivot_longer(c(error_percent, pred_percent, ultimate_percent),
               names_to = "Type",
               values_to = "Value")

ggplot(ultimates_v2a, aes(x = initial_percent, color = Type)) +
  geom_point(aes(y = Value))

ggplot(ultimates_v2a, aes(x = InitialIncurredCalimsCost, y = Prediction))+
  geom_point() +
  coord_equal()

ultimates_v3 <- original_ultimates %>% 
  mutate(error = abs(UltimateIncurredClaimCost-Prediction)) %>% 
  pivot_longer(c(Prediction, UltimateIncurredClaimCost, error),
               names_to = "Type",
               values_to = "Value")

ggplot(ultimates_v3, aes(x = Value)) +
  geom_density() +
  facet_grid(Type~ ., scales = "free") +
  coord_cartesian(xlim = c(0, 100000))

ultimates_v4 <- original_ultimates %>% 
  mutate(error = abs(UltimateIncurredClaimCost-Prediction)) %>% 
  arrange(UltimateIncurredClaimCost) %>% 
  mutate(mean_error = mean(error)) %>% 
  pivot_longer(c(mean_error, error),
               names_to = "Type",
               values_to = "Value")

ggplot(ultimates_v4, aes(x = UltimateIncurredClaimCost, y = Value, color = Type)) +
  geom_point()

gg_qq_empirical <- function(a, b, quantiles = seq(0, 1, 0.01))
{
  a_lab <- deparse(substitute(a))
  if(missing(b)) {
    b <- rnorm(length(a), mean(a), sd(a))
    b_lab <- "normal distribution"
  }
  else b_lab <- deparse(substitute(b))
  
  ggplot(mapping = aes(x = quantile(a, quantiles), 
                       y = quantile(b, quantiles))) + 
    geom_point() +
    geom_abline(aes(slope = 1, intercept = 0), linetype = 2) +
    labs(x = paste(deparse(substitute(a)), "quantiles"), 
         y = paste(deparse(substitute(b)), "quantiles"),
         title = paste("Empirical qq plot of", a_lab, "against", b_lab))
}

qq <- gg_qq_empirical(ultimates_v2a$UltimateIncurredClaimCost, 
                      ultimates_v2a$Prediction)
qq + theme_light() +
  coord_cartesian(xlim = c(0, 50000)) +
  coord_equal()

ultimates_v5 <- original_ultimates %>% 
  filter(InitialIncurredCalimsCost > 9000,
         InitialIncurredCalimsCost < 11000,) %>% 
  mutate(error = abs(UltimateIncurredClaimCost-Prediction)) %>% 
  mutate(back = str_detect(ClaimDescription, "BACK"))

ggplot(ultimates_v5, aes(x = UltimateIncurredClaimCost, 
                         y = Prediction,
                         color = back)) +
  geom_point()+
  scale_color_viridis_d() +
  coord_cartesian(xlim=c(0,30000))

ultimates_v6 <- original_ultimates %>% 
  mutate(error = abs(UltimateIncurredClaimCost-Prediction),
         error_initial = abs(UltimateIncurredClaimCost-InitialIncurredCalimsCost)) %>% 
  arrange(UltimateIncurredClaimCost) %>% 
  mutate(error_cum = cumsum(error),
         error_in_cum = cumsum(error_initial),
         ultimate_cum = cumsum(UltimateIncurredClaimCost),
         pred_cum = cumsum(Prediction),
         initial_cum = cumsum(InitialIncurredCalimsCost)) %>% 
  mutate(error_percent = error_cum/sum(error),
         error_in_percent = error_in_cum/sum(error_initial),
         ultimate_percent = ultimate_cum/sum(UltimateIncurredClaimCost),
         pred_percent = pred_cum/sum(Prediction),
         initial_percent = initial_cum/sum(InitialIncurredCalimsCost)) %>% 
  pivot_longer(c(error_percent, error_in_percent),
               names_to = "Type",
               values_to = "Value")
  
ggplot(ultimates_v6, aes(x = ultimate_percent, color = Type)) +
  geom_point(aes(y = Value))

ultimates_v7 <- original_ultimates %>% 
  select(InitialIncurredCalimsCost,UltimateIncurredClaimCost, Prediction) %>% 
  pivot_longer(c(Prediction, UltimateIncurredClaimCost, InitialIncurredCalimsCost),
               names_to = "Type",
               values_to = "Value") %>% 
  mutate(Type = ifelse(Type == "InitialIncurredCalimsCost",
                       "Initial Estimation",
                       Type),
         Type = ifelse(Type == "UltimateIncurredClaimCost",
                       "True Ultimate",
                       Type))

ggplot(ultimates_v7, aes(x = log(Value))) +
  geom_density(aes(fill = Type), alpha = .5) +
  geom_line(stat = "density") +
  xlim(5, 13)+
  facet_grid(.~Type) +
  guides(fill = "none") +
  labs(x="Logarithm of the ultimate",
       y ="Density",
       title = "Comparison of the ultimates")+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(filter(ultimates_v7, Type != "Prediction"), aes(x = log(Value))) +
  geom_density(aes(fill = Type), alpha = .75) +
  geom_line(stat = "density") +
  xlim(5, 13)+
  facet_grid(Type~.) +
  guides(fill = "none") +
  labs(x="Logarithm of the ultimate",
       y ="Density",
       title = "Ultimates Comparison")+
  scale_fill_manual(values = c("True Ultimate" = "#A65200", "Initial Estimation" = "#51504B")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(color = "#737111"),
        axis.text = element_text(color = "#737111"),
        strip.text = element_text(color = "#737111"),
        plot.background = element_rect(fill = "#EAEBE3"),
        panel.background = element_rect(fill = "#EAEBE3")) 

ultimates_v7a <- original_ultimates %>% 
  select(UltimateIncurredClaimCost, ClaimDescription) %>% 
  mutate(back = str_detect(ClaimDescription, "BACK")) %>% 
  mutate(Total_Ultimate = UltimateIncurredClaimCost,
         Total_Back = Total_Ultimate*back,
         Total_nonBack = Total_Ultimate*(1-back)) %>% 
  select(-c(UltimateIncurredClaimCost,ClaimDescription, back)) %>% 
  pivot_longer(c(Total_Ultimate, Total_Back, Total_nonBack),
               names_to = "Type",
               values_to = "Value") %>% 
  mutate(Source = ifelse(Type == "Total_Ultimate",
                       "Original",
                       "Differentiating Back injuries"))

ggplot(ultimates_v7a, aes(x = (Value))) +
  geom_density(aes(fill = Type), alpha = .5) +
#  geom_line(stat = "density") +
  xlim(1, 20000)+
  facet_wrap(.~Source, scales = "free") +
  guides(fill = "none") +
  labs(x="Logarithm of the ultimate",
       y ="Density",
       title = "Comparison of the ultimates")+
  theme(plot.title = element_text(hjust = 0.5))

ppi <- 300
png("UltimateComparison_back.png",
    width = 3120,
    height = 1200,
    res = ppi)
ggplot(ultimates_v7a, aes(x = (Value))) +
  geom_density(aes(fill = Type), alpha = .5) +
  xlim(1, 20000)+
  facet_wrap(.~Source, scales = "free") +
  guides(fill = "none") +
  labs(x="Logarithm of the ultimate",
       y ="Density",
       title = "Comparison of the ultimates")+
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

ppi <- 300
png("UltimateComparison.png",
    width = 4*ppi,
    height = 4*ppi,
    res = ppi)
ggplot(ultimates_v7, aes(x = log(Value))) +
  geom_density(aes(fill = Type), alpha = .5) +
  geom_line(stat = "density") +
  xlim(5, 13)+
  facet_grid(.~Type) +
  guides(fill = "none") +
  labs(x="Logarithm of the ultimate",
       y ="Density",
       title = "Comparison of the ultimates")+
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

ggplot(ultimates_v7, aes(x = Value)) +
  geom_density(aes(fill = Type), alpha = .5) +
  geom_line(stat = "density") +
  xlim(0, 50000)+
  facet_grid(Type~.)

gesamtschaden <- original_ultimates %>% 
  mutate(Accident_Year = year(DateTimeOfAccident),
         Sign = str_detect(ClaimDescription, "BACK")) %>% 
  group_by(Accident_Year) %>% 
  summarise(Total_Ultimate = sum(UltimateIncurredClaimCost),
            Total_Prediction = sum(Prediction),
            Total_Sign = sum(Sign),
            Total = n()) %>% 
  mutate(Error = (Total_Ultimate - Total_Prediction),
         Error_Percent = abs(Error)/Total_Ultimate,
         Sign_Percent = abs(Total_Sign)/Total)

ppi <- 300
png("TotalError.png",
    width = 4*ppi,
    height = 4*ppi,
    res = ppi)
ggplot(gesamtschaden, aes(x = as.factor(Accident_Year), y = (Error_Percent))) +
  geom_line(group= 0, aes(color = Total_Ultimate)) +
  geom_point(aes(color = Total_Ultimate, size = Total_Prediction)) +
  labs(x="Accident year",
       y ="Error (%)",
       title = "Prediction error (%) of the ultimate claim per accident year",
       color = "Ultimate",
       size = "Prediction")+
  guides(size = guide_legend(reverse = TRUE)) +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5)) +
  scale_color_viridis_c(labels = comma) +
  scale_size_continuous(labels = comma) +
  scale_y_continuous(labels = percent)
dev.off()

ggplot(gesamtschaden, aes(x = as.factor(Accident_Year), y = ((Error)))) +
  geom_line(group= 0, aes(color = Total_Ultimate)) +
  geom_point(aes(color = Total_Ultimate, size = Total_Prediction)) +
  labs(x="Accident year",
       y ="Error",
       title = "Percentual error on the yearly ultimate",
       color = "Total Ultimate",
       size = "Total Prediction")+
  guides(size = guide_legend(reverse = TRUE)) +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5)) +
  scale_color_viridis_c(labels = comma) +
  scale_size_continuous(labels = comma) +
  scale_y_continuous(labels = comma)

gesamtschaden_v2 <- original_ultimates %>% 
  mutate(Accident_Year = year(DateTimeOfAccident),
         Sign = sign(UltimateIncurredClaimCost - Prediction),
         Error = UltimateIncurredClaimCost - Prediction)

ggplot(gesamtschaden_v2, aes(x = as.factor(Accident_Year), y = Error)) +
  geom_violin()

ggplot(gesamtschaden_v2, aes(x = log(UltimateIncurredClaimCost), fill =as.factor(Sign))) +
  geom_histogram(position = "dodge")

cor_jahr <- function(jahr) {
  a <- filter(gesamtschaden_v2, Accident_Year == jahr)
  cor(a$UltimateIncurredClaimCost, a$InitialIncurredCalimsCost)
}

ggplot(arrange(gesamtschaden_v2,desc(Error)), aes(x = as.factor(Accident_Year), y = (UltimateIncurredClaimCost), fill = ClaimNumber)) +
  geom_col(color = "black") +
  theme(legend.position = "none")

ggplot(gesamtschaden_v2, aes(x = (Accident_Year), y = (Error))) +
  geom_point()

gesamtschaden_v3 <- original_ultimates %>% 
  mutate(Accident_Year = year(DateTimeOfAccident)) %>% 
  group_by(Accident_Year) %>% 
  summarise(`Real` = sum(UltimateIncurredClaimCost),
            `Predicted` = sum(Prediction)) %>%
  pivot_longer(c(`Real`, `Predicted`),
               names_to = "Ultimate",
               values_to = "Value")
  
ppi <- 300
png("230129TotalUltimate.png",
    width = 8*ppi,
    height = 4*ppi,
    res = ppi)
ggplot(gesamtschaden_v3, aes(x = as.factor(Accident_Year), y = (Value), group = Ultimate, fill = Ultimate)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x="Accident year",
       y ="Ultimate Value",
       title = "Ultimate per accident year",
       color = "Ultimate",
       size = "Prediction")+
  guides(size = guide_legend(reverse = TRUE)) +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5)) +
  scale_fill_viridis_d(option = "E") +
  scale_size_continuous(labels = comma) +
  scale_y_continuous(labels = comma)
dev.off()

gesamtschaden_v4 <- original_ultimates %>% 
  mutate(Accident_Year = year(DateTimeOfAccident)) %>% 
  group_by(Accident_Year) %>% 
  summarise(`Real` = sum(UltimateIncurredClaimCost),
            `Predicted` = sum(Prediction),
            Error = Real-Predicted) 

ggplot(gesamtschaden_v4, aes(x = as.factor(Accident_Year), y = Error)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x="Accident year",
       y ="Error",
       title = "Error per accident year",
       color = "Ultimate",
       size = "Prediction")+
  guides(size = guide_legend(reverse = TRUE)) +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5)) +
  scale_fill_viridis_d(option = "E") +
  scale_size_continuous(labels = comma) +
  scale_y_continuous(labels = comma)

ggplot(original, aes(y = UltimateIncurredClaimCost)) +
  geom_boxplot()

# Oslo #######

ggplot(filter(ultimates_v7, Type != "Prediction"), aes(x = log(Value))) +
  geom_density(aes(fill = Type), alpha = .75) +
  geom_line(stat = "density") +
  xlim(5, 13)+
  facet_grid(.~Type) +
  guides(fill = "none") +
  labs(x="Logarithm of the ultimate",
       y ="Density",
       title = "Ultimates Comparison")+
  scale_fill_manual(values = c("True Ultimate" = "#A65200", "Initial Estimation" = "#51504B")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(color = "#737111"),
        axis.text = element_text(color = "#737111"),
        strip.text = element_text(color = "#737111"),
        plot.background = element_rect(fill = "#EAEBE3"),
        panel.background = element_rect(fill = "#EAEBE3")) 

set.seed(42)
ggplot(sample_n(original_ultimates,2500), aes(x = log(InitialIncurredCalimsCost),
                      y = log(UltimateIncurredClaimCost))) +
  geom_point(alpha = 1, color = "#355368") +
  geom_smooth(color = "#A65200") +
  xlim(5, 13)+
  guides(fill = "none") +
  labs(x="Logarithm of the initial estimation",
       y ="Logarithm of the true ultimate",
       title = "True Ultimate vs Initial Estimation")+
  scale_fill_manual(values = c("True Ultimate" = "#A65200", "Initial Estimation" = "#51504B")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(color = "#737111"),
        axis.text = element_text(color = "#737111"),
        strip.text = element_text(color = "#737111"),
        plot.background = element_rect(fill = "#EAEBE3"),
        panel.background = element_rect(fill = "#EAEBE3")) 

ultimates_v7b <- ultimates_v7a %>% 
  mutate(Type = ifelse(Type == "Total_Ultimate",
                       "Total Ultimate",
                       Type),
         Type = ifelse(Type == "Total_nonBack",
                       "Does not contain the word 'Back'",
                       Type),
         Type = ifelse(Type == "Total_Back",
                       "Contains the word 'Back'",
                       Type))

ggplot(ultimates_v7b, aes(x = (Value))) +
  geom_density(aes(fill = Type), alpha = .5) +
  xlim(1, 20000)+
  facet_wrap(.~Source) +
  #guides(fill = "none") +
  guides(fill = guide_legend(reverse = TRUE)) +
  scale_fill_manual(values = c("Total Ultimate" = "#737111", "Does not contain the word 'Back'" = "#A65200", "Contains the word 'Back'" = "#B49056")) +
  labs(x="Ultimate",
       y ="Density",
       title = "Comparison of the ultimates")+
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.y = element_blank(),  # Remove y-axis text/labels
       # axis.ticks.y = element_blank(),
        text = element_text(color = "#737111"),
        axis.text = element_text(color = "#737111"),
        strip.text = element_text(color = "#737111"),
        plot.background = element_rect(fill = "#EAEBE3"),
        panel.background = element_rect(fill = "#EAEBE3"),
       legend.position = "bottom") 

ggplot(filter(gesamtschaden_v3, Accident_Year %in% seq(1980,2015)), aes(x = as.factor(Accident_Year), y = (Value), group = Ultimate, fill = Ultimate)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 1) +
  labs(x="Accident year",
       y ="Ultimate Value",
       title = "Ultimate per accident year",
       color = "Ultimate",
       size = "Prediction")+
  guides(size = guide_legend(reverse = TRUE)) +
  scale_fill_manual(values = c("Predicted" = "#56758A", "Real" = "#DF690B")) +
  scale_size_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5),
        #axis.text.y = element_blank(),  # Remove y-axis text/labels
        # axis.ticks.y = element_blank(),
        text = element_text(color = "#737111"),
        axis.text = element_text(color = "#737111"),
        strip.text = element_text(color = "#737111"),
        plot.background = element_rect(fill = "#EAEBE3"),
        panel.background = element_rect(fill = "#EAEBE3"),
        legend.position = "bottom") 

