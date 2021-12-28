# ML Algorithms for reserving

## A case study from Kaggle

## Introduction

There is much ado about data science and machine learning these days. ML algorithms can identify pictures of traffic lights and drive cars: from an actuarial perspective, the next step is necessarily that they be able to adequately reserve for accidents they might cause.

Jokes aside, the question of how and how much such algorithms can be integrated in the reserving processes of insurance companies is very real and urgent. Companies making use of these new solutions might gain a huge competitive advantage in the market, on the other laying themselves open to their critical aspects, for example lack of transparency.

The Kaggle competition "Actuarial Loss Prediction" (ALP) has been for the authors a good place to start understanding the issue and gain some practical insight. Kaggle is the leading platform for online data science competitions, with a community of over 8 millions of user. The competition was presented as follows:

> The Actuaries Institute of Australia, Institute and Faculty of Actuaries and the Singapore Actuarial Society are delighted to host the Actuarial loss prediction competition 2020/21 to promote development of data analytics talent especially among actuaries. The challenge is to predict Workers Compensation claims using highly realistic synthetic data.
>
> The data is fully synthetic and not specific to any legal jurisdiction or country. We are grateful to Colin Priest for building and supplying the dataset. 
>
> We invite the competitors to take claims inflation into account.

ALP ran from December 2020 to April 2021, keeping 140 teams of actuaries and data scientists busy. At the end we retained the second place and some good insights about how machine learning in reserving might look like in the future, which we would like to share in this article.

## Aim of the competition

The introducing text of ALP already provides a succint description of the aim: "The challenge is to predict Workers Compensation claims using highly realistic synthetic data.". In this section we provide a bit more detail on the aim of the competition and the scoring system.

The 90.000 rows dataset was divided in a train set (54.000 rows), test set (18.000 rows) and validation set (18.000 rows), with each row containing the following data fields:

- **ClaimNumber**: Unique policy identifier
- **DateTimeOfAccident**: Date and time of accident
- **DateReported**: Date that accident was reported
- **Age**: Age of worker
- **Gender**: Gender of worker
- **MaritalStatus**: Martial status of worker. (M)arried, (S)ingle, (U)nknown.
- **DependentChildren**: The number of dependent children
- **DependentsOther**: The number of dependants excluding children
- **WeeklyWages**: Total weekly wage
- **PartTimeFullTime**: Binary (P) or (F)
- **HoursWorkedPerWeek**: Total hours worked per week
- **DaysWorkedPerWeek**: Number of days worked per week
- **ClaimDescription**: Free text description of the claim
- **InitialIncurredClaimCost**: Initial estimate by the insurer of the claim cost
- **UltimateIncurredClaimCost**: Total claims payments by the insurance company. 

The last field is the field that needed to be predicted. 

The typical ML workflow is to train the algorithm on the train dataset and check its predictions on the test dataset, to make sure it is not too tightly tailored on the data it was trained upon. Based on the results on the test dataset the algorithms is tweaked, and in the very last step - to check whether the model was indirectly tailored upon the test dataset - the model runs on the never seen before validation dataset. An algorithm able to perform well and equally well on the train, test and validation dataset has captured the general features of the problem and neglected idiosincracies of the single datasets. 

For this competition the predictions were scored according to the root mean squared error (RMSE)

## Data Exploration

## Natural Language Processing

## Modelling with xgboost 4076

We implemented our model in Python making use of the xgboost library. In the words of its own authors this library "implements machine learning algorithms under the Gradient Boosting framework", and we applied it to boost a random forest algorithm. 

In this section we shortly refresh the concepts of random forest and gradient boosting and then describe the way xgboost introduces probability distributions in its algorithms.

### Random Forests and Gradient Boosting

Here is a short recap of the concepts of random forests and gradient boosting:

- **Random forests** are made up of decision trees - hence the "forest" -, each of which is only allowed to train on a casually selected subset of the training set - hence the "random". Each of the decision trees populating the forest predicts a result, and a unique result is then provided by taking the average of all results (for regression problems) or the most common one (for classification problems).
- **Gradient boosting** is a machine learning technique in which the same algorithm is applied iteratively. After a first prediction has been provided, we fit a new model on its residuals - this is the first boosting round. We can then combine the first two models and model their residuals, which would be the second boosting round, and so forth.

In combining random forests and gradient boosting we refine the classic decision tree algorithm in two different ways. The predictions improve at the cost of interpretability: building up a forest we lose the possibility of looking at the individual splits of our decision tree, and each boosting round adds a new forest. A simple way of visualizing feature importance is the F-score, which counts how many times a variable was split on (see Figure X).

### Regression to distributions in xgboost

Suppose that we want to predict a numerical outcome Y based on a set of n dependent variables X1, ..., Xn, and that we suspect Y to be gamma distributed with mean mu: a common way to approach this regression problem would be to set up a generalized linear model (GLM) around the equation -1/mu = g(mu) = betaX, where the betas are regression parameters to be determined. We can determine the betas by maximing likelihood.

This approach cannot be translated directly to decision trees, which describe the outcome Y in term of piecewise constant functions rather than polynomials. But it is possible to introduce the likelihood in the computation via the loss function. We explain this important point in more detail.

The learning process of a machine learning algorithm is guided by minimizing a loss function. Given a vector of true values Y and of model outcomes Yhat we can for example ask for the Yhat that minimizes the squared error: insert equation or the squared log error: insert equation. One typical issue of the squared error is that it is driven by large outcomes - a small percentual error in predicting a large value, possibly due to noise, has a great impact on the squared error. This effect is tamed by the squared log error.

Loss functions are in principle fully customizable, but xgboost already provides a wide selection, including the negative of the likelihood to logistic, gamma and tweedie distributions. Minimizing the negative of the likelihood obviously maximizes the likelihood; this is however not achieved (as in GLMs) by finding the optimal betas, but by finding the best splits for each decision tree.

## Results

In this article we wrote about different techniques to analyse and model data. In this section we compare different versions of our model. This is our final model: insert list of parameters. In the following sections we modify single aspects and look at differences in model performance.

### Random forest

The plot below shows the difference between a model with random forest (50 trees) and a model with only one tree:

Here there is a comment about the picture. Insert comment on runtime.

### Loss functions



### NLP



### True data

## Conclusion

This competition was a useful exercise in machine learning and its potential in the insurance industry. While very little hard coding was necessary, we realised how important it was to have a good grasp of the theory behind the models: choosing to use random forests only leads to up to many more questions, for example what the maximal depth of the trees should be, or how many trees the random forest should be made up of[footnote]. Already choosing a range of hyperparameters has been at times challenging.

We also tried several different machine learning algorithms, including neural networks. While boosting on neural network is a known concept (see X), an equivalent of xgboost is missing. So it is well possible that powerful models remain unused just because they are hard to implement with the available packages.

Interpretability of the results remained a problem throughout. We used feature importance as a guideline and spent a lot of time looking at what the model was missing to understand how to improve our results. A straightforward way of interpreting is still not available and is perhaps just not achievable, given the way ML algorithms are constructed. This remains an open challenge for the future blah blah blah

