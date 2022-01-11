# ML Algorithms for reserving

## A case study from Kaggle

## Introduction

Machine learning advances at an unprecedented pace these days. ML algorithms can read traffic lights and are expected to drive cars soon. From an actuarial perspective, the next step is necessarily that they adequately reserve for accidents they might cause.

Jokes aside, the question of how and how much such algorithms can be integrated into the reserving processes of insurance companies is very real and urgent. Companies making use of these new solutions might gain an edge over their competitors, on the other laying themselves open to their critical aspects, for example lack of transparency.

The "Actuarial Loss Prediction" (ALP) competition aimed to further this discussion by creating a reserving challenge that goes beyond conventional methods. The challenge was hosted on Kaggle, the leading platform for data science competitions with a community of over 8 million data scientists, and was presented as follows:

> The Actuaries Institute of Australia, Institute and Faculty of Actuaries and the Singapore Actuarial Society are delighted to host the Actuarial loss prediction competition 2020/21 to promote development of data analytics talent especially among actuaries. The challenge is to predict Workers Compensation claims using highly realistic synthetic data.
>
> The data is fully synthetic and not specific to any legal jurisdiction or country. We are grateful to Colin Priest for building and supplying the dataset. 
>
> We invite the competitors to take claims inflation into account.

The ALP ran from December 2020 to April 2021, keeping 140 teams of actuaries and data scientists busy. At the end we retained the second place and some good insights about how machine learning in reserving might look like in the future, which we would like to share in this article.

## Aim of the competition

The introduction of the ALP already provides a succint description of the aim: "The challenge is to predict Workers Compensation claims using highly realistic synthetic data.". In this section we provide a bit more detail on the aim of the competition and the scoring metric.

The 90.000 rows dataset was divided into a train set (54.000 rows) and a test set (36.000 rows). Approximately 50% of the test data was retained by the competition host and only used once for the final evaluation of the models. 

The typical ML workflow is to train the algorithm on the train set and check its predictions on a validation set, to make sure it is not too tightly tailored to the data it was trained upon. Based on the results on the validation set the model is fine-tuned, and in the last step - to check whether the model was indirectly tailored upon the validation dataset - the model is evaluated on the unseen test set. An algorithm that is able to perform on the test set has generalized well and successfully neglected the idiosyncracies of the training data. 

In this competition the predictions were scored according to the root mean squared error (RMSE).

## Data Exploration

The train dataset contained the fields below. The test dataset contained the same fields except for the explained variable *UltimateIncurredClaimCost*.

- **ClaimNumber**: Unique policy identifier
- **DateTimeOfAccident**: Date and time of accident
- **DateReported**: Date the accident was reported
- **Age**: Age of the worker
- **Gender**: Gender of the worker
- **MaritalStatus**: Martial status of the worker: (M)arried, (S)ingle, (U)nknown
- **DependentChildren**: The number of dependent children
- **DependentsOther**: The number of dependants excluding children
- **WeeklyWages**: Total weekly wage
- **PartTimeFullTime**: Whether the worker was employed (P)art time or (F)ull time
- **HoursWorkedPerWeek**: Total hours worked per week
- **DaysWorkedPerWeek**: Number of days worked per week
- **ClaimDescription**: Free text description of the claim
- **InitialIncurredClaimCost**: Initial estimate by the insurer of the claim cost
- **UltimateIncurredClaimCost**: Total claims payments by the insurance company

Outliers were removed or recoded based on expert judgment. Some new features were added, e.g. whether the accident happened within core working hours. Date(time) features were also used to create new variables, such as reporting delay. The explanatory variables were partly transformed to alleviate or reinforce the effect of extreme data points. For instance, polynomial features were created for the variables *DependentChildren*, *DependentsOther*.

Different large claim thresholds were tested for different kinds of models. Moreover, several models were calibrated for large claims. However, explicit large claim modelling did not lead to significant model improvements. Therefore, the large claims were accounted for implicitly in the final model.

## Natural Language Processing

The chapter is titled Natural Language Processing (NLP) because the *ClaimDescription* was analyzed in part with NLP techniques. However, the full potential of NLP techniques could not be exploited because some descriptions had been distorted for data protection reasons to a point where they became hardly intelligible, e.g. "TO RIGHT LEG RIGHT KNEE". Therefore, NLP was combined with simple text analysis to extract information from the claim descriptions.

The descriptions were preprocessed as follows: First, stopwords were removed to reduce the noise in the data. The set of stopwords were taken from the nltk corpus. Second, the words were lemmatized using SpaCy, as well as stemmed using the SnowBall stemmer of nltk. Lemmatization reduces the word to its form that appears in the dictionary, stemming reduces the word to its root form. The former produces semantically meaningful word forms, whereas the latter may not. For example, the lemma of lacerated is lacerate, the stem form of lacerated, laceration or lacerate is lacer.

Next, we introduced 3 categories: body part, type of wound, and accident cause. For each category we defined a list of words based on the dataset as well as our own judgment. The lists contained words like: ankle, knee, eye (body parts), strain, bruise, fracture (types of wound), slip, explosion, spider (accident cause). Moreover, we grouped these words based semantic similarity. For instance, face, cheek and jaw were clustered together.

For each word in the lists as well as for each cluster, a new column was created and added to the dataset. The columns contained the number of occurences of the word or cluster(?).

## Modelling with xgboost

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

