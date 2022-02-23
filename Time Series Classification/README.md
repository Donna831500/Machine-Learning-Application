# Time Series Classification

## Abstract
This project is the homework 3 for USC DSCI-552 Machine Learning for Data Science.
The Step section explained the process in detail.
An interesting task in machine learning is classification of time series. In this problem,
we will classify the activities of humans based on time series obtained by a Wireless
Sensor Network


## Data
Download the AReM data from:
https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+\%28AReM\%29.

Or find the data in 'data' directory.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) or Anaconda to install following packages :

- pandas
- numpy
- seaborn
- sklearn
- warnings
- matplotlib
- IPython
- statsmodels

The packages can be installed by following command:
```bash
pip install <package>
```

## Execution
The 'main.ipynb' file is in the 'notebook' directory and data are in the 'data' directory.
Please put the 'notebook' directory and 'data' directory in same directory and execute the 'main.ipynb' file.


## Step
### (a) Download the AReM data from:
https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+\%28AReM\%29 .
The dataset contains 7 folders that represent seven types of activities. In
each folder, there are multiple files each of which represents an instant of a human
performing an activity. Each file contains 6 time series collected from activities
of the same person, which are called avg_rss12, var_rss12, avg_rss13, var_rss13,
vg_rss23, and ar_rss23. There are 88 instances in the dataset, each of which contains 6 time series and each time series has 480 consecutive values.


### (b)
Keep datasets 1 and 2 in folders bending1 and bending 2, as well as datasets 1,
2, and 3 in other folders as test data and other datasets as train data.

### (c) Feature Extraction
Classification of time series usually needs extracting features from them. In this
problem, we focus on time-domain features.

i. Research what types of time-domain features are usually used in time series
classification and list them (examples are minimum, maximum, mean, etc).

ii. Extract the time-domain features minimum, maximum, mean, median, standard deviation, first quartile, and third quartile for all of the 6 time series
in each instance. You are free to normalize/standardize features or use them
directly.

Your new dataset will look like this:
| Instance | min1 | max1 | mean1 | median1 ... 1st quart6 | 3rd quart6 |
| -------- |:----:| ----:|:-----:| ----------------------:|:----------:|
| 1        |      |      |       |                        |            |
| 2        |      |      |       |                        |            |
| 2        |      |      |       |                        |            |
| ...      |  ... |  ... |   ... |           ...          |    ...     |
| 88       |      |      |       |                        |            |

where, for example, 1st quart6, means the first quartile of the sixth time series
in each of the 88 instances.

iii. Estimate the standard deviation of each of the time-domain features you
extracted from the data. Then, use Python’s bootstrapped or any other
method to build a 90% bootsrap confidence interval for the standard deviation
of each feature.

iv. Use your judgement to select the three most important time-domain features
(one option may be min, mean, and max).


### (d) Binary Classification Using Logistic Regression

i. Assume that you want to use the training set to classify bending from other
activities, i.e. you have a binary classification problem. Depict scatter plots
of the features you specified in part c) extracted from time series 1, 2, and 6 of
each instance, and use color to distinguish bending vs. other activities.

ii. Break each time series in your training set into two (approximately) equal
length time series. Now instead of 6 time series for each of the training
instances, you have 12 time series for each training instance. Repeat the
experiment in part c), i.e depict scatter plots of the features extracted from both
parts of the time series 1,2, and 12. Do you see any considerable difference
in the results with those of part c)?

iii. Break each time series in your training set into l ∈ {1, 2, . . . , 20} time series
of approximately equal length and use logistic regression to solve the binary
classification problem, using time-domain features. Remember that breaking
each of the time series does not change the number of instances. It only
changes the number of features for each instance. Calculate the p-values for
your logistic regression parameters in each model corresponding to each value
of l and refit a logistic regression model using your pruned set of features.
Alternatively, you can use backward selection using sklearn.feature selection
or glm in R. Use 5-fold cross-validation to determine the best value of the pair
(l, p), where p is the number of features used in recursive feature elimination.
Explain what the right way and the wrong way are to perform cross-validation
in this problem. Obviously, use the right way! Also, you may encounter the
problem of class imbalance, which may make some of your folds not having
any instances of the rare class. In such a case, you can use stratified cross
validation. Research what it means and use it if needed.
In the following, you can see an example of applying Python’s Recursive
Feature Elimination, which is a backward selection algorithm, to logistic regression.

```bash
# Recursive Feature Elimination
from sklearn import datasets
from sklearn.featureselection import RFE
from sklearn.linearmodel import LogisticRegression
# load the iris datasets
dataset = dataset.loadiris()
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE( model , 3 )
rfe = rfe.fit( dataset.data , dataset.target)
# summarize the selection of the attributes
print( rfe.support )
print( rfe.ranking)
```

iv. Report the confusion matrix and show the ROC and AUC for your classifier
on train data. Report the parameters of your logistic regression βi’s as well
as the p-values associated with them.

v. Test the classifier on the test set. Remember to break the time series in
your test set into the same number of time series into which you broke your
training set. Remember that the classifier has to be tested using the features
extracted from the test set. Compare the accuracy on the test set with the
cross-validation accuracy you obtained previously.

vi. Do your classes seem to be well-separated to cause instability in calculating
logistic regression parameters?

vii. From the confusion matrices you obtained, do you see imbalanced classes?
If yes, build a logistic regression model based on case-control sampling and
adjust its parameters. Report the confusion matrix, ROC, and AUC of the
model.
