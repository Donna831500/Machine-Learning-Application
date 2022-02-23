# Predicting the Hourly Output of a Power Plant

## Abstract
This project is the homework 2 for USC DSCI-552 Machine Learning for Data Science.
The Step section explained the process in detail.
The dataset contains data points collected from a Combined Cycle Power Plant over
6 years (2006-2011), when the power plant was set to work with full load. Features
consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP),
Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical
energy output (EP) of the plant.


## Data
Download the Combined Cycle Power Plant data from: https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

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
### (a) Download the Combined Cycle Power Plant data from:
https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant

### (b) Exploring the data:

i. How many rows are in this data set? How many columns? What do the rows
and columns represent?

ii. Make pairwise scatterplots of all the varianbles in the data set including the
predictors (independent variables) with the dependent variable. Describe
your findings.

iii. What are the mean, the median, range, first and third quartiles, and interquartile ranges of each of the variables in the dataset? Summarize them
in a table.

### (c)
For each predictor, fit a simple linear regression model to predict the response.
Describe your results. In which of the models is there a statistically significant
association between the predictor and the response? Create some plots to back
up your assertions. Are there any outliers that you would like to remove from
your data for each of these regression tasks?


### (d)
Fit a multiple regression model to predict the response using all of the predictors.
Describe your results. For which predictors can we reject the null hypothesis
H0 : βj = 0?

### (e)
How do your results from part c) compare to your results from part d)? Create a plot
displaying the univariate regression coefficients from part c) on the x-axis, and the
multiple regression coefficients from part d) on the y-axis. That is, each predictor is
displayed as a single point in the plot. Its coefficient in a simple linear regression
model is shown on the x-axis, and its coefficient estimate in the multiple linear
regression model is shown on the y-axis.

### (f)
Is there evidence of nonlinear association between any of the predictors and the
response? To answer this question, for each predictor X, fit a model of the form

Y = β0 + β1X + β2X^2 + β3X^3 + epsilon

### (g)
Is there evidence of association of interactions of predictors with the response? To
answer this question, run a full linear regression model with all pairwise interaction
terms and state whether any interaction terms are statistically significant.

### (h)
Can you improve your model using possible interaction terms or nonlinear associations between the predictors and response? Train the regression model on a randomly selected 70% subset of the data with all predictors. Also, run a regression model involving all possible interaction terms and quadratic nonlinearities,
and remove insignificant variables using p-values (be careful about interaction
terms). Test both models on the remaining points and report your train and test
MSEs.


### (i)
KNN Regression:

i. Perform k-nearest neighbor regression for this dataset using both normalized
and raw features. Find the value of k ∈ {1, 2, . . . , 100} that gives you the
best fit. Plot the train and test errors in terms of 1/k.

### (g)
Compare the results of KNN Regression with the linear regression model that has
the smallest test error and provide your analysis.
