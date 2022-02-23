# Supervised, Semi-Supervised, Unsupervised, and Active Learning

## Abstract
This project is the homework 7 for USC DSCI-552 Machine Learning for Data Science.
There are two tasks in this project.
The first task is training a model for Auto-Generating article which mimic Bertrand Russell's writing style.
The second task is training a model to color grayscale bird images.
The Step section explained the process in detail.

## Data
Download the Data Set of Bertrand Russell's 7 books and bird images from 'data' directory.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) or Anaconda to install following packages :

- pandas
- numpy
- seaborn
- sklearn
- warnings
- matplotlib
- pickle
- statistics
- tensorflow
- scipy
- IPython
- Keras
- skimage

The packages can be installed by following command:
```bash
pip install <package>
```

## Execution
The 'main.ipynb' file is in the 'notebook' directory and data are in the 'data' directory.
Please put the 'notebook' directory and 'data' directory in same directory and execute the 'main.ipynb' file.


## Step
### 1. Auto-Generating article

1) Load the texts from 7 books and only keep words in the books.

2) Convert all the words to lower case.

3) Assign an integer to each unique word in the texts. (Word Embedding)

4) For each word, using previous 100 words as features. Training a model in following structure:

LSTM + linear + softmax  with Cross Entropy Loss

5) Make a prediction of next 1000 words give the start with following phrase.
```commandline
'There are those who take mental phenomena naively, just as they would physical phenomena. This school of psychologists tends not to emphasize the object.'
```

### 2. Color Grayscale Images

Due to the limitation of personal laptop computation power, this project only apply 4 colors.

1) Load the images and convert them to RGB vectors.

2) Apply K-means cluster to cluster pixels to the following 4 colors.
navy = RGB(0,0,128)
red = RGB(230,25,75)
mint = RGB(170,255,195)
white = RGB(255,255,255)

3) Convert original images to 4 color images.

4) Training a model in following structure:

Conv2D + MaxPooling2D + Conv2D + MaxPooling2D + softmax  with Cross Entropy Loss

5) Plot train and test error for each epoch

6) Using the model to color images. 
