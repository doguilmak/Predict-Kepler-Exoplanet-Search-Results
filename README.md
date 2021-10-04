
# Predicting Disposition from Kepler Exoplanet Search Results with Using XGBoost and Artificial Neural Networks 

## Problem Statement

The purpose of this study is based on the available data, it was estimated whether **disposition (koi_disposition)** are ***CONFIRMED***, ***FALSE POSITIVE*** or ***CANDIDATE***. 

## Dataset

Datasets are downloaded from [Kaggle NASA](https://www.kaggle.com/nasa/kepler-exoplanet-search-results). You can find the details of the datasets in that website. **cumulative.csv** dataset has ***50*** columns and ***9565*** rows with the header.

## Methodology

In this project, as stated in the title, results were obtained through **XGBoost** and **artificial neural networks (ANN)** methods. 

## Analysis

Please run the ***console_kepler_output.html*** HTML file to examine more statistical data about the dataset and see the output on the console. You can run the HTML file in your browser by selecting which browser you are using in the ***RUN*** tab of **Notepad++**. File has output of ***kepler_expo_ANN.py*** and ***kepler_expo_XGBoost.py***.

**Number of NaN:**

> **Number of NaN values:**   13813  
> 
> After inplace mean of the columns 
> **Number of NaN values:**   0

 **Number of Duplicated Rows:**

> **0 duplicated rows.**

The estimated data was made on the first row in the data set. Actual result is **CONFIRMED**.

### XGBoost

Confusion Matrix(XGBoost):

| 598 | 115 | 16 |
|--|--|--|
| **149** | **613** | **2** |
| **8** | **15** | **1641** |

**Prediction of XGBoost:**
Model predicted as FALSE POSITIVE.
Model predicted class as [0].

> **Accuracy score(XGBoost): 0.9033892936331961**

**Process took 19.06718349456787 seconds.**

### Artificial Neural Network

**Prediction of ANNs:**
Model predicted as CONFIRMED.
Model predicted class as [1].

> **Mean of validation accuracy: 0.8379785418510437**

**Process took 32.26679515838623 seconds.**

In ***Plot*** folder, you can find ***ANN_Model_Test_Train.png***  which is showing plot of test and train accuracy. Accuracy values and also plot can change a bit after you run the algorithm. In that folder you can also find heatmap of dataset (***heatmap.png***) and XGBoost Tree (***XGBoost_Tree.png***).

## How to Run Code

Before running the code make sure that you have these libraries:

 - pandas 
 - time
 - sklearn
 - seaborn
 - numpy
 - warnings
 - xgboost
 - matplotlib
 - keras
 - statistics
    
## Contact Me

If you have something to say to me please contact me: 

 - Twitter: [Doguilmak](https://twitter.com/Doguilmak).  
 - Mail address: doguilmak@gmail.com
 
