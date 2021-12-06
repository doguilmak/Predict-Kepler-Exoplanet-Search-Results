
# Predicting Disposition from Kepler Exoplanet Search Results with Using XGBoost and Artificial Neural Networks 

## Problem Statement

The purpose of this study is based on the available data, it was estimated whether **disposition (koi_disposition)** are ***CONFIRMED***, ***FALSE POSITIVE*** or ***CANDIDATE***. 

## Dataset

Datasets are downloaded from [Kaggle NASA](https://www.kaggle.com/nasa/kepler-exoplanet-search-results). You can find the details of the datasets in that website. **cumulative.csv** dataset has ***50*** columns and ***9565*** rows with the header.

| Column | Details |
|--|--|
| kepoi_name  | A KOI is a target identified by the Kepler Project that displays at least one transit-like sequence within Kepler time-series photometry that appears to be of astrophysical origin and initially consistent with a planetary transit hypothesis |
| kepler_name | [These names] are intended to clearly indicate a class of objects that have been confirmed or validated as planets—a step up from the planet candidate designation. |
| koi_disposition | The disposition in the literature towards this exoplanet candidate. One of CANDIDATE, FALSE POSITIVE, NOT DISPOSITIONED or CONFIRMED. |
| koi_pdisposition | The disposition Kepler data analysis has towards this exoplanet candidate. One of FALSE POSITIVE, NOT DISPOSITIONED, and CANDIDATE. |
| koi_score | A value between 0 and 1 that indicates the confidence in the KOI disposition. For CANDIDATEs, a higher value indicates more confidence in its disposition, while for FALSE POSITIVEs, a higher value indicates less confidence in that disposition. |

### Acknowledgements

This dataset was published as-is by **NASA**. You can access the original table  [here](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi). More data from the Kepler mission is available from the same source  [here](https://exoplanetarchive.ipac.caltech.edu/docs/data.html).


## Methodology

In this project, as stated in the title, results were obtained through **XGBoost** and **artificial neural networks (ANN)** methods. 

## Analysis

Please run the ***console_kepler_output.html*** HTML file to examine more statistical data about the dataset and see the output on the console. You can run the HTML file in your browser by selecting which browser you are using in the ***RUN*** tab of **Notepad++**. File has output of ***kepler_expo_ANN.py*** and ***kepler_expo_XGBoost.py***.

<p align="center">
    <img width="400" height="600" src="Plots/heatmap.png"> 
</p>

---

**Number of NaN**

**Number of NaN values:**   13813  

After inplace mean of the columns 

**Number of NaN values:**   0

---

 **Number of Duplicated Rows:**

**0 duplicated rows.**

The estimated data was made on the first row in the data set. Actual result is **CONFIRMED**.

---

### XGBoost

XGBoost or extreme gradient boosting is one of the well-known gradient boosting techniques(ensemble) having enhanced performance and speed in **tree-based** (sequential decision trees) machine learning algorithms.

<p align="center">
    <img src="Plots/XGBoost_Tree.png"> 
</p>

Confusion Matrix(XGBoost):

| 598 | 115 | 16 |
|--|--|--|
| **149** | **613** | **2** |
| **8** | **15** | **1641** |

**Prediction of XGBoost:**
Model predicted as CANDIDATE.
Model predicted class as [0].

**Accuracy score(XGBoost): 0.9033892936331961**

Process took 5.090279579162598 seconds.

---

### Artificial Neural Network

**Prediction of ANNs:**
Model predicted as CONFIRMED.

**Mean of validation accuracy: 0.8379785418510437**

| Validaton Accuracy | Validation Loss |
|--|--|
| ![val_acc](Plots/model_acc.png) | ![val_loss](Plots/model_loss.png) |

Process took 14.049815893173218 seconds.

In ***Plots*** folder, you can find ***ANN_Model_Test_Train.png***  which is showing plot of test and train accuracy. Accuracy values and also plot can change a bit after you run the algorithm. In that folder you can also find heatmap of dataset (***heatmap.png***) and XGBoost Tree (***XGBoost_Tree.png***).

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

 - Twitter: [Doguilmak](https://twitter.com/Doguilmak) 
 - Mail address: doguilmak@gmail.com
 
