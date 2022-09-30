
**<h1 align=center><font size = 5>Predicting Kepler Exoplanet Search Results</font></h1>**

<img  src="https://dynaimage.cdn.cnn.com/cnn/c_fill,g_auto,w_1200,h_675,ar_16:9/https%3A%2F%2Fcdn.cnn.com%2Fcnnnext%2Fdam%2Fassets%2F190927113859-gas-giant-exoplanet-dwarf-star.jpg"  height=500  width=1000  alt="CNN Exoplanet Picture">

<small>Picture Source: <a  href="https://dynaimage.cdn.cnn.com/cnn/c_fill,g_auto,w_1200,h_675,ar_16:9/https%3A%2F%2Fcdn.cnn.com%2Fcnnnext%2Fdam%2Fassets%2F190927113859-gas-giant-exoplanet-dwarf-star.jpg">dynaimage.cdn.cnn.com</a>

<br>

<h2>Context</h2>

<p>10000 exoplanet candidates examined by the <i>Kepler Space Observatory</i>. The <i>Kepler Space Observatory</i> is a NASA-build satellite that was launched in 2009. The telescope is dedicated to searching for exoplanets in star systems besides our own, with the ultimate goal of possibly finding other habitable planets besides our own. The original mission ended in 2013 due to mechanical failures, but the telescope has nevertheless been functional since 2014 on a <i>K2</i> extended mission.

<i>Kepler</i> had verified 1284 new exoplanets as of May 2016. As of October 2017 there are over 3000 confirmed exoplanets total (using all detection methods, including ground-based ones). The telescope is still active and continues to collect new data on its extended mission.</p>  

<br>

<h2>Acknowledgements</h2>

 <p>This dataset was published as-is by NASA. You can access the original table <a  href="https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi"  target="_blanck">here</a>. More data from the Kepler mission is available from the same source <a  href="https://exoplanetarchive.ipac.caltech.edu/docs/data.html"  target="_blanck">here</a>.</p>

<br>

<h2>Sources and Dataset</h2>

<p>You can find the dataset on <a  href="https://www.kaggle.com/nasa/kepler-exoplanet-search-results"  target="_blanck">Kaggle NASA</a> website. This dataset is a cumulative record of all observed Kepler <i>objects of interest</i> — basically, all of the approximately 10,000 exoplanet candidates Kepler has taken observations on.

  

This dataset has an extensive data dictionary, which can be accessed here. Highlightable columns of note are:</p>

<ul>
	<li><code>kepoi_name</code>: A KOI is a target identified by the Kepler Project that displays at least one transit-like sequence within Kepler time-series photometry that appears to be of astrophysical origin and initially consistent with a planetary transit hypothesis</li>
	<li><code>kepler_name</code>: These names are intended to clearly indicate a class of objects that have been confirmed or validated as planets—a step up from the planet candidate designation.</li>
	<li><code>koi_disposition:</code> The disposition in the literature towards this exoplanet candidate. One of CANDIDATE, FALSE POSITIVE, NOT DISPOSITIONED or CONFIRMED.</li>
	<li><code>koi_pdisposition</code>: The disposition Kepler data analysis has towards this exoplanet candidate. One of FALSE POSITIVE, NOT DISPOSITIONED, and CANDIDATE.</li>
	<li><code>koi_score:</code> A value between 0 and 1 that indicates the confidence in the KOI disposition. For CANDIDATEs, a higher value indicates more confidence in its disposition, while for FALSE POSITIVEs, a higher value indicates less confidence in that disposition.</li>
</ul>

<br>

<b>License</b>
CC0: Public Domain

<br>

<h2>Keywords</h2>
<ul>
	<li>Earth and Nature</li>
	<li>Science and Technology</li>
	<li>Astronomy</li>
	<li>Business</li>
	<li>Classification</li>
</ul>

<br>

<h2> Objective for this Notebook</h2>

<p>The purpose of this <a  href="https://github.com/doguilmak/Predict-Kepler-Exoplanet-Search-Results/blob/main/kepler_exoplanet_koi_score.ipynb" target="_blanck">study</a> is based on the available data, we had tried to predict <code>koi_score</code> based on independent variables.  In addition, The purpose of this <a  href="https://github.com/doguilmak/Predict-Kepler-Exoplanet-Search-Results/blob/main/kepler_expo_ANN.py" target="_blanck">study</a> is based on the available data, it was estimated whether disposition (<code>koi_disposition</code>) are CONFIRMED, FALSE POSITIVE or CANDIDATE.</p>

<br>

<h2>Methodology</h2>

<p>In this project, as stated in the title, results were obtained through <i>Multiple linear regression</i>,  <i>XGBoost</i> and <i>Artificial neural networks</i>. </p>

<p align="center">
    <img src="Plots/heatmap.png"> 
</p>

<br>

<h3>XGBoost</h3>

<p>XGBoost or extreme gradient boosting is one of the well-known gradient boosting techniques(ensemble) having enhanced performance and speed in <b>tree-based</b> (sequential decision trees) machine learning algorithms.</p>

<p align="center">
    <img src="Plots/XGBoost_Tree.png"> 
</p>

Confusion Matrix (XGBoost):

$$
\begin{matrix}  
598 & 115 & 16\\ 
149 & 613 & 2\\ 
8 & 15 & 1641  
\end{matrix}
$$

<b>Prediction of XGBoost:</b>
Model predicted as CANDIDATE.
Model predicted class as [0].

<b>Accuracy score (XGBoost): 0.9033892936331961</b>

Process took 5.090279579162598 seconds.

<br>

<h3>Artificial Neural Networks</h3>

| Validaton Accuracy | Validation Loss |
|--|--|
| ![val_acc](Plots/model_acc.png) | ![val_loss](Plots/model_loss.png) |

<b>Prediction of Artificial Neural Networks:</b>

Model predicted as CONFIRMED.

Mean of validation accuracy: 0.8379785418510437

Process took 14.049815893173218 seconds.

<br>

</h2>How to Run Code</h2>

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
    
<h2>Contact Me</h2>

If you have something to say to me please contact me: 

 - Twitter: [Doguilmak](https://twitter.com/Doguilmak) 
 - Mail address: doguilmak@gmail.com
 
