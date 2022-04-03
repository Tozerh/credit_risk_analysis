# Credit Risk Analysis
Module 17 Challenge - Credit Risk Analysis with Supervised Machine Learning

*Project Overview*:

For this project, we were asked to use supervised machine learning to evaluate how much of a credit risk a given applicant might be. As a part of our results, we are reporting on which machine learning model works best for this particular project. 

*Resources*:

Code and Software: Python, Numpy, Jupyter Notebook, and Anaconda

## *Results*

The results from all six models are found below, with visualisations of the confusion matrices and a presenation of the balanced accuracy scores. 

### Naive Random Oversampling

Confusion Matrix: 

![NROS](https://github.com/Tozerh/credit_risk_analysis/blob/main/RandomOvrSmpl.png)

Balanced Accuracy Score: 0.6388002

### SMOTE Oversampling

Confusion Matrix: 

![SMOTE](https://github.com/Tozerh/credit_risk_analysis/blob/main/SMOTE.png)

Balanced Accuracy Score: 0.6374415

### Cluster Centroids Undersampling

Confusion Matrix: 

![ClusterCentroids](https://github.com/Tozerh/credit_risk_analysis/blob/main/ClusterCentroid.png)

Balanced Accuracy Score: 0.5292151

### SMOTEENNN Over- and Undersampling

Confusion Matrix: 

![SMOTEENN](https://github.com/Tozerh/credit_risk_analysis/blob/main/SMOTEEN.png)

Balanced Accuracy Score: 0.6244385

### Balanced Random Forest Classifier Ensemble Learning

Confusion Matrix: 

![bRandomForest](https://github.com/Tozerh/credit_risk_analysis/blob/main/En-BalancedRF.png)

Balanced Accuracy Score: 0.81928480

### Easy Ensemble AdaBoost Classifier Ensemble Learning

Confusion Matrix: 

![AdaBoost](https://github.com/Tozerh/credit_risk_analysis/blob/main/En-AdaBoost.png)

Balanced Accuracy Score: 0.9254274

## Summary

My recommendation to my client here would be to use one of the ensemble classifiers, which should produce a much better result than any of the individual resampling models that we deployed. Given the inherent imbalance to the problem of determining credit risk, employing a sampling method to decrease the variance in prediction is a must, and both Random Forest and AdaBoost allow us to be sure that we remove as much variance as possible from our modeling. Our ensemble classifiers grossly outperform any other models, and I would give AdaBoost the edge due to its slightly higher Balanced Accuracy Score. AdaBoost also has a ~30% lower amount of False negatives, which should help to mitigate any customer complaints and lost business due to incorrectly flagging a low-risk client as high-risk. 

