# Binary Classification with Neuron Network
This project performs various machine learning tasks on a dataset with 19020 samples and 11 attributes. Demonstrates the application of various machine learning algorithms to a dataset, allowing the comparison of their performance using classification reports. Additionally, it conducts a hyperparameter search for a neural network model.
The project including:
  - Importing Libaries
  - Dataset Loading and Preprocessing
  - Split dataset
  - Data Scaling and Oversampling
  - k-Nearest Neighbour (kNN) Classifier
  - Naive Bayes Classifier
  - Logistic Regression Classifier
  - Support Vector Machine (SVM) Classifier
  - Neurol Network (Deep Learning) Classifier
  

## Dataset:
Data are MC generated to simulate registration of high energy gamma particles in an atmospheric Cherenkov telescope
**Reference**
Bock,R.. (2007). MAGIC Gamma Telescope. UCI Machine Learning Repository. https://doi.org/10.24432/C52C8B.

## Importing Libaries
- Necessary libraries and its purpose for this project:
  - `numpy`: data manipulation and array operation.
  - `pandas`: load and manipulate dataset, as well as for data preprocessing and analysis.
  - `matplotlib`: `matplotlib.pyplot`, create histograms for data visualization.
  - `sklearn.preprocessing`: `StandardScaler` from this library is used for feature scaling, which standardizes the data to have a mean of 0 and a standard deviation of 1.
  - `imblearn.over_sampling`: `RandomOverSampler` is used to oversample the minority class to balance the class distribution.
  - `sklearn.neighbors`: `KNeighborsClassifier` to create and evaluate a kNN model.
  - `sklearn.naive_bayes`: `GaussianNB` to create and evaluate a Naive Bayes model.
  - `sklearn.linear_model`: `LogisticRegression` to create and evaluate a logistic regression model.
  - `sklearn.svm`: `SVC` to create and evaluate an SVM model
  - `tensorflow`: used to create and train a neural network for classification
 

