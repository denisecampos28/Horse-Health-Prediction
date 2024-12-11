![](UTA-DataScience-Logo.png)

# Horse Health Prediction 

* **Quick Summary** This repository holds my completed attempt for Kaggle's Horse Health Prediction Competition. I conducted an EDA and implemented classification models. https://www.kaggle.com/competitions/playground-series-s3e22/overview
## Overview


 * **Definition of the tasks/challenge**:  The task is to predict the health outcome of horses based on features derived from medical observations and tests.
 * **My approach**: The problem was tackled as a classification task using Random Forest and XGBoost models. Comprehensive preprocessing steps ensured data quality and effective model training.
 * **Summary of the performance achieved**: The models achieved competitive performance on the training set. Final predictions were submitted for evaluation on Kaggle.

## Summary of Workdone

### Data

* Data:
  * Type: CSV file with structured medical and observational data.
  * Size: 2999 rows (train.csv) and 824 rows (test.csv).
  * Instances: Data was split into training (80%) and testing (20%) sets.
    

#### Preprocessing / Clean up

* Dropped weak predictors based on domain knowledge and exploratory analysis.

* Imputed missing values using the mode or specific default values like "Unknown."

* One-hot encoded categorical variables.

* Scaled numerical features using MinMaxScaler.

* Mapped target classes ("lived," "died," "euthanized") to integers for model compatibility.
  

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.
*Examined distributions.

*Identified and visualized outliers using boxplots.
![download](https://github.com/user-attachments/assets/69cd0a92-0fae-4420-b355-d5d9eff18de7)
* The target variable was imbalanced, as you can see above
  

### Problem Formulation

 * Input /Output
     * Input: Preprocessed features such as rectal temperature, heart rate, and mucous membrane color.
     * Output: Health outcome prediction ("lived," "died," "euthanized").
 * Models
    * Random Forest Classifier.
    * XGBoost Classifier.
    * I chose these because they both are classification machine learning algorithms.
      
 * Hyperparameters:
    * Grid search was used to optimize parameters like n_estimators and max_depth.
    

### Training

* Training Environment:
  * Google Colab
  * Required Python libraries: scikit-learn, XGBoost, pandas, and joblib.
* Training Time: Less than 3 minutes per model.
* Stopping Criteria: Evaluated models based on F1-score and accuracy.

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* XGBoost performed marginally better than Random Forest in terms of both accuracy and F1-score.
* Both models demonstrated robustness in handling the dataset's class imbalance.

### Future Work

* Investigate feature importance to refine the input feature set further.
* Explore additional hyperparameter tuning to improve model performance.

## How to reproduce results

* hh_cleaning_and_train_test.ipynb: Contains preprocessing steps, model training, and evaluation.
* rf_submission.csv: Predictions from the Random Forest model for Kaggle submission.
* xgb_submission.csv: Predictions from the XGBoost model for Kaggle submission.

### Overview of files in repository

 * utils.py: various functions that are used in cleaning and visualizing data.
 * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
 * visualization.ipynb: Creates various visualizations of the data.
 * models.py: Contains functions that build the various models.
 * training-model-1.ipynb: Trains the first model and saves model during training.
 * training-model-2.ipynb: Trains the second model and saves model during training.
 * training-model-3.ipynb: Trains the third model and saves model during training.
 * performance.ipynb: loads multiple trained models and compares results.
 * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

### Software Setup
* Required packages:
  * pandas
  * scikit-learn
  * xgboost
  * joblib
* Install the packages using pip:
  * pip install pandas scikit-learn xgboost joblib
    
### Data

* Download data from the Kaggle competition page.
* Place train.csv and test.csv in the working directory.

### Training

* Run the Jupyter notebook hh_cleaning_and_train_test.ipynb to reproduce the training pipeline.

* Save the trained models as best_rf_model.pkl and best_xgb_model.pkl for inference.

#### Performance Evaluation

* Evaluate models on the test set predictions using the metrics provided in the notebook.


## Citations

* Kaggle Playground Series S3E22:
  https://www.kaggle.com/competitions/playground-series-s3e22









