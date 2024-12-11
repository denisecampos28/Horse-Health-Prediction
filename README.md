![insert__](UTA-DataScience-Logo.png)

# Horse Health Prediction 

* **Quick Summary** This repository holds my completed attempt for Kaggle's Horse Health Prediction Competition. I conducted an EDA and implemented classification models. https://www.kaggle.com/competitions/playground-series-s3e22/overview
## Overview


 * **Definition of the tasks/challenge**:  The task is to predict the health outcome of horses based on features derived from medical observations and tests.
 * **My approach**: The problem was tackled as a classification task using Random Forest and XGBoost models. Comprehensive preprocessing steps ensured data quality and effective model training.
 * **Summary of the performance achieved**: The models achieved good performance on the training set. Final predictions were submitted for evaluation on Kaggle.

## Summary of Workdone

### Data

* Data:
  * Type: CSV file with structured medical and observational data.
  * Size: 2999 rows (train.csv) and 824 rows (test.csv).
  * Instances: Data was split into training (80%) and testing (20%) sets.
    
#### Preprocessing / Clean up

* Dropped weak predictors based on domain knowledge and exploratory analysis.
* Imputed missing values using the mode or specific default values like "Unknown."
* One hot encoded categorical variables.
* Scaled numerical features using MinMaxScaler.
* Mapped target classes ("lived," "died," "euthanized") to integers for model compatibility.
  
#### Data Visualization

* Examined distributions.
* Identified and visualized outliers using boxplots.
![insert__](https://github.com/user-attachments/assets/21eb414d-528a-4860-bde3-8fcc1bb75168)
* The target variable is imbalanced, as you can see above
  
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

* My main evaluation metric is F1 score. The F1 score is the harmonic mean of a model's precision and recall scores. It's a way to combine these two metrics into a single value that provides a balanced evaluation of a model's performance.
![need](https://github.com/user-attachments/assets/2035a435-852a-40f1-8c15-e90a51eefd7a)
![byclass](https://github.com/user-attachments/assets/e0a0c399-0fdf-4076-af18-abc1db535c49)
![f1avg](https://github.com/user-attachments/assets/2717eb87-ccf4-45c6-a140-f4137d94b3ef)

### Conclusions

* XGBoost performed slightly better than Random Forest in terms of both accuracy and F1-score.
* Both models did ok in handling the dataset's class imbalance.

### Future Work

* Investigate feature importance to refine the input feature set further.
* Explore additional hyperparameter tuning to improve model performance.

## How to reproduce results

* hh_cleaning_and_train_test.ipynb: Contains preprocessing steps, model training, and evaluation.
* rf_submission.csv: Predictions from the Random Forest model for Kaggle submission.
* xgb_submission.csv: Predictions from the XGBoost model for Kaggle submission.

### Overview of files in repository

 *  Kaggle Tabular Data.ipynb: This file contains my project guidelines
 *  train.csv: This file contains a certain portion of the dataset that Kaggle set aside for cleaing and model creation/training.
 *  test.csv: This file contains a certain portion of the dataset that Kaggle set aside for model testing. 
 *  sample_submission.csv: This file is an example of what the sample submission 
 *  hh_loading_and_visualization.ipynb: This file contains my inital look and visulizations of my raw data.
 *  hh_cleaning_and_train_test.ipynb: This file contains my data cleaning, model creation, and evaluation.
 *  hh_research.doxc: This file contains research over the dataset, to better understand my analysis.
 *  rf_submission(1).csv: This file is in the Kaggle's sample_submission format and contains my test results for my test.csv data using my best random forest model.
 *  xgb_submission(1).csv: This file is in the Kaggle's sample_submission format and contains my test results for my test.csv data using my best xgboost model.

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









