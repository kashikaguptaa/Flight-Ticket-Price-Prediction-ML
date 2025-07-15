Overview
This project focuses on building a machine learning model to accurately predict flight ticket prices based on various features such as airline, source, destination, number of stops, duration, and days left for departure. The goal is to demonstrate a complete end-to-end machine learning workflow, from data loading and preprocessing to model training, evaluation, and hyperparameter tuning, culminating in generating price predictions for unseen data.

Features & Technologies Used
Programming Language: Python

Core Libraries:

pandas (for data manipulation and analysis)

numpy (for numerical operations)

scikit-learn (for machine learning models, preprocessing, model selection)

matplotlib (for data visualization)

seaborn (for enhanced data visualization)

xgboost (for XGBoost Regressor)

lightgbm (for LightGBM Regressor)

Machine Learning Models: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, K-Neighbors Regressor.

Techniques: Data Cleaning (Duplicates, Outliers), Feature Engineering, One-Hot Encoding, Standard Scaling, Mean/Mode Imputation, Train-Validation Split, Cross-Validation, Hyperparameter Tuning (RandomizedSearchCV).

Dataset
The project utilizes a dataset containing historical flight information and their corresponding prices. It comprises:

train.csv: Training data with features and the target variable (price).

test.csv: Test data with features for which prices need to be predicted.

sample_submission.csv: A sample file illustrating the required submission format.

Key features include: airline, source, destination, stops, duration, days_left, departure_time, arrival_time, class, and flight number.

Project Structure & Workflow
The project follows a standard machine learning pipeline:

Data Loading & Initial Setup:

Loading train.csv, test.csv, and sample_submission.csv.

Separating target variable (price) and identifying unique flight IDs.

Initial Feature Engineering:

Converting categorical 'stops' feature (zero, one, two_or_more) into numerical values.

Ensuring 'flight' numbers are treated as strings.

Data Cleaning:

Duplicate Handling: Identifying and removing duplicate rows in the training data to ensure unique observations.

Outlier Handling: Detecting outliers in numerical features (duration, days_left) using the Interquartile Range (IQR) method and applying capping to mitigate their influence.

Data Preprocessing Pipeline:

Missing Value Imputation: Using mean for numerical features and most frequent for categorical features.

Feature Scaling: Applying StandardScaler to numerical features.

Categorical Encoding: Using OneHotEncoder for categorical features.

Assembling these steps into a ColumnTransformer and Pipeline for consistent transformations on both training and test data.

Exploratory Data Analysis (EDA):

Analyzing the distribution of the target variable (price) using histograms and box plots.

Investigating relationships between numerical features (e.g., duration, days_left) and price via scatter plots and correlation.

Exploring the impact of categorical features (e.g., airline, source, class) on price using count plots and box plots.

Model Training & Comparison:

Splitting the processed training data into training and validation sets (80/20 split) to evaluate generalization.

Training multiple diverse regression models, including Linear models, Tree-based models (Decision Tree, Random Forest), and Boosting models (Gradient Boosting, XGBoost, LightGBM), and K-Neighbors Regressor.

Evaluating each model's performance on the validation set using RMSE and R2 score.

Presenting a comparative table of model results.

Hyperparameter Tuning:

Performing hyperparameter optimization using RandomizedSearchCV on three top-performing ensemble models (Random Forest, LightGBM, XGBoost).

Searching for optimal parameter combinations to maximize model performance (measured by negative mean squared error).

Final Model Training & Prediction:

Selecting the best model configuration based on tuning results.

Training the chosen final_model on the entire processed training dataset.

Generating predictions on the unseen X_test_processed data.

Ensuring all predictions are non-negative.

Submission File Generation:

Creating a submission.csv file with flight IDs and the predicted prices, formatted for competition submission.
