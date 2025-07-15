# ✈️ Flight Price Prediction Project

## 📝 Overview
This project focuses on building a machine learning model to accurately predict flight ticket prices based on various features such as:

- Airline
- Source & destination
- Number of stops
- Duration
- Days left for departure

The goal is to demonstrate a complete end-to-end ML workflow—from data loading and preprocessing to model training, evaluation, and hyperparameter tuning—culminating in generating price predictions for unseen data.

---

## 🛠️ Features & Technologies Used

### Programming Language
- **Python**

### Core Libraries
- `pandas` – Data manipulation and analysis  
- `numpy` – Numerical operations  
- `scikit-learn` – ML models, preprocessing, model selection  
- `matplotlib` – Data visualization  
- `seaborn` – Enhanced data visualization  
- `xgboost` – XGBoost Regressor  
- `lightgbm` – LightGBM Regressor  

### Machine Learning Models
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- LightGBM Regressor  
- K-Neighbors Regressor  

### Techniques Used
- Data Cleaning (duplicates, outliers)
- Feature Engineering
- One-Hot Encoding
- Standard Scaling
- Mean/Mode Imputation
- Train-Validation Split
- Cross-Validation
- Hyperparameter Tuning (`RandomizedSearchCV`)

---

## 📂 Dataset

The project utilizes a dataset containing historical flight information and corresponding prices:

- **`train.csv`** – Training data with features and target variable (`price`)  
- **`test.csv`** – Test data with features for price prediction  
- **`sample_submission.csv`** – Submission format sample  

**Key Features Include:**
- `airline`, `source`, `destination`, `stops`, `duration`, `days_left`, `departure_time`, `arrival_time`, `class`, `flight`

---

## 🔁 Project Structure & Workflow

### 1. **Data Loading & Initial Setup**
- Load all CSV files: `train.csv`, `test.csv`, `sample_submission.csv`
- Separate target variable (`price`)
- Identify unique flight IDs

### 2. **Initial Feature Engineering**
- Convert `stops` from categories to numerical values
- Ensure `flight` numbers are treated as strings

### 3. **Data Cleaning**
- **Duplicates:** Remove duplicate rows from training data  
- **Outliers:** Use IQR method for detecting and capping outliers in `duration` and `days_left`

### 4. **Data Preprocessing Pipeline**
- **Missing Values:** Mean imputation for numerics, mode for categoricals  
- **Feature Scaling:** `StandardScaler` for numerical features  
- **Categorical Encoding:** `OneHotEncoder`  
- **Pipeline:** Combine transformations using `ColumnTransformer` and `Pipeline` for consistency

### 5. **Exploratory Data Analysis (EDA)**
- Histograms and box plots for price distribution  
- Correlation analysis using scatter plots  
- Box plots and count plots for categorical feature impact

### 6. **Model Training & Comparison**
- Split data into training and validation sets (80/20)
- Train multiple models:
  - Linear, Tree-based, Boosting, and KNN models
- Evaluate using:
  - **RMSE** (Root Mean Squared Error)
  - **R² Score**
- Present results in a comparison table

### 7. **Hyperparameter Tuning**
- Use `RandomizedSearchCV` for:
  - Random Forest  
  - XGBoost  
  - LightGBM  
- Optimize based on **negative mean squared error**

### 8. **Final Model Training & Prediction**
- Select the best model from tuning results
- Train on full dataset
- Predict on test set
- Ensure predictions are non-negative

### 9. **Submission File Generation**
- Create `submission.csv` with `flight` ID and predicted `price`
- Format as per competition requirements

---

## ✅ Results & Learnings
This project showcases the practical implementation of a full ML pipeline, comparison of multiple regression models, and effective use of hyperparameter tuning. It demonstrates model deployment readiness for real-world pricing predictions.

---

## 📬 Contact
For questions or collaboration, feel free to reach out at: **Kashikag2004@gmail.com**
