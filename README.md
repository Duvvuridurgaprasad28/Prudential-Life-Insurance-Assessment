# Prudential-Life-Insurance-Assessment
A predictive model for evaluating life insurance applications based on customer data, built using Python and machine learning techniques like decision trees and logistic regression. This project involves building a machine learning model to predict the **Response** variable for life insurance applicants based on a set of features provided by Prudential Life Insurance. The goal is to predict the likelihood of an applicant's response, which is rated on 8 levels. 

### Overview
In this project:
1. Downloaded a real-world dataset from a Kaggle competition.
2. Performed data preprocessing and feature engineering.
3. Trained multiple machine learning models, including Random Forest, Gradient Boosting, Logistic Regression, and XGBoost.
4. Tuned hyperparameters and evaluated the performance of the models.
5. Made predictions on the test dataset and generated a submission file.

### Steps Involved:

#### 1. Dataset Download and Exploration
- We used the Kaggle dataset for life insurance applicants, which contains 59,381 rows and 128 columns.
- The data includes details such as:
  - Product information, employment history, medical history, family history, and more.
  - The target variable is **Response**, which has 8 possible levels.

```python
import opendatasets as od
dataset = 'https://www.kaggle.com/c/prudential-life-insurance-assessment'
od.download(dataset)
```

#### 2. Data Preprocessing and Feature Engineering
- We examined and cleaned the dataset by handling missing values and scaling numeric features.
- Missing values in numerical columns were replaced using the mean strategy.
- Categorical features were encoded using **OneHotEncoder**.
- Numeric features were scaled using **StandardScaler**.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
```

#### 3. Data Splitting
- We split the dataset into **training** and **validation** sets (75% training, 25% validation).
- Input features were separated from the target column **Response**.

```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
```

#### 4. Model Training
We trained four models to predict the **Response**:
- **Random Forest**: A collection of decision trees.
- **Gradient Boosting**: A sequential model-building technique.
- **Logistic Regression**: A statistical model used for classification.
- **XGBoost**: A highly efficient implementation of gradient boosting.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
```

#### 5. Model Evaluation
We evaluated model performance using **accuracy** and **RMSE (Root Mean Squared Error)**.

```python
from sklearn.metrics import accuracy_score, mean_squared_error
```

#### 6. Hyperparameter Tuning
To improve model performance, we tuned hyperparameters using a helper function to test and visualize different values for **n_estimators**, **learning_rate**, and other hyperparameters for XGBoost.

```python
def test_params(**params):
    model = XGBClassifier(random_state=42, **params).fit(X_train, y_train)
    return mean_squared_error(model.predict(X_train), y_train)
```

#### 7. Final Predictions and Submission
After training and evaluating the models, we selected the best-performing model (XGBoost) and made final predictions on the test dataset.

We then generated a **submission file** in the required format.

```python
submission_df['Response'] = test_preds
submission_df.to_csv('submission.csv', index=False)
```

#### 8. Project Results

| Model               | Training RMSE | Validation RMSE | Accuracy Score |
|---------------------|---------------|-----------------|----------------|
| Random Forest       | 0.809         | 0.812           | 0.797          |
| Gradient Boosting   | 0.801         | 0.814           | 0.803          |
| Logistic Regression | 0.816         | 0.822           | 0.786          |
| XGBoost             | 0.801         | 0.815           | 0.805          |

### Requirements:
To run this project locally, you need to install the following libraries:

```bash
pip install opendatasets scikit-learn plotly folium xgboost matplotlib seaborn jupyter
```

### Conclusion:
In this project, we successfully built, trained, and evaluated several machine learning models for predicting life insurance applicants' responses. The **XGBoost** model achieved the best performance on both training and validation datasets. This project provided valuable experience in data preprocessing, feature engineering, model evaluation, and hyperparameter tuning.
