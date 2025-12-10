# DS.v2.5.3.4.1


# Home Credit Default Risk Modeling

## 📌 Project Overview

This project focuses on developing, deploying, and serving a machine learning model for **credit risk prediction** using the Home Credit dataset. The goal is to build an **interpretable**, **deployable**, and **financially sound** model that effectively identifies potential loan defaulters.

The workflow is modular and structured across multiple notebooks, covering everything from raw data exploration to feature engineering and model training.

---

## Live Application

The application is deployed and accessible at:

[<https://home-credit-image-909100323557.europe-west1.run.app/>](https://home-credit-image-909100323557.europe-west1.run.app/)

REST API :
[text](https://home-credit-image-909100323557.europe-west1.run.app/predict)


## Primary Datasets

| Filename                        | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `application_train.csv`        | Main training dataset with customer features and target variable (`TARGET`) |
| `application_test.csv`         | Test dataset with same features as train, but without `TARGET`              |
| `bureau.csv`                   | Credit history from other financial institutions                            |
| `bureau_balance.csv`           | Monthly balance snapshots for each bureau record                            |
| `previous_application.csv`     | Historical loan applications submitted by customers                         |
| `POS_CASH_balance.csv`         | Point-of-sale and cash loan monthly balances                                |
| `installments_payments.csv`    | Customer payment history for previous loans                                 |
| `credit_card_balance.csv`      | Monthly credit card balances and usage                                      |

---

## Notebooks Structure

###  Exploratory Data Analysis (EDA)
- `application_eda.ipynb`: Analysis of core applicant data and target variable
- `bureau_eda.ipynb`: Insights from external credit history and bureau records
- `previous_application_eda.ipynb`: Patterns and signals from past loan applications

### Feature Engineering & Aggregation
- `feature_engineering.ipynb`: Creation of derived features, aggregation across datasets, and preparation of the final training set

###  Modeling
- `modeling.ipynb`: Training, evaluation, and tuning of machine learning models for credit risk classification

---

##  Key Findings & Modeling Insights

### EDA Highlights
- Removed features with **low variance**, **low correlation**, and **high missingness** to reduce noise and improve model performance.
- Detected **anomalies** in applicant data, which were cleaned during feature engineering.

### Feature Engineering & Aggregation
- Created a comprehensive feature set of **580 engineered features** across all datasets.
- Aggregated historical and behavioral data to enrich the training set.

### Modeling Strategy
- Used **LightGBM** to identify and rank feature importance.
- Trained models using both **XGBoost** and **LightGBM**, then built an **ensemble model** that outperformed individual models.
- Selected the **top 198 most important features** and retrained the final model for efficiency and interpretability.
- Extracted **43 raw features** from `application_train.csv` to serve as inputs for the **API**, enabling lightweight and real-time predictions.



## 📊 Results

### Ensemble Model Performance Summary  
**VotingClassifier (LightGBM + XGBoost, Soft Voting)**

- **Kaggle Competition Score:** 77.20%
- **Test Set Metrics:**
  - ROC AUC: **0.785**
  - Precision (Class 1): **47.00%**
  - Recall (Class 1): **11.00%**
  - F1-Score (Class 1): **18.00%**
  - PR AUC: **0.28**

### Strengths
- High overall accuracy driven by strong performance on the majority class (Class 0)
- ROC AUC of 0.785 suggests good class separation
- Ensemble benefits from model diversity and complementary strengths

### Limitations
- Low recall and F1-score for minority class (Class 1), indicating under-detection
- PR AUC of 0.28 highlights difficulty in balancing precision and recall


## Feature Importance

The ensemble model leverages both **XGBoost** and **LightGBM**, and the following features emerged as the most influential based on their average importance scores:

| Feature                   | XGBoost Importance | LightGBM Importance | Mean Importance |
|---------------------------|--------------------|----------------------|------------------|
| CREDIT_DIV_ANNUITY        | 225                | 0.007036             | 112.50           |
| EXT_SOURCE_MEAN           | 195                | 0.091210             | 97.55            |
| DAYS_BIRTH                | 179                | 0.005210             | 89.50            |
| PREV_CNT_PAYMENT_MEAN     | 141                | 0.005952             | 70.50            |
| DAYS_PAYMENT_RATIO_MAX    | 130                | 0.011108             | 65.01            |
| EXT_SOURCE_1_BIRTH_RATIO  | 119                | 0.008139             | 59.50            |
| EXT_SOURCE_3_BIRTH_RATIO  | 98                 | 0.006478             | 49.00            |
| EXT_SOURCE_2              | 97                 | 0.010327             | 48.51            |
| DAYS_BEFORE_DUE_SUM       | 91                 | 0.006874             | 45.50            |
| DAYS_REGISTRATION         | 89                 | 0.003324             | 44.50            |

### 🧠 Insights
- **CREDIT_DIV_ANNUITY** and **EXT_SOURCE_MEAN** are the most dominant features, suggesting strong predictive power in financial ratios and external credit scores.
- **DAYS_BIRTH** consistently ranks high, indicating age-related patterns in credit behavior.
- Engineered ratios like **EXT_SOURCE_1_BIRTH_RATIO** and **DAYS_PAYMENT_RATIO_MAX** contribute meaningfully, validating the value of feature engineering.

   
## Installation Guide

**Note:** This project uses Python 3.11 as specified in the `.python-version` file.

### Using uv (Recommended)

1. **Install uv:**

   ```bash
   # On Unix/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   ```

2. **Clone the Repository:**

   ```bash
   git clone https://github.com/TuringCollegeSubmissions/anplien-DS.v2.5.3.4.1.git
   ```

3. **Create and Activate a Virtual Environment:**

   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

4. **Install Dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```
5. Build Docker Image

```bash
   docker build -t home_credit-risk-api .
 ```

6 Run API
  - Local
     
    uvicorn app.main:app --reload
  - Docker
    docker run -p 8000:8000 home_credit-risk-api


## Running the Application

    
```bash
  - Local

    uvicorn app.main:app --reload
 
  - Docker
  
        docker run -p 8000:8000 home_credit-risk-api

```


