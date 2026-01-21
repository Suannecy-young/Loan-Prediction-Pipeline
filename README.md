# 🏦 Loan Eligibility Prediction & Fairness Audit Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Focus](https://img.shields.io/badge/Focus-Ethical%20AI-red)

# 📖 Project Overview

This project builds an end-to-end machine learning pipeline to predict loan eligibility.  
Beyond prediction accuracy, the main goal is to **evaluate fairness and statistical reliability**.

Instead of stopping at model performance, I audited the model for **algorithmic bias** using an **Equal Opportunity (TPR parity)** framework and conducted **stability stress tests**.  
These tests revealed how fairness metrics can become misleading when evaluated on **small and sparse datasets**.

---

# 🚀 Key Features

- **End-to-End Pipeline**  
  Covered the full workflow from raw data cleaning to model training and evaluation.

- **Advanced Data Handling**  
  - Used **KNN Imputation** to preserve relationships between features  
  - Applied **SMOTE** to address severe class imbalance

- **Fairness Audit Module**  
  Built a custom module to calculate **True Positive Rate (TPR) parity** across:
  - Gender  
  - Marital Status

- **Stability Analysis**  
  Ran random-seed stress tests and found that an apparent “perfect fairness” score was unstable and driven by limited sample size.

---

# 🛠️ Data Wrangling & Feature Engineering

## 1. Missing Value Imputation

Rather than using simple mean or median filling, I applied a mixed strategy to maintain data quality.

- **Numeric Features**  
  - Used **KNNImputer (k = 5)**  
  - Missing values were filled based on similar samples  
  - This approach preserves multivariate relationships better than average-based methods

- **Categorical Features**  
  - Used **Mode (Most Frequent) imputation**

---

## 2. Handling Skewed Distributions

Exploratory analysis showed that **ApplicantIncome** and **LoanAmount** were heavily right-skewed.

- **Action:** Applied **log transformation**
- **Result:**  
  - Reduced the influence of extreme outliers  
  - Improved model stability and convergence

---

# 📊 Exploratory Data Analysis (EDA)

Before modeling, I performed detailed EDA to understand the data.

## Univariate Analysis
- Used histograms and boxplots to examine feature distributions
- Detected strong outliers in **ApplicantIncome**, motivating log transformation

## Bivariate Analysis
- **Credit History is critical**  
  - Approval rate ≈ **80%** when `Credit_History = 1.0`  
  - Approval rate ≈ **7%** when `Credit_History = 0.0`
- **Income paradox**  
  - High income alone did not guarantee approval  
  - The **Loan Amount-to-Income ratio** mattered more

## Correlation Analysis
- Used correlation matrices to check multicollinearity
- Found strong correlation between **LoanAmount** and **ApplicantIncome**

---

# ⚙️ Model Training & Optimization

## Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)**  
- Prevented the model from favoring the majority class (“Loan Approved”)

## Hyperparameter Tuning
Used **GridSearchCV** to optimize:
- Decision Tree (`max_depth`)
- Random Forest (`n_estimators`)
- Gradient Boosting Machine (GBM)

## Best Model
- **Decision Tree Classifier (max_depth = 5)**
- **Accuracy:** ~72%
- **Design choice:**  
  Prioritized **interpretability and generalization** over aggressive optimization

---

# ⚖️ Fairness Audit: The “Illusion of Equity”

This section is the core contribution of the project.

I evaluated **Equal Opportunity (TPR parity)** across gender groups.

| Stage | Observation |
|------|-------------|
| Initial Audit | Female applicants showed **100% TPR**, suggesting perfect fairness |
| Stress Test | Changing random seeds caused TPR to fluctuate between **50% and 100%** |
| Root Cause | Only **9 qualified female samples** existed in the test set |
| Conclusion | The fairness result was **statistically unreliable** |

### Key Takeaway
> Fairness metrics can be misleading if they are not supported by sufficient sample size and stability testing.  
> Ethical AI evaluation must include **statistical robustness checks**, not just surface-level metrics.

---

## 📂 Project Structure

index.html: [RECOMMENDED] Full interactive report with code, visualizations, and analysis.

Loan_Eligibility_Prediction.ipynb: Original Jupyter Notebook source code.

test_Y3wMUE5_7gLdaTN.csv: Test dataset used for generating predictions.

train_u6lujuX_CVtuZ9i.csv: Training dataset.

---

Xuanci(Charley) Yang
