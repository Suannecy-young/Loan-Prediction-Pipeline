# 🏦 Loan Eligibility Prediction & Fairness Audit Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Focus](https://img.shields.io/badge/Focus-Ethical%20AI-red)

## 📖 Project Overview

This project engineers an end-to-end machine learning pipeline to predict loan eligibility. Unlike standard classification tasks, the core focus of this repository is on **Ethical AI** and **Statistical Robustness**.

I audited the model for algorithmic bias using an **"Equal Opportunity"** framework (TPR Parity) and conducted **stability stress tests** to expose the risks of evaluating fairness on small, sparse datasets.

---

## 🚀 Key Features

* **End-to-End Pipeline**: From raw data cleaning (KNN imputation, log transformation) to model deployment.
* **Advanced Data Handling**: Implemented **KNN Imputer** for preserving data structure and **SMOTE** to address class imbalance.
* **Fairness Audit Module**: Developed a custom module to calculate True Positive Rate (TPR) parity across gender and marital status.
* **Stability Analysis**: Conducted random-seed stress tests, revealing that an initial "100% fairness" metric was a statistical mirage driven by data sparsity ($n=9$).

---

## 🛠️ Data Wrangling & Feature Engineering

### 1. Advanced Missing Value Imputation
Instead of simple mean/median filling, I employed a hybrid strategy to maximize data integrity.
* **Numeric Columns**: Used **KNNImputer** ($k=5$). This fills missing values based on the similarity of other features, preserving multivariate correlations better than mean imputation.
* **Categorical Columns**: Used Mode (Most Frequent) imputation.

### 2. Handling Skewness (EDA Driven)
Through Exploratory Data Analysis, I identified that `ApplicantIncome` and `LoanAmount` were highly right-skewed.
* **Action:** Applied **Log Transformation** to normalize distributions.
* **Result:** Reduced the impact of extreme outliers and improved model convergence.

---

## 📊 Exploratory Data Analysis (EDA)
Before modeling, extensive EDA was conducted to understand feature relationships:

* **Univariate Analysis:**
    * Visualized distributions using histograms and boxplots.
    * Detected significant outliers in `ApplicantIncome`, necessitating the log transformation mentioned above.
* **Bivariate Analysis:**
    * **Credit History is King:** Applicants with a Credit History of `1.0` showed an approval rate **~80%**, vs. **~7%** for those without.
    * **Income Paradox:** Surprisingly, high income alone did not guarantee approval; the **Loan Amount to Income ratio** played a more critical role.
* **Correlation Matrix:** Used Seaborn heatmaps to check for multicollinearity (e.g., `LoanAmount` correlates strongly with `ApplicantIncome`).

---

## ⚙️ Model Optimization

* **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model doesn't bias towards the majority class ("Loan Approved").
* **Hyperparameter Tuning:** Utilized `GridSearchCV` to optimize parameters for:
    * Decision Trees (`max_depth`)
    * Random Forests (`n_estimators`)
    * Gradient Boosting Machines (GBM)

**Best Model Performance:**
> **Decision Tree Classifier** ($max\_depth=5$)
> * **Accuracy:** ~72%
> * **Focus:** Prioritized interpretability and generalization over overfitting.

---

## ⚖️ Fairness Audit: The "Illusion of Equity"
This is the critical differentiator of this project. I tested the model for **Equal Opportunity** (True Positive Rate Parity) across Genders.

| Metric | Observation |
| :--- | :--- |
| **Initial Audit** | Showed a **100% approval rate** (TPR) for qualified female applicants. |
| **Stress Test** | Random seed variation caused this metric to fluctuate wildly between **50% and 100%**. |
| **Conclusion** | The "Fairness" was a statistical mirage driven by **data sparsity** ($n=9$ qualified females in test set). |

**Takeaway:** Technical fairness metrics must be backed by rigorous statistical power analysis.

---

## 💻 Tech Stack
* **Language:** Python 3.8+
* **Libraries:**
    * `Pandas`, `NumPy` (Data Manipulation)
    * `Scikit-learn` (Modeling, KNN Imputer, GridSearch)
    * `Imbalanced-learn` (SMOTE)
    * `Matplotlib`, `Seaborn` (Visualization)

**Project Structure**

index.html: [RECOMMENDED] Full interactive report with code, visualizations, and analysis.

Loan_Eligibility_Prediction.ipynb: Original Jupyter Notebook source code.

test_Y3wMUE5_7gLdaTN.csv: Test dataset used for generating predictions.

train_u6lujuX_CVtuZ9i.csv: Training dataset.



Xuanci(Charley) Yang
