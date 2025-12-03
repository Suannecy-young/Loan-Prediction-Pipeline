# Loan-Prediction-Pipeline
🏦 Loan Eligibility Prediction & Fairness Audit Pipeline

📖 Project Overview

This project engineers an end-to-end machine learning pipeline to predict loan eligibility. Beyond standard accuracy metrics, the core focus is on Ethical AI and Statistical Robustness.

I audited the model for algorithmic bias using an "Equal Opportunity" framework and conducted stability stress tests to expose the risks of evaluating fairness on small sample sizes.

🚀 Key Features

End-to-End Pipeline: From raw data cleaning (imputation, log transformation) to model deployment.

Handling Imbalance: Implemented SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.

Model Optimization: Utilized GridSearchCV to tune hyperparameters across Decision Trees, Random Forests, and GBMs.

Fairness Audit: Developed a custom module to calculate True Positive Rate (TPR) parity across gender and marital status.

Stability Analysis: Conducted random-seed stress tests, revealing that an initial "100% fairness" metric was a statistical mirage driven by data sparsity ($n=9$).

🛠️ Tech Stack

Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn

Techniques: Cross-Validation, Feature Engineering, Fairness Metrics, Data Funneling

📊 Key Findings

Model Performance: Achieved ~72% accuracy with a robust Decision Tree classifier ($max\_depth=5$).

The "Illusion of Equity":

Initial audits showed a perfect 100% approval rate for qualified female applicants.

Subsequent stability testing revealed this metric fluctuated between 50% and 100% across random splits.

Conclusion: Technical fairness metrics must be backed by rigorous statistical power analysis.

📂 Project Structure

index.html: [RECOMMENDED] Full interactive report with code, visualizations, and analysis.

Loan_Eligibility_Prediction.ipynb: Original Jupyter Notebook source code.

test_Y3wMUE5_7gLdaTN.csv: Test dataset used for generating predictions.

train_u6lujuX_CVtuZ9i.csv: Training dataset.

✍️ Author

Xuanci Yang

UCSD MSDS Applicant
