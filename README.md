This repository contains a Customer Churn Prediction application developed using multiple machine learning models.
After training and evaluating different classifiers, Gradient Boosting was selected as the best-performing model based on overall evaluation metrics.

📌 Project Overview

Live App : https://churn-prediction-app-gr-by-hafiz-rayyan-asif.streamlit.app/
Customer churn prediction helps businesses identify customers who are likely to leave so that preventive actions can be taken.
This project applies supervised machine learning techniques to predict churn using historical customer data.

The following models were trained and compared:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Gradient Boosting Classifier

Model Selection

All models were trained using the same dataset and preprocessing pipeline.
After evaluating their performance, Gradient Boosting outperformed the other models and was selected as the final model for prediction.

Reasons for selection:

Higher predictive accuracy

Better balance between precision and recall

Strong performance on unseen test data

Models Trained
Logistic Regression

Used as a baseline model to understand linear relationships.

Decision Tree

Captured non-linear patterns but showed signs of overfitting.

Random Forest Classifier

Improved stability over a single decision tree.

Gradient Boosting Classifier (Final Model)

Provided the best overall performance and generalization.

Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

Jupyter Notebook / Streamlit

Workflow

Data loading and preprocessing

Feature engineering and encoding

Train-test split

Training multiple models

Model evaluation and comparison

Selection of the best model

Final churn prediction using Gradient Boosting
