---
title: Credit-Scoring-Analysis
sdk: gradio
app_file: app.py
---

## **Credit Scoring Prediction App**
Predict whether a customer has *Good Credit* or *Bad Credit* using a machine-learning model trained in Python and deployed with *Gradio* and *Hugging Face Spaces*.


Live Demo: https://huggingface.co/spaces/sathwik061/credit-scoring
**Objective**
Financial institutions need to estimate the risk of giving loans.
This project builds a credit scoring model that predicts creditworthiness based on customer financial and personal attributes.

**Project Overview**
  - Data preprocessing & encoding
  - Machine learning model (Logistic Regression)
  - One-hot encoded categorical features
  - Saved model with joblib
  - Interactive web app UI (Gradio)
  - Deployment to Hugging Face Spaces
**Model**
*Algorithm*: Logistic Regression
*Reason*: simple, interpretable, commonly used in credit scoring.

The model outputs:
  - Good Credit
  - Bad Credit

A probability score is used behind the scenes to make the decision.

**Features Used**
  - Checking account status
  - Credit history
  - Loan amount
  - Duration
  - Employment history
  - Savings
  - Age
  - Housing
  - Job type
  - Number of existing loans
(Feature values are encoded to match how the model was trained.)

**Application Features**
  - Simple dropdown-based inputs
  - Real-time predictions
  - Probability displayed with decision

**How to Use**
  - Select values from each input field
  - Click Submit
  - View predicted credit risk and probability

**Tech Stack**
  - Python
  - Pandas
  - Scikit-learn
  - Joblib
  - Gradio
  - Hugging Face Spaces
