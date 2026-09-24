# PowerCo Customer Churn & Price Sensitivity Analysis

## Overview

This project develops and evaluates a machine learning model to investigate customer churn for PowerCo, with a particular focus on the relationship between electricity pricing and customer attrition.

The analysis combines customer characteristics, consumption behaviour, service information, and historical pricing data to identify factors associated with customer churn and determine whether changes in electricity prices provide meaningful predictive information.


The objective is to identify patterns associated with customer churn and determine whether electricity price levels and price changes contain useful information for predicting customer attrition.


## Problem Statement

Customer churn represents a significant commercial challenge for energy suppliers because the loss of customers directly affects recurring revenue.
For most energy providers, understanding whether electricity pricing and changes in pricing are associated with customer churn can provide useful insight into customer behaviour.

The stakeholders at PowerCo are specifically interested in understanding to what extent electricity prices and changes in electricity prices are associated with customer attrition, and which price-related features provide the strongest predictive information?

The goal is to develop a predictive model that identifies customers with a higher probability of churn while examining the contribution of electricity price-related features to those predictions.

## Dataset Summary

**Dataset:** PowerCo Customer Churn Dataset

**Target Variable:** `churn` *"with approximately 9:1 class imbalance"*

**Customer Dataset:** Approximately 14,606 customer records.

**Feature Categories**

* Customer and account information
* Electricity consumption behaviour
* Customer tenure
* Sales and acquisition channels
* Electricity pricing
* Price differences between tariff periods
* 12-month price variations
* Gross and net power margins

The project combines customer-level information with historical electricity pricing data. Historical price records were aggregated to the customer level before being merged with the customer dataset.

## Repository Structure

```text
project_root/
│
├── notebook/
│   └── PowerCo_Churn.ipynb
│
├── data/
│   │
│   │──── client_data.csv
│   │──── price_data.csv
│   │──── out_of_sample_data_with_predictions.csv
│   │──── Data Description.pdf
│   └──── engineered_customer_data.csv
│
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Analytical Approach
### Data Understanding & Integration
The dataset presented a granularity mismatch across the two core data sources:
* Customer Data (client_df): A single unique record per customer.
* Pricing Data (price_df): Multiple historical, time-series records per customer.
A critical finding during this phase was the identification of 1,490 customer IDs present in the pricing dataset but entirely missing from the customer dataset. Because their corresponding behavioral data was unavailable, these records were excluded from the final modeling dataset to protect data integrity.

### Preprocessing & Feature Engineering
Data cleaning involved standardizing data types, handling missing values, validating unique identifiers, and applying structural transformations to match the granularities.

#### Price Feature Engineering
Historical pricing logs were aggregated into unique customer-level metrics across three pricing periods `off-peak, peak, and mid-peak`.

- Price Levels: Extracted the average historical pricing per customer.

- Price Differentials: Calculated the relative structure of customer electricity tariffs by measuring period-to-period differences `(pp12, pp23, pp13)`.

- Price Variation: Computed changes in pricing over a 12-month period to capture historical trends rather than relying solely on static, current price levels. 

#### Customer Behavior Engineering
To capture behavioral indicators of churn without introducing feature redundancy, several consumption metrics were engineered:
- Total 12-months consumption
- Total 6-months consumption
- Forecasted consumption
- Consumption deviation and patterns

#### Encoding & Formatting
* Categorical Variables: Converted binary categories `(e.g., has_gas)` into numerical binary flags. Multi-class categorical variables were mapped structurally based on their intended modeling application.
* Numerical Features: Retained in their original scale. Because tree-based algorithms split data points orthogonally, feature standardization/normalization was not required.


### Model Training & Optimization
A Random Forest Classifier was selected as the core architecture due to its robustness against nonlinear interactions, compatibility with mixed data scales, and inherent interpretability via feature importance and SHAP values.
- Validation Strategy: Implemented a stratified train-test split followed by a 5-fold stratified cross-validation loop to evaluate models independently of single-split biases.
- Hyperparameter Tuning: Executed via RandomizedSearchCV, tuning tree depth, minimum sample splits, leaf sizes, and maximum feature counts per split.
- Class Imbalance & Evaluation Strategy: Because customer churn represents a severe minority class, raw accuracy was rejected as a reliable success metric.
- Algorithmic Adjustment: Enforced cost-sensitive learning by configuring the classifier with class_weight="balanced" to dynamically penalize minority class misclassifications during training.
- Success Metrics: Performance was tracked using Precision, Recall, F1-Score, ROC-AUC, and PR-AUC. Primary business optimization was placed heavily on optimizing Class-1 Recall and F1-Score to ensure maximum identification of true churners.



## Key Findings

* The initial Random Forest model showed limited ability to identify churners, particularly in terms of recall.
* Class weighting and hyperparameter tuning improved churn detection.
* The tuned Random Forest increased churn recall from approximately **10% to 36%** and F1-Score from approximately **17% to 32%**.
* Lower classification thresholds substantially increased churn recall, but this came with a corresponding reduction in precision.
* Historical pricing information contains useful information for distinguishing customers.
* Among the price-related variables, **off-peak pricing showed relatively strong model importance**, indicating that it contained useful predictive information for churn classification.


## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SHAP

## Author

**CHIGOZIE OKONKWO**

Electrical & Electronics Engineer | Data Scientist | PowerCo Customer Churn & Price Sensitivity Analysis
* [LinkedIn](linkedin.com/in/chigozie-okonkwo)



```python

```
