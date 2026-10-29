Machine Learning for Food Demand Forecasting

MSc Data Science Dissertation | University of East London

Author: Satish Kumar Nagaraj
Degree: MSc Data Science (Distinction)
Dissertation Grade: 78%
Research Area: Machine Learning | Demand Forecasting | Supply Chain Analytics | Food Manufacturing

---

Project Overview

This research investigates the application of machine learning techniques to forecast product demand and explore their potential contribution to food manufacturing operations.

Using the Corporación Favorita Grocery Sales Forecasting dataset, three ensemble learning algorithms were developed, optimised and evaluated:

- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

The project follows an adapted CRISP-DM methodology, incorporating data preprocessing, exploratory data analysis, feature engineering, hyperparameter optimisation and chronological model validation.

Key Result: XGBoost achieved the strongest overall predictive performance, with an MAE of 3.7960, RMSE of 28.3397 and R² of 0.1312.

The findings demonstrate the potential of machine learning for demand forecasting while identifying important limitations affecting real-world deployment.

---

Business Problem

Demand uncertainty creates significant challenges for food manufacturers and supply chain organisations.

Inaccurate forecasts can contribute to:

- Overproduction and food waste
- Inventory shortages
- Increased operational costs
- Inefficient production scheduling
- Poor resource allocation

This project examines how historical demand data and machine learning could provide additional information to support operational planning.

Research objective: Develop and compare machine learning models for product demand forecasting and critically assess their potential relevance to food manufacturing decision-making.

---

Dataset

Source: Corporación Favorita Grocery Sales Forecasting — Kaggle

The research uses historical grocery retail sales data from Ecuador.

Experimental Dataset

Attribute| Details
Total observations| 300,000
Processed dataset columns| 37
Training observations| 240,000
Testing observations| 60,000
Training/testing split| 80:20 chronological
Data period| January–September 2013
Target variable| "unit_sales"
Random seed| 42

The dataset includes information relating to products, stores, promotions, transactions and temporal demand characteristics.

Important distinction: The dataset represents retail sales rather than manufacturing production records. Consequently, this research investigates demand forecasting as potential decision support rather than demonstrating direct production optimisation.

---

Technologies and Libraries

Category| Technologies
Programming| Python 3
Data Processing| Pandas, NumPy
Machine Learning| Scikit-learn
Ensemble Models| Random Forest, XGBoost, LightGBM
Hyperparameter Tuning| RandomizedSearchCV
Cross-Validation| TimeSeriesSplit
Data Visualisation| Python visualisation libraries
Research Framework| Adapted CRISP-DM

---

Methodology

1. Data Preparation

Multiple related data sources were integrated and processed.

Activities included:

- Data quality assessment
- Missing-value handling
- Duplicate checks
- Date transformations
- Categorical encoding
- Numerical preprocessing

2. Exploratory Data Analysis

Exploratory analysis investigated demand distributions, temporal patterns, extreme observations and relationships between variables.

The analysis identified substantial positive skewness in product demand, with a relatively small number of exceptionally high-sales observations.

3. Feature Engineering

Historical demand and calendar-based features were developed.

These included:

- Calendar variables
- Day-of-week and weekend indicators
- Cyclical temporal transformations
- Lagged demand
- Rolling averages
- Rolling standard deviations

Lag and rolling features were designed using historical observations to reduce the risk of target leakage.

4. Machine Learning Development

Three supervised regression algorithms were evaluated using a consistent preprocessing and testing framework.

Random Forest: Bagging-based ensemble learning.

XGBoost: Gradient boosting with regularisation and sequential error correction.

LightGBM: Gradient boosting designed for computational efficiency.

5. Hyperparameter Optimisation

RandomizedSearchCV was combined with three-fold TimeSeriesSplit validation.

The executed configuration evaluated four hyperparameter combinations per algorithm, resulting in 12 cross-validation fits per model.

6. Model Evaluation

Models were assessed using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Root Mean Squared Logarithmic Error (RMSLE)
- R² Score

---

Experimental Results

Model Performance Comparison

Model| MAE ↓| RMSE ↓| MAPE ↓| R² ↑
Random Forest| 4.2890| 28.6838| 0.8692| 0.1100
XGBoost| 3.7960| 28.3397| 0.8328| 0.1312
LightGBM| 3.9644| 28.4217| 0.8277| 0.1262

MAPE was calculated using observations with non-zero actual demand.

Best-Performing Model: XGBoost

XGBoost achieved:

- MAE: 3.7960
- MSE: 803.1371
- RMSE: 28.3397
- MAPE: 0.8328
- RMSLE: 0.5409
- R²: 0.1312

XGBoost achieved the lowest MAE, MSE, RMSE and RMSLE, along with the highest R².

However, LightGBM achieved the lowest MAPE.

The differences between XGBoost and LightGBM were relatively small.

Interpretation

Although XGBoost performed best overall, its R² of 0.1312 indicates that the model explained approximately 13.1% of the variation in individual test observations.

This highlights the difficulty of forecasting highly variable retail demand and the need for additional explanatory information.

The results should therefore be interpreted as an experimental comparison rather than evidence of production-ready forecasting accuracy.

---

Key Findings

Historical Demand Matters

Lagged sales and rolling-demand statistics emerged as important predictors, particularly for Random Forest and XGBoost.

Gradient Boosting Performed Competitively

XGBoost and LightGBM achieved slightly lower forecasting errors than Random Forest.

High-Demand Events Remain Challenging

Prediction errors were particularly noticeable for unusually high-demand observations.

Forecasting Is Not Production Optimisation

Machine learning forecasts can provide information for production planning, but separate operational models and constraints are needed to determine optimal production decisions.

---

Potential Business Applications

The research identifies possible applications in:

Production Planning: Using anticipated demand to inform production requirements.

Inventory Management: Supporting stock replenishment and procurement decisions.

Waste Reduction: Identifying opportunities to reduce mismatches between supply and demand.

Capacity Planning: Providing demand information for workforce and resource planning.

Continuous Improvement: Using predictive analytics to support operational decision-making.

These applications are proposed opportunities and were not validated through a live manufacturing deployment.

---

Research Limitations

The study identified several limitations:

1. Retail sales were used as a proxy for downstream demand rather than direct manufacturing records.
2. All three models exhibited relatively low R² values.
3. Extreme demand observations remained difficult to forecast.
4. The chronological test partition contained only three distinct dates.
5. The experiment used a 300,000-observation sample rather than the complete dataset.
6. Further comparisons with simple forecasting baselines are necessary.
7. No live manufacturing deployment or direct production optimisation was conducted.

---

Future Development

Future improvements could include:

- Longer out-of-time validation
- Additional demand-related features
- Forecasting baseline comparisons
- Forecast uncertainty estimation
- Interactive forecasting dashboards
- Integration with manufacturing-specific operational datasets
- Production planning and inventory optimisation models

---

Academic Information

Dissertation Title: The Use of Machine Learning Techniques in Optimizing Food Manufacturing Operations to Meet Demands

University: University of East London

Programme: MSc Data Science

Dissertation Grade: 78%

Year: 2026

---

About the Author

Satish Kumar Nagaraj

MSc Data Science graduate with Distinction from the University of East London.

My professional interests include data analytics, machine learning, business intelligence, operational analytics, supply chain management and continuous improvement.

I am particularly interested in connecting technical data science methods with practical operational challenges in food manufacturing and other industries.

Portfolio: https://portfolio-wwoq.vercel.app/

---

Project Status

Completed academic research — 2026

The dissertation's experimental findings are documented. Source code, visualisations and reproducibility materials can be added to this repository as they are prepared for publication.

---

This project was developed for academic research and demonstrates machine learning experimentation, model evaluation and critical analysis of potential business applications.
