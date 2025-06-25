Title: Electricity Demand Forecasting and Pattern Analysis Using Machine Learning

👥 Group Members
Bilal Bashir - 22i-1901

Riyyan Ahmad - 22i-2069

📌 Project Overview
This project presents a full data science pipeline focused on electricity demand forecasting, pattern analysis, and interactive deployment. It integrates weather data with energy consumption records to model, cluster, and predict electricity demand using modern data mining techniques. The project is structured across four major components:

dataset.ipynb - Data Collection, Integration, and Preprocessing

clustering.ipynb - Unsupervised Clustering using PCA, K-Means, DBSCAN, and Hierarchical Clustering

prediction.ipynb - Supervised Modeling using ML/DL/TS approaches (LSTM, XGBoost, SARIMA, etc.)

app.py - Streamlit-based Interactive Web Application

📁 File Descriptions
1. dataset.ipynb – Data Processing Pipeline
Integrated weather (JSON) and demand (CSV) datasets

Handled missing values, datetime conversions, and time zone normalization

Engineered features: hour, day of week, month, season

Applied standardization using StandardScaler

Performed anomaly detection via:

Z-Score

Isolation Forest

Aggregated data for temporal analysis (daily & weekly)

Final dataset fields: local_time, temperature, humidity, windspeed, demand, etc.

2. clustering.ipynb – Unsupervised Pattern Recognition
Used 4 features: temperature, humidity, windspeed, and demand

Applied PCA for 2D visualization

Implemented:

K-Means (Elbow method & Silhouette score)

DBSCAN (eps=0.5, min_samples=5)

Hierarchical Clustering (Ward’s method)

Visualized cluster groupings in 2D PCA space

3. prediction.ipynb – Forecasting and Modeling
Feature Engineering:

Lag features: 24, 48, 168 hrs

Rolling averages, cyclical encoding

One-hot encoding for season/city

Weekend indicator

Models used:

Baseline: Previous Day, Week Average

Regression: Linear, Polynomial

Tree Models: Random Forest, XGBoost

Neural Networks: Feedforward ANN, LSTM

Time Series: ARIMA, SARIMA

Ensemble: Stacking with Gradient Boosting

Evaluation Metrics:

MAE, RMSE, MAPE, R²

Visualizations: Prediction comparisons, MAE bar charts, feature importances

4. app.py – Interactive Forecasting App (Streamlit)
Generates or accepts uploaded data

Visualizes clustering using PCA + K-Means

Supports real-time demand prediction via:

LSTM

ARIMA/SARIMA

Ensemble Averaging

Features:

Lag-based feature generation

Real-time MAE/RMSE/MAPE scores

Customizable forecast horizons (up to 72 hrs)

🧠 Key Findings
🔹 Demand Patterns
Strong correlation between temperature extremes and energy demand

Seasonal effects most prominent in summer and winter

Daily cycles: Peak during working hours; low during night

Cluster analysis revealed unique consumption behaviors based on time and season

🔹 Forecasting Insights
LSTM: Best for long-term, nonlinear pattern capture

SARIMA: Effective for short-term seasonality

Ensemble Models: Most robust and balanced across scenarios

Most important features: Temperature, previous day’s demand, time of day, season

⚙️ Technologies & Libraries
Python, Jupyter Notebook, Streamlit

pandas, numpy, scikit-learn, xgboost, keras, statsmodels, matplotlib, seaborn

pytz, datetime, isolationforest, ARIMA/SARIMA, LSTM

✅ How to Run
Preprocessing: Run dataset.ipynb to generate cleaned and feature-rich dataset

Clustering: Execute clustering.ipynb to view unsupervised pattern discovery

Modeling: Use prediction.ipynb for model training, evaluation, and feature importance

Deployment: Launch app.py using:

bash
Copy
Edit
streamlit run app.py
📞 Contact
For queries or collaboration, feel free to contact:
