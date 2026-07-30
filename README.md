🌦️ Weather station containg Rain Prediction & Weather Forecasting AI.

This repository contains an end-to-end Machine Learning pipeline that uses Random Forest architectures to predict rainfall and forecast future weather trends (Temperature and Humidity) using real-time data from the OpenWeather API.

🚀 Overview

The project implements a dual-model approach:

Classification: A Random Forest Classifier determines the probability of rain tomorrow (Binary Classification).

Regression: A Random Forest Regressor predicts the numerical values for Temperature and Humidity for the next 5 hours based on historical patterns.

📊 The Pipeline
The project follows a standard data science lifecycle:

Data Acquisition: Real-time weather fetching via requests and OpenWeather API.

Preprocessing: Categorical encoding of wind directions and handling missing values with Pandas.

Feature Engineering: Conversion of compass degrees to cardinal directions for better interpretability.

Modeling: Implementation of ensemble learning using 100 decision trees to ensure model stability.

🛠️ Technical Stack
Language: Python

Libraries: Scikit-learn, Pandas, NumPy, Requests

API: OpenWeatherMap API

height="281" alt="image" src="https://github.com/user-attachments/assets/c2673425-429b-45e2-8609-525ac391e07a" />

 
Model Evaluation & Performance Analysis
📊 Performance Metrics Summary
Overall Accuracy: 85%

Log Loss: 0.3366 (Indicates strong probability calibration and low uncertainty)

Dataset Size: 73 evaluation samples (57 No Rain / 16 Rain)

<img width="1023" height="454" alt="image" src="https://github.com/user-attachments/assets/fe04092c-0daa-4b57-a731-f3717309d958" />

<img width="1075" height="476" alt="image" src="https://github.com/user-attachments/assets/31da7a67-e103-4da5-a3af-4e38ea508604" />


Clone the repository:
https://github.com/nafissa1-web/Weather-station


📂 Project Structure
Plaintext
├── main.py              # Main logic for API fetching and model training
├── weather.csv          # Historical dataset used for training
├── requirements.txt     # List of required Python packages
├── .gitignore           # Prevents clutter (pycache, checkpoints)
└── LICENSE              # MIT License

Author: Nafissa KADRI

Focus: Artificial Intelligence | Machine Learning | Data Analysis
