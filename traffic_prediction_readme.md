# 🚦 Traffic Prediction using Artificial Neural Networks (ANN)

## 📌 Project Overview

This project implements an Artificial Neural Network (ANN) to predict traffic situations based on various features such as time, day of the week, and vehicle counts. The model assists city transportation departments in analyzing traffic congestion and improving infrastructure planning using vehicle count data collected through computer vision.

---

## 🧠 Problem Statement

City transportation departments need data-driven insights before making traffic management decisions. This project aims to optimize traffic monitoring and prediction by leveraging vehicle count data and machine learning.

---

## 📊 Dataset Description

The dataset includes traffic information with the following features:

- **Vehicle Counts**: `CarCount`, `BikeCount`, `BusCount`, `TruckCount`
- **Time Information**: `Time` (in hours), `Date`, `Day of the week`
- **Total**: Sum of all vehicle types within a 15-minute interval
- **Target Variable**: `Traffic Situation` (categorical)

### Key Characteristics:

| Feature     | Range / Type              |
| ----------- | ------------------------- |
| CarCount    | 5 to 180                  |
| BikeCount   | 0 to 70                   |
| BusCount    | 0 to 50                   |
| TruckCount  | 0 to 60                   |
| Total Count | 21 to 279 (15-min window) |
| Date        | 1 to 31 (Day of Month)    |
| Time        | Hour of day (0–23)        |

---

## ⚙️ Methodology

### 1. Data Preprocessing

- **Exploratory Data Analysis (EDA)**

  - Univariate and bivariate analysis
  - Traffic pattern visualization by hour and weekday

- **Feature Engineering**

  - Log transformation of skewed features (`BikeCount`, `BusCount`, `TruckCount`)
  - Time converted to hour
  - Label encoding for categorical variables

- **Data Preparation**

  - 80:20 Train-Test Split
  - Feature standardization

### 2. Model Architecture

- **Input Layer**: CarCount, log-transformed counts, Time, Date, Day, etc.
- **Hidden Layers**:
  - 1st Hidden Layer: 32 neurons (ReLU)
  - 2nd Hidden Layer: 16 neurons (ReLU)
- **Output Layer**: 4 neurons (one for each class)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (lr = 0.001)

### 3. Training Details

- **Epochs**: 40
- **Batch Size**: 128 (train), 256 (test)
- **Monitoring**: Training & validation loss curves

---

## 💪 Implementation Details

- **Framework**: PyTorch
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Evaluation**: Precision, Recall, F1-score via Classification Report

---

## 📈 Results

The model demonstrated successful training with a decreasing loss trend. It was evaluated on classification metrics (precision, recall, F1-score) for each traffic situation class.

---

## 🚀 Usage

1. **Install Dependencies**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn torch
   ```

2. **Place Dataset**\
   Ensure `Traffic.csv` is in the same directory as the script.

3. **Run Script**

   ```bash
   python traffic_prediction_ann.py
   ```

---

## 🔮 Future Improvements

- Hyperparameter tuning for better performance
- Deeper ANN architectures
- Adding features like weather data
- Deploying the model for real-time predictions

---

## 📜 License

This project is open-source and available for educational and research use.

---

## 🙏 Acknowledgments

- Dataset collected using computer vision techniques
- Developed as part of a deep learning project/course

