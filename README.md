# Stock Prediction using Sentiment Analysis on Twitter using LSTM

## Project Overview

This project, titled "Stock Prediction using Sentiment Analysis on Twitter using LSTM," aims to develop a predictive system that enhances the accuracy of stock market predictions. The system integrates traditional time series analysis and sentiment analysis from Twitter to predict stock prices. The project specifically focuses on Netflix (NFLX) stock and employs a combination of ARIMA and LSTM models to achieve its objectives.

## Authors

- V Naveen Kumar (Roll No: 20N31A05N9)
- Venishetty Sriram (Roll No: 20N31A05P2)
- Gaddipati Rakesh (Roll No: 20N31A05Q0)

## Guide

- Mrs. T Padmaja, Assistant Professor, Department of CSE, Malla Reddy College of Engineering and Technology

## Table of Contents

1. [Introduction](#introduction)
2. [Purpose and Objectives](#purpose-and-objectives)
3. [Existing and Proposed System](#existing-and-proposed-system)
4. [Scope of the Project](#scope-of-the-project)
5. [System Requirements](#system-requirements)
6. [Methodology](#methodology)
7. [Implementation](#implementation)
8. [Conclusion](#conclusion)
9. [Bibliography](#bibliography)

## Introduction

The stock market is influenced by various unpredictable factors, including political events and social media impact. This project combines traditional statistical methods like ARIMA with deep learning techniques such as LSTM to predict stock prices, integrating sentiment analysis from Twitter to enhance prediction accuracy.

## Purpose and Objectives

### Purpose

To develop and evaluate advanced predictive models for stock market forecasting, focusing on Netflix stock, and integrating sentiment analysis from Twitter.

### Objectives

1. **Enhance Prediction Accuracy**: Improve stock market prediction accuracy by comparing ARIMA with LSTM.
2. **Analyze Social Media Impact**: Quantify the influence of Twitter sentiment on stock prices.
3. **Explore Deep Learning Techniques**: Utilize CNN and LSTM to capture intricate patterns in stock market data.
4. **Provide Insights for Stakeholders**: Offer valuable insights for financial analysts and investors.
5. **Contribute to Financial Research**: Advance understanding of stock market behavior through integrated models.

## Existing and Proposed System

### Existing System

Existing approaches primarily use basic sentiment classification and linear regression models for stock market prediction. They often do not account for temporal dynamics and rely on traditional machine learning classifiers.

### Proposed System

The proposed system incorporates:
1. **Deep Learning Techniques**: Uses CNN and LSTM for better pattern recognition in time series data.
2. **Social Media Sentiment Analysis**: Integrates Twitter sentiment to gauge market sentiment.
3. **Comparison with ARIMA**: Uses ARIMA as a baseline to evaluate the effectiveness of deep learning models.
4. **Real-Time Evaluation**: Adapts to changing market conditions with continuous evaluation.
5. **Enhanced Insights**: Provides nuanced insights for better decision-making.
6. **Scalability and Generalizability**: Designed to be scalable and applicable to various stocks and market conditions.

## Scope of the Project

The project encompasses:
- **Data Collection**: Using the Twitter API to gather relevant tweets.
- **Sentiment Analysis**: Developing classifiers to categorize tweets.
- **Temporal Modeling**: Implementing LSTM networks for time series data.
- **Data Integration**: Combining sentiment data with technical indicators.
- **Model Training**: Training LSTM models for stock prediction.
- **Performance Evaluation**: Validating model accuracy and comparing with baseline models.
- **Scalability**: Ensuring the system can handle expanding datasets and multiple target companies.
- **Hybrid Model Enhancement**: Creating a robust hybrid model by combining LSTM and ARIMA outputs.

## System Requirements

### Hardware Requirements

- **Processor**: Multi-core processor (Intel i5 or equivalent) with 2.5 GHz or higher.
- **RAM**: 8 GB or higher.
- **Storage**: 256 GB HDD or SSD.
- **GPU**: NVIDIA GeForce GTX or AMD equivalent with 4 GB or higher (optional but recommended).
- **Network**: High-speed internet connection.
- **Backup**: External Hard Drive or Cloud Storage.

### Software Requirements

- **Development Environment**: Visual Studio Code, Jupyter Notebooks.
- **Programming Language**: Python.
- **Version Control**: Git.
- **Libraries and Frameworks**: TensorFlow, Keras, pandas, scikit-learn, pmdarima, snscrape.
- **APIs**: Twitter API.
- **Operating System**: Windows, Linux, or macOS.

## Methodology

The project involves:
1. **Data Collection and Preprocessing**: Gathering historical stock data and tweets, cleaning and preprocessing the data.
2. **Sentiment Analysis**: Using NLP techniques to analyze tweet sentiments.
3. **Model Development**: Developing ARIMA and LSTM models for prediction.
4. **Integration and Testing**: Integrating sentiment analysis with predictive models and testing their performance.

## Implementation

### Sample Code

```python
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess data
stock_data = pd.read_csv('stock_data.csv')
tweet_data = pd.read_csv('tweet_data.csv')

# Example LSTM model
model = keras.Sequential()
model.add(layers.LSTM(50, return_sequences=True, input_shape=(timesteps, features)))
model.add(layers.LSTM(50))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

### Output Screens and Test Cases

- Implement test cases to validate model performance.
- Output screens showcasing stock predictions and sentiment analysis results.

## Conclusion

The project successfully integrates sentiment analysis with traditional and deep learning models to enhance stock market prediction accuracy. It demonstrates the significant impact of social media on stock prices and provides a robust framework for future financial forecasting research.

## Bibliography

References and research papers related to stock market prediction, sentiment analysis, and deep learning methodologies. 
