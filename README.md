# Real-Time Smart Traffic Routing Using Graph Neural Networks with Dynamic Route Optimization

# Project Overview

This project presents an intelligent traffic routing system using Graph Convolutional Networks (GCNs) to predict traffic conditions and optimize vehicle routes in real time.

The proposed system constructs a road network graph from GPS trajectory data, predicts traffic levels using a Graph Neural Network, dynamically updates edge weights, and computes the optimal route using Dijkstra's and A* algorithms.

------------------------

## Features

- GPS Data Processing
- Data Cleaning and Feature Engineering
- Road Network Graph Construction
- Graph Convolutional Network (GCN)
- Traffic Level Prediction
- Dynamic Edge Weight Updating
- Dijkstra Shortest Path
- A* Shortest Path
- Interactive Folium Map
- Performance Evaluation
- Real-Time Traffic Simulation

------------------------

# Dataset

TaxiGPS Dataset

Features:

- Taxi_ID
- Timestamp
- Latitude
- Longitude
- Road_ID
- Speed_kmh
- Distance_km
- Travel_Time_min
- Traffic_Level


------------------------
# Project Structure


Traffic-Routing-GNN/
│
├── dataset/
│      taxigps.csv
│
├── notebooks/
│      traffic_routing.ipynb
│
├── models/
│      gcn_model.pth
│
├── outputs/
│      confusion_matrix.png
│      roc_curve.png
│      accuracy.png
│      loss.png
│      traffic_route.html
│
├── app/
│      app.py
│
├── requirements.txt
│
└── README.md


------------------------

## Model Architecture

Input Features

- Speed
- Distance
- Hour
- Day
- Month

↓

GCN Layer 1

↓

GCN Layer 2

↓

Output Layer

↓

Traffic Prediction

↓

Dynamic Routing

------------------------

# Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

-------------------------

# Routing Algorithms

- Dijkstra
- A*

-------------------------

# Technologies

- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Folium
- Streamlit

-------------------------

# Running the Project

Install dependencies

#bash
pip install -r requirements.txt


Run Streamlit

#bash
streamlit run app/app.py




# Results

- GCN Traffic Prediction
- Dynamic Route Optimization
- Interactive Map
- Performance Evaluation

---------------------------------

# Author

Nimesh Wijebandara

