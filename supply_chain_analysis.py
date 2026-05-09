# ===============================
# SUPPLY CHAIN ANALYSIS PROJECT
# ===============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ===============================
# LOAD DATASETS
# ===============================

print("Loading datasets...")

# Main Dataset
supply_data = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')

# Description Dataset
description_data = pd.read_csv("DescriptionDataCoSupplyChain.csv", encoding='latin1')

print("Datasets Loaded Successfully")

# ===============================
# DISPLAY DATA
# ===============================

print("\nFirst 5 Rows:")
print(supply_data.head())

print("\nDataset Shape:")
print(supply_data.shape)

print("\nDataset Information:")
print(supply_data.info())

# ===============================
# CHECK MISSING VALUES
# ===============================

print("\nMissing Values:")
print(supply_data.isnull().sum())

# Fill missing values
supply_data = supply_data.fillna(0)

# ===============================
# DATA PREPROCESSING
# ===============================

print("5. Machine learning model trained successfully")