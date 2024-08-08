Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats
step 2: Load the Dataset
file_path = '/content/diabetes_unclean.csv'
data = pd.read_csv(file_path)
step 3:Handle Missing Values
print("Initial dataset:")
print(data.head())
print("\nMissing values per column:")
print(data.isnull().sum())
Step 4: Handle Outliers
num_cols = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
data[num_cols] = imputer.fit_transform(data[num_cols])
Step 5: Standardize Data Formats
cat_cols = data.select_dtypes(include=[object]).columns
imputer = SimpleImputer(strategy='most_frequent')
data[cat_cols] = imputer.fit_transform(data[cat_cols])
Step 7: Remove Duplicates
print("\nNumber of duplicate rows:")
print(data.duplicated().sum())
data = data.drop_duplicates()
Step 8: Feature outliers
z_scores = np.abs(stats.zscore(data[num_cols]))
outliers = (z_scores > 3).any(axis=1)
print("\nNumber of outlier rows:")
print(np.sum(outliers))
data = data[~outliers]
Step 9: Apply Transformation Techniques
if 'date_column' in data.columns:  # Replace 'date_column' with your actual date column name
    data['date_column'] = pd.to_datetime(data['date_column'])
    if 'feature1' in data.columns and 'feature2' in data.columns:
    data['feature_ratio'] = data['feature1'] / (data['feature2'] + 1e-6) 
    scaler = MinMaxScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])
standard_scaler = StandardScaler()
data[num_cols] = standard_scaler.fit_transform(data[num_cols])
for col in num_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
sns.pairplot(data[num_cols])
plt.show()
Step 11: Document Procedures
with open('data_cleaning_log.txt', 'w') as f:
    f.write("Data Cleaning Log:\n")
    f.write("1. Missing values handled by imputation.\n")
    f.write("2. Outliers detected and removed using Z-scores.\n")
    f.write("3. Data formats standardized.\n")
    f.write("4. Duplicates removed.\n")
    f.write("5. New feature 'age_group' created.\n")
    f.write("6. Numerical features scaled using StandardScaler.\n")
    Step 12: Save the Cleaned Dataset
    df_clean.to_csv('clean_diabetes.csv', index=False)
    
