# -------------------------------------------------------------------------
# AUTHOR: Aaliyah Divinagracia
# FILENAME: pca.py
# SPECIFICATION: apply PCA multiple times on the
# heart_disease_dataset.csv, each time removing a single and distinct feature and printing the
# corresponding variance explained by PC1
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
file_path = "heart_disease_dataset.csv"
df = pd.read_csv(file_path)

#Create a training matrix without the target variable (Heart Diseas)
#--> add your Python code here

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#Get the number of features
#--> add your Python code here
num_features = df_features.shape[1]

# Run PCA for 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    reduced_data = pd.DataFrame(scaled_data, columns=df_features.columns).drop(columns=[df_features.columns[i]])

    reduced_scaled_data = scaler.fit_transform(reduced_data)

    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    pca_variances[df_features.columns[i]] = pca.explained_variance_ratio_[0]

# Find the maximum PC1 variance
# --> add your Python code here
max_pc1_feature = max(pca_variances, key=pca_variances.get)
max_pc1_variance = pca_variances[max_pc1_feature]

#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print(f"Highest PC1 variance found: {max_pc1_variance:.6f} when removing {max_pc1_feature}")





