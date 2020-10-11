#%%
"""
SVMs are a type of binary classifier that use geometric boundaries 
to distinguish data points from two separate groups. 

More specifically, SVMs try to calculate a geometric hyperplane 
that maximizes the distance between the closest data point of both groups

SVMs can build adequate models with linear or nonlinear data

SVMs perform one task and one task very well—they classify and create regression using two groups

Neural networks and deep learning models are capable of producing many outputs.   
Neural network models can be used to classify multiple groups within the same model.

If we only compare binary classification problems, 
SVMs have an advantage over neural network and deep learning models:

Neural networks and deep learning models will often converge on a local minima. 
These models will often focus on a specific trend in the data and could miss the “bigger picture.”
SVMs are less prone to overfitting because they are trying to maximize the distance, 
rather than encompass all data within a boundary.
"""
# %%
# Import dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import tensorflow as tf

# Import input dataset
tele_df = pd.read_csv('bank_telemarketing.csv')
tele_df.head()

# GOAL: predict whether or not a customer is likely to subscribe to 
# a banking service after being targeted by telemarketing advertisements
# %%
# Unlike neural networks and deep learning models, SVMs can handle unprocessed and processed tabular data
# will process data anyway to get ready for neural network comparison

# 1st make sure no categorical variable require bucketing 
# by checking column names & their unique values

# Generate categorical variable list
tele_cat = tele_df.dtypes[tele_df.dtypes == "object"].index.tolist()


# Check the number of unique values in each column
tele_df[tele_cat].nunique()
# %%
# encode categorical data

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(tele_df[tele_cat]))

# Add the encoded variable names to the dataframe
encode_df.columns = enc.get_feature_names(tele_cat)
encode_df.head()
# %%
# Merge one-hot encoded features and drop the originals
tele_df = tele_df.merge(encode_df,left_index=True, right_index=True)
tele_df = tele_df.drop(tele_cat,1)
tele_df.head()
# %%
# split data into training and testing sets
# then normalize the data

# Remove loan status target from features data
y = tele_df.Subscribed_yes.values
X = tele_df.drop(columns=["Subscribed_no","Subscribed_yes"]).values

# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
# %%
# train & evaluate SVM

# Create the SVM model
svm = SVC(kernel='linear')

# Train the model
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test_scaled)
print(f" SVM model accuracy: {accuracy_score(y_test,y_pred):.3f}")
# %%
# train and evaluate deep learning model
