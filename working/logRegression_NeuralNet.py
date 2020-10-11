#%%
# find similiarty between logistic regression and basic neural network models 
# are in terms of performance; build and evaluate both 
# models using the same training/testing dataset
#%%
# Import dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf

# Import input dataset
diabetes_df = pd.read_csv('diabetes.csv')
diabetes_df.head()
# %%
# prepare data to train both models

# With a logistic regression model, there is no preprocessing or scaling required for the data. 
# However, a basic neural network needs the numerical variables standardized. 
# Therefore, need to keep track of a scaled and unscaled training dataset 
# such that both models have the correct input data in their preferred formats.

# Remove diabetes outcome target from features data
y = diabetes_df.Outcome
X = diabetes_df.drop(columns="Outcome")

# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
# %%
# Next, standardize the numerical variables using Scikit-learn’s StandardScaler

# Preprocess numerical data for neural network

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
# %%
# next train & evalutate models

# Define the logistic regression model
log_classifier = LogisticRegression(solver="lbfgs",max_iter=200)

# Train the model
log_classifier.fit(X_train,y_train)

# Evaluate the model
y_pred = log_classifier.predict(X_test)
print(f" Logistic regression model accuracy: {accuracy_score(y_test,y_pred):.3f}")
# %%
# Next, build, compile, and evaluate the basic neural network model

# defube the basic neural network
nn_model = tf.keras.models.Sequential()
nn_model.add(tf.keras.layers.Dense(units=16, activation='relu', input_dim=8))
nn_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

