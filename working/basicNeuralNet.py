#%%
# Import dependencies
import pandas as pd
import matplotlib as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import sklearn as skl
import tensorflow as tf
# %%
# create dummy data with scikit-Learn's module
# 1000 data points with two dimensions/features
# Generate dummy dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=78)

# Creating a DataFrame with the dummy data
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Target"] = y

# Plotting the dummy data
df.plot.scatter(x="Feature 1", y="Feature 2", c="Target", colormap="winter")
# %%
# Use sklearn to split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
# %%
# Normalize/standardize data before training
# Create scaler instance
X_scaler = skl.preprocessing.StandardScaler()

# Fit the scaler
X_scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
# %%
# Create the Keras Sequential model
# input and first hidden layer are always built in the same instance
nn_model = tf.keras.models.Sequential()
# %%
# add layers to our Sequential model using Keras’ Dense class. 
# define few params for first layer

# input_dim parameter indicates how many inputs will be in the model (in this case two)
# units parameter indicates how many neurons we want in the hidden layer (in this case one)
# activation parameter indicates which activation function to use...
# use the ReLU activation function to allow  hidden layer to identify and train on nonlinear relationships in the dataset

# Add our first Dense layer, including the input layer
nn_model.add(tf.keras.layers.Dense(units=1, activation="relu", input_dim=2))
# %%
