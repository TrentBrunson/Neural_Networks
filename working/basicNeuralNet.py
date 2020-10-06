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
# build input and hidden layers
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
# build output layer
# binary outputs only require one output neuron
# Add the output layer that uses a probability activation function
nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
# %%
# verify

# Check the structure of the Sequential model
nn_model.summary()
# %%
# use the adam optimizer, which uses a gradient descent approach 
# to ensure that the algorithm will not get stuck on weaker classifying variables and features
# As for the loss function, use binary_crossentropy, 
# which is specifically designed to evaluate a binary classification model
# two main types of evaluation metrics—the model predictive accuracy and model mean squared error (MSE) 
# accuracy for classification models and mse for regression models
# model predictive accuracy, the higher the number the better, whereas for regression models, 
# MSE should reduce to zero
# %%
# Compile the Sequential model together and customize metrics
nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# %%
# fit method and provide the x training values and y training values, as well as the number of epochs
# Fit the model to the training data
fit_model = nn_model.fit(X_train_scaled, y_train, epochs=100)
# %%
# the model object stores loss and accuracy metrics for all epochs
# Create a DataFrame containing training history
history_df = pd.DataFrame(fit_model.history, index=range(1,len(fit_model.history["loss"])+1))

# Plot the loss
history_df.plot(y="loss")
# %%
# Plot the accuracy over time/# of epochs
history_df.plot(y="accuracy")
# %%
# use the evaluate method and print the testing loss and accuracy values

# Evaluate the model using the test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
# %%
# use predict_classes method to generate predictions on new data

# Predict the classification of a new set of blob data
new_X, new_Y = make_blobs(n_samples=10, centers=2, n_features=2, random_state=78)
new_X_scaled = X_scaler.transform(new_X)
nn_model.predict_classes(new_X_scaled)
# %%
# generate some nonlinear moon-shaped data using Scikit-learn’s make_moons method 
# and visualize it using Pandas and Matplotlib

from sklearn.datasets import make_moons

# Creating dummy nonlinear data
X_moons, y_moons = make_moons(n_samples=1000, noise=0.08, random_state=78)

# Transforming y_moons to a vertical vector
y_moons = y_moons.reshape(-1, 1)

# Creating a DataFrame to plot the nonlinear dummy data
df_moons = pd.DataFrame(X_moons, columns=["Feature 1", "Feature 2"])
df_moons["Target"] = y_moons

# Plot the nonlinear dummy data
df_moons.plot.scatter(x="Feature 1",y="Feature 2", c="Target",colormap="winter")
# %%
# split data into training & test sets
# standardize & scale the data

# Create training and testing sets
X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(
    X_moons, y_moons, random_state=78
)

# Create the scaler instance
X_moon_scaler = skl.preprocessing.StandardScaler()

# Fit the scaler
X_moon_scaler.fit(X_moon_train)

# Scale the data
X_moon_train_scaled = X_moon_scaler.transform(X_moon_train)
X_moon_test_scaled = X_moon_scaler.transform(X_moon_test)
# %%
# Training the model with the nonlinear data
model_moon = nn_model.fit(X_moon_train_scaled, y_moon_train, epochs=100, shuffle=True)
# %%
# Create a DataFrame containing training history
history_df = pd.DataFrame(model_moon.history, index=range(1,len(model_moon.history["loss"])+1))

# Plot the loss
history_df.plot(y="loss")
# %%
# Plot the loss
history_df.plot(y="accuracy") 
# %%
