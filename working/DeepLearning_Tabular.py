#%%
"""
This dataset contains employee career profile and attrition 
(departure from company) information, such as age, department, 
job role, work/life balance, and so forth. 

Using this data, generate a deep learning model that can 
help to identify whether or not a person is likely to depart from 
the company given his or her current employee profile.
"""

# %%
# Import dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import tensorflow as tf

# Import input dataset
attrition_df = pd.read_csv('HR-Employee-Attrition.csv')
attrition_df.head()
# %%
# check column data types are interpreted correctly
attrition_df.dtypes
# %%
# Next encode categorical data before standardizing data set

# Generate our categorical variable list from the DF
attrition_cat = attrition_df.dtypes[attrition_df.dtypes == "object"].index.tolist()
attrition_cat
# %%
# Check the number of unique values in each column
attrition_df[attrition_cat].nunique()
# %%
# encode columns of data type object: pass attrition_cat variable list in

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(attrition_df[attrition_cat]))

# Add the encoded variable names to the DataFrame
encode_df.columns = enc.get_feature_names(attrition_cat)
encode_df.head()
# %%
# merge ad drop DFs to replace unencoded categorical variables
# merge in new numerically derived categories; drop original column

# Merge one-hot encoded features and drop the originals
attrition_df = attrition_df.merge(encode_df,left_index=True, right_index=True)
attrition_df = attrition_df.drop(attrition_cat,1)
attrition_df.head()
# %%
# next split training and testing data
# then scale and fit
# %%
# only need to keep “Attrition_Yes” column; ignore “Attrition_No” column-it's redundant

# Split our preprocessed data into our features and target arrays
y = attrition_df["Attrition_Yes"].values
X = attrition_df.drop(["Attrition_Yes","Attrition_No"],1).values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# %%
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

"""
Data pre-processing complete
"""
# %%
# after scaling check to make sure no data is lost from DF

# input layer: number of input features to = number of variables
# hidden layers: 2
# reLu activation function to ID non-linear characteristics of the input values
# output layer: sigmoid activation f(x) to help predict probability
# an employee is at risk for attrition

# Define the model - deep neural net
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  8
hidden_nodes_layer2 = 5

nn = tf.keras.models.Sequential()

# 1st hidden layer
nn.add(
    tf.keras.layers.Dense(
        units=hidden_nodes_layer1, 
        input_dim=number_input_features, 
        activation='relu'
    )
)

# 2nd hidden layer
nn.add(
    tf.keras.layers.Dense(
        units=hidden_nodes_layer2, 
        activation='relu'
    )
)

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# check structure of the model
nn.summary()
#%%
# label checkpoints by epoch number & contain them within their own folder
# Import checkpoint dependencies
import os
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the checkpoint path and filenames
os.makedirs("checkpoints/",exist_ok=True)
checkpoint_path = "checkpoints/weights.{epoch:02d}.hdf5"

# %%
# next compile model & define loss and accuracy metrics

# compile the model
# since model tp be used as a binary classifier, 
# use the binary_crossentropy loss function, 
# adam optimizer, and accuracy metrics, 
# which are the same parameters we used for a basic neural network
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# create a callback object for our deep learning model
# Create a callback that saves the model's weights every 5 epochs
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=1000)

# Train the model with callback
fit_model = nn.fit(X_train_scaled,y_train,epochs=100,callbacks=[cp_callback])

# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %%
# Train the model without callbacks
fit_model = nn.fit(X_train,y_train,epochs=100)
# %%
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
# %%
# Practice restoring weights
# 
# use Keras Sequential model’s load_weights method to restore model weights. 
# Defining another model to test this functionality, 
# but with restored weights using the checkpoints rather than training the model

# Define the model - deep neural net
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  8
hidden_nodes_layer2 = 5

nn_new = tf.keras.models.Sequential()

# First hidden layer
nn_new.add(
    tf.keras.layers.Dense(
        units=hidden_nodes_layer1, 
        input_dim=number_input_features, 
        activation="relu"
        )
)

# Second hidden layer
nn_new.add(
    tf.keras.layers.Dense(
        units=hidden_nodes_layer2, 
        activation="relu"
        )
)

# Output layer
nn_new.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compile the model
nn_new.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Restore the model weights
nn_new.load_weights("checkpoints/weights.86.hdf5")

# Evaluate the model using the test data
model_loss, model_accuracy = nn_new.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
# %%
