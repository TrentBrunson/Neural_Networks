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
# %%
# after scaling check to make sure no data is lost from DF
