#%%
"""
Build a classifier that can predict if a loan will be provided.

Random forest models will only handle tabular data

Deep learning will handle images, NLP, etc.
"""
#%%
# Import dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import tensorflow as tf

# Import input dataset
loans_df = pd.read_csv('loan_status.csv')
loans_df.head()
# %%
# both models require pre-processing data
# pre-processing first this time; get ready to encode:

# Generate our categorical variable list
loans_cat = loans_df.dtypes[loans_df.dtypes == "object"].index.tolist()

# Check the number of unique values in each column
loans_df[loans_cat].nunique()
# %%
# Check the unique value counts to see if binning is required
loans_df.Years_in_current_job.value_counts()
# %%
# categorical variables look ready to encode

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(loans_df[loans_cat]))

# Add the encoded variable names to the DataFrame
encode_df.columns = enc.get_feature_names(loans_cat)
encode_df.head()
# %%
# Merge one-hot encoded features and drop the originals
loans_df = loans_df.merge(encode_df,left_index=True, right_index=True)
loans_df = loans_df.drop(loans_cat,1)
loans_df.head()
# %%
