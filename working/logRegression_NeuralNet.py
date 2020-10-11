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
