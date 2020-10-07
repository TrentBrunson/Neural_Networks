#%%
# Import dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read in our dataset
hr_df = pd.read_csv("hr_dataset.csv")
hr_df.head()
# %%
# scale the data
# Create the StandardScaler instance
scaler = StandardScaler()
# %%
# Fit the StandardScaler
scaler.fit(hr_df)
# %%
# Scale the data
scaled_data = scaler.transform(hr_df)
# %%
# Create a DataFrame with the scaled data
transformed_scaled_data = pd.DataFrame(scaled_data, columns=hr_df.columns)
transformed_scaled_data.head()
# %%
