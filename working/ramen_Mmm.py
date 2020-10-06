#%%
# Practice one-hot encoder
# %%
# Import  dependencies
import pandas as pd
import sklearn as skl

# Read in our ramen data
ramen_df = pd.read_csv("ramen-ratings.csv")

# Print out the Country value counts
country_counts = ramen_df.Country.value_counts()
country_counts
# %%
# count rows
ramen_df.Country.count().sum()
# %%
# visualize which values are uncommon enough to bin into 'other' category
# Visualize the value counts
country_counts.plot.density()
# %%
# from density plot chose to bin all countries with 
# less than 100 entries into other category
# Determine which values to replace
replace_countries = list(country_counts[country_counts < 100].index)

# Replace in DataFrame
for country in replace_countries:
    ramen_df.Country = ramen_df.Country.replace(country,"Other")

# Check to make sure binning was successful
ramen_df.Country.value_counts()
# %%
# done binning, next transpose country variable
# Create the OneHotEncoder instance
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)

# Fit the encoder and produce encoded DataFrame
encode_df = pd.DataFrame(enc.fit_transform(ramen_df.Country.values.reshape(-1,1)))

# Rename encoded columns
encode_df.columns = enc.get_feature_names(['Country'])
encode_df.head()
# %%
