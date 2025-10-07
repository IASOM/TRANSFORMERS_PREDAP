import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#load and visualize data 

# RAW DATA --------------------------------------------------------------
data_path = "C:/Users/34648/OneDrive - Generalitat de Catalunya/Escriptori/waikato/input data/longitudinalitat_DIAGNOSTICS_GROUPED.csv"
df = pd.read_csv(data_path, index_col=0)
df['COV-19'] = df["B34"]+df["U07"]   # join cov19 codes
df = df.drop(['B34', 'U07'], axis=1)

df.index = pd.to_datetime(df.index)
d = pd.to_datetime('2000-01-01')
mask = df.index < d
df = df.loc[~mask]

d = pd.to_datetime('2010-01-01')
mask = df.index < d
df = df.loc[~mask]

df["Overall"] = df.iloc[:, 1:].sum(axis=1)
print(df)

# plot example 10 diags -------------------------
def plot_example(df,title):
    # PLOT RAW DATA (example 10 diags)
    dff = df
    dff["date"] = dff.index
    dff = dff[['J00','COV-19','date']]
    sns.set_theme(rc={'figure.figsize':(20,8)})
    sns.lineplot(data=dff.replace('nan', float('nan')).melt(id_vars=['date']),x='date', y='value', hue='variable').set(title=title)
    
plot_example(df,"RAW DATA (example 10 diags) COVID")

# MAIN TRANSFORMER MODEL 



# RESIDUAL DIAGNOSTICS TRANSFORMER



# RESIDUAL SEASONAL TRANSFORMER 