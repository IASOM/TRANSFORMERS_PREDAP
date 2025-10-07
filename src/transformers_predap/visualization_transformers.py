# plot example 10 diags -------------------------
import seaborn as sns

def plot_example(df,title):
    # PLOT RAW DATA (example 10 diags)
    dff = df
    dff["date"] = dff.index
    dff = dff[['J00','COV-19','date']]
    sns.set_theme(rc={'figure.figsize':(20,8)})
    sns.lineplot(data=dff.replace('nan', float('nan')).melt(id_vars=['date']),x='date', y='value', hue='variable').set(title=title)