import pandas, seaborn
import matplotlib.pyplot as plt
def CorrelationMatrixHeatMap(df : pandas.DataFrame):
    matrix = df.corr()
    fig, ax = plt.subplots(figsize=(15,10))
    ax = seaborn.heatmap(matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu") # https://r02b.github.io/seaborn_palettes/
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5) # To avoid the truncation in y-axis