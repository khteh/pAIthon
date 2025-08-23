import matplotlib.pyplot as plt
import pandas

def FeatureImportance(columns, metrics, n=25):
    df = pandas.DataFrame({
        "Features": columns,
        "Metrics": metrics
    }).sort_values("Metrics", ascending=False, ignore_index=True).reset_index(drop=True)
    fig, ax = plt.subplots()
    ax.barh(df.Features[:n], df.Metrics[:n])
    ax.set_ylabel("Features")
    ax.set_xlabel("Metrics")
    ax.set_title("Feature Importance")
    ax.invert_yaxis()