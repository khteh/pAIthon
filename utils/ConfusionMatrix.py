import pandas, seaborn, matplotlib.pyplot as plt

def ConfusionMatrix(truths, predictions, title):
    """
    Show confusion matrix using:
    (1) Pandas' crosstab
    (2) Seaborn heatmap
    """
    assert truths.shape[0] == 1
    assert predictions.shape[0] == 1
    print("Confusion Matrix:")
    matrix = pandas.crosstab(truths, predictions, rownames=["Truths"], colnames=["Predictions"])
    print(matrix)
    ax = seaborn.heatmap(matrix, annot=True)
    ax.set_title(title)
    plt.show()
