import numpy, matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')

def UniVariatePlot(x: list[float], y: list[float], w: list[float], b: float, title: str, yLabel: str, xLabel: str):
    plt.title(title)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.scatter(x, y, marker='x', c='r')
    plt.show()
