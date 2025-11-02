import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#plt.style.use('./deeplearning.mplstyle')
"""
tight_layout:
The rect parameter defines a rectangle in normalized figure coordinates (left, bottom, right, top) into which the entire subplots area (including labels) will be fit. 
This allows you to specify a custom region within the figure for tight_layout to operate within, leaving space for elements outside this rectangle, such as a main title or a figure-level legend. 
The default value for rect is (0, 0, 1, 1), meaning the entire figure area.
"""
def plot_dataset(x, y, title):
    plt.figure(figsize=(12, 10)) # (width, height)
    plt.scatter(x, y, marker='x', c='r', s=100)
    plt.title(title, fontsize=22, fontweight="bold", y=1.05)
    plt.xlabel("X", fontsize=22)
    plt.ylabel("Y", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)    
    plt.show()

def plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title):
    plt.figure(figsize=(12, 10)) # (width, height)
    plt.scatter(x_train, y_train, marker='x', c='r', label='Training', s=100)
    plt.scatter(x_cv, y_cv, marker='o', c='b', label='Cross Validation', s=100)
    plt.scatter(x_test, y_test, marker='^', c='g', label='Test', s=100)
    plt.title(title, fontsize=22, fontweight="bold", y=1.05)
    plt.xlabel("X", fontsize=22) 
    plt.ylabel("Y", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)    
    plt.legend(fontsize='x-large')
    plt.show()

def plot_train_cv_mses(degrees, train_mses, cv_mses, title):
    degrees = range(1,11)
    plt.figure(figsize=(12, 10)) # (width, height)
    plt.plot(degrees, train_mses, marker='o', c='r', label='Training MSEs')
    plt.plot(degrees, cv_mses, marker='o', c='b', label='CV MSEs')
    plt.title(title, fontsize=22, fontweight="bold", y=1.05)
    plt.xlabel("Degree", fontsize=22)
    plt.ylabel("MSE", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)    
    plt.legend(fontsize='x-large')
    plt.show()

def plot_bc_dataset(x, y, title):
    """
    Plot binary classification dataset
    """
    plt.figure(figsize=(12, 10)) # (width, height)
    for i in range(len(y)):
        marker = 'x' if y[i] == 1 else 'o'
        c = 'r' if y[i] == 1 else 'b'
        plt.scatter(x[i,0], x[i,1], marker=marker, c=c, s=100)
    plt.title(title, fontsize=22, fontweight="bold", y=1.05)
    plt.xlabel("X1", fontsize=22)
    plt.ylabel("X2", fontsize=22)
    y_0 = mlines.Line2D([], [], color='r', marker='x', markersize=12, linestyle='None', label='y=1')
    y_1 = mlines.Line2D([], [], color='b', marker='o', markersize=12, linestyle='None', label='y=0')
    plt.legend(handles=[y_0, y_1], fontsize='x-large')
    plt.show()

def plot_roc_curve(tpr, fpr):
    """
    Plot an ROC curve given the true-positive and false-positive rates of a model.
    sklearn.metrics.RocCurveDisplay provides this functionality
    """
    plt.figure(figsize=(12, 10)) # (width, height)
    # Plot ROC curve
    plt.plot(fpr, tpr, color="red", label="ROC")
    # Plot a line with no predictive power (baseline)
    plt.plot([0, 1], [0, 1], color="blue", linestyle="--", label="Guessing")
    # Customize the plot
    plt.xlabel("False-Positive Rate (fpr)", fontsize=22)
    plt.ylabel("True-Positive Rate (tpr)", fontsize=22)
    plt.title("Receiver Operating Characteristics (ROC)) curve", fontsize=22, fontweight="bold", y=1.05)
    plt.legend(fontsize='x-large')
    plt.show()