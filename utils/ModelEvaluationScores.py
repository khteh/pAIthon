# Model evaluation using scoring parameter
import numpy, pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold

def KFoldCrossValidation(model, X, Y, k, title="Model Evaluation Scores"):
    """
    K-fold cross-validation is a statistical technique used to evaluate how well a machine learning model generalizes to unseen data. 
    It is particularly valuable for getting a robust estimate of model performance when you have a limited amount of data.
    
    How It Works:
    The process follows these core steps:
    (1) Split: The entire dataset is divided into \(k\) equal-sized subsets, known as "folds".
    (2 Iterate: The model is trained and tested \(k\) times. 
        In each iteration:One fold is reserved as the testing (validation) set.The remaining \(k-1\) folds are used as the training set.
    (3) Aggregate: After \(k\) iterations, the performance scores (e.g., accuracy, MSE) from each fold are averaged to provide a single, more reliable performance metric.
    
    Why Use It?
    Maximizes Data: Every data point is used for both training and testing at some point in the process.
    Reduces Bias/Variance: By averaging results across multiple splits, it reduces the risk that your performance score is just a result of a "lucky" or "unlucky" random train-test split.
    Reliable Evaluation: It provides a more comprehensive assessment of how the model will perform on completely new data compared to a simple one-time holdout split.

    Common Variations:
    Stratified K-Fold: Ensures each fold has the same proportion of class labels as the original dataset. This is essential for imbalanced classification tasks.
    Repeated K-Fold: The \(k\)-fold process is repeated multiple times with different random shuffles to further stabilize performance estimates.
    Leave-One-Out (LOOCV): An extreme case where \(k\) equals the total number of samples (\(n\)). Each sample is used as a test set exactly once.
    
    Choosing \(K\):
    A common choice is \(k=10\) (or sometimes \(k=5\)), as these values are generally found to provide a good balance between computational cost and reliable performance estimates. 
    Higher values of \(k\) increase computational time because the model must be trained more times.    
    """
    print(f"\n=== {KFoldCrossValidation.__name__} {type(model).__name__} (k:{k}) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X, Y, cv=cv, scoring=None) # scoring=None uses model's default scoring evaluation metric, i.e., accuracy for classification, R^2 for regression
    cv_acc = numpy.mean(cv_acc)
    print(f"Cross-validation accuracy: {cv_acc * 100:.2f}%")
    cv_precision = cross_val_score(model, X, Y, cv=cv, scoring="precision") # scoring=None uses model's default scoring evaluation metric, i.e., accuracy for classification, R^2 for regression
    cv_precision = numpy.mean(cv_precision)
    print(f"Cross-validation precision: {cv_precision}")
    cv_recall = cross_val_score(model, X, Y, cv=cv, scoring="recall") # scoring=None uses model's default scoring evaluation metric, i.e., accuracy for classification, R^2 for regression
    cv_recall = numpy.mean(cv_recall)
    print(f"Cross-validation recall: {cv_recall}")
    cv_f1 = cross_val_score(model, X, Y, cv=cv, scoring="f1") # scoring=None uses model's default scoring evaluation metric, i.e., accuracy for classification, R^2 for regression
    cv_f1 = numpy.mean(cv_f1)
    print(f"Cross-validation F1: {cv_f1}")
    # Visualize the metrics
    metrics = pd.DataFrame({
        "Accuracy": cv_acc,
        "Precision": cv_precision,
        "Recall": cv_recall,
        "F1": cv_f1
    }, index=[0])
    metrics.T.plot.bar(title=title, legend=False);