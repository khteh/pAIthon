# Model evaluation using scoring parameter
import numpy, pandas as pd
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X, Y, k, title="Model Evaluation Scores"):
    print(f"\n=== {evaluate_model.__name__} {type(model).__name__} (k:{k}) ===")
    cv_acc = cross_val_score(model, X, Y, cv=k, scoring=None) # scoring=None uses model's default scoring evaluation metric, i.e., accuracy for classification, R^2 for regression
    cv_acc = numpy.mean(cv_acc)
    print(f"Cross-validation accuracy: {cv_acc * 100:.2f}%")
    cv_precision = cross_val_score(model, X, Y, cv=k, scoring="precision") # scoring=None uses model's default scoring evaluation metric, i.e., accuracy for classification, R^2 for regression
    cv_precision = numpy.mean(cv_precision)
    print(f"Cross-validation precision: {cv_precision}")
    cv_recall = cross_val_score(model, X, Y, cv=k, scoring="recall") # scoring=None uses model's default scoring evaluation metric, i.e., accuracy for classification, R^2 for regression
    cv_recall = numpy.mean(cv_recall)
    print(f"Cross-validation recall: {cv_recall}")
    cv_f1 = cross_val_score(model, X, Y, cv=k, scoring="f1") # scoring=None uses model's default scoring evaluation metric, i.e., accuracy for classification, R^2 for regression
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