import math, numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm

"""
https://realpython.com/logistic-regression-python/
L1 regularization penalizes the LLF with the scaled sum of the absolute values of the weights: |𝑏₀|+|𝑏₁|+⋯+|𝑏ᵣ|.
L2 regularization penalizes the LLF with the scaled sum of the squares of the weights: 𝑏₀²+𝑏₁²+⋯+𝑏ᵣ².
Elastic-net regularization is a linear combination of L1 and L2 regularization.

LogisticRegression:
- penalty is a string ('l2' by default) that decides whether there is regularization and which approach to use. Other options are 'l1', 'elasticnet', and 'none'.
- solver is a string ('liblinear' by default) that decides what solver to use for fitting the model. Other options are 'newton-cg', 'lbfgs', 'sag', and 'saga'.
- C is a positive floating-point number (1.0 by default) that defines the relative strength of regularization. Smaller values indicate stronger regularization.
- random_state is an integer, an instance of numpy.RandomState, or None (default) that defines what pseudo-random number generator to use.
You should carefully match the solver and regularization method for several reasons:

'liblinear' solver doesn’t work without regularization.
'newton-cg', 'sag', 'saga', and 'lbfgs' don’t support L1 regularization.
'saga' is the only solver that supports elastic-net regularization.    
"""
x = numpy.array([4, 3, 0])
c1 = numpy.array([-.5, .1, .08])
c2 = numpy.array([-.2, .2, .31])
c3 = numpy.array([.5, -.1, 2.53])

def sigmoid(z):
    # add your implementation of the sigmoid function here
    # s(z)=1÷(1+exp(−z))
    print(1/(1+math.exp(-z)))

# calculate the output of the sigmoid for x with all three coefficients
result1 = x @ c1
result2 = x @ c2
result3 = x @ c3
sigmoid(result1)
sigmoid(result2)
sigmoid(result3)

def single_variate_binary_classification(C: float = 1.0):
    """
    larger value of C means weaker regularization, or weaker penalization related to high values of 𝑏₀ and 𝑏₁
    """
    x = numpy.arange(10).reshape(-1, 1) # one column for each input, and the number of rows should be equal to the number of observations.
    y = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    model = LogisticRegression(solver='liblinear', C=C, random_state=0).fit(x, y)
    # The attribute .classes_ represents the array of distinct values that y takes:
    print(f"LogisticRegression model.classes_: {model.classes_}, intercept (b0): {model.intercept_}, slope (b1): {model.coef_}")
    """
    matrix of probabilities that the predicted output is equal to zero or one
    each row corresponds to a single observation. The first column is the probability of the predicted output being zero, that is 1 - 𝑝(𝑥). The second column is the probability that the output is one, or 𝑝(𝑥).
    """
    print(f"prediction probabilities: {model.predict_proba(x)}")
    predictions = model.predict(x)
    confusion = confusion_matrix(y, predictions)
    print(f"Actual predictions: {predictions}, score: {model.score(x, y)}, confusion matrix: {confusion}")
    ShowConfusionMatrix(confusion)
    print("Classification Report:")
    print(classification_report(y, predictions))

def single_variate_binary_classification_statsmodels():
    x = numpy.arange(10).reshape(-1, 1) # one column for each input, and the number of rows should be equal to the number of observations.
    y = numpy.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])
    """
    add_constant() takes the array x as the argument and returns a new array with the additional column of ones.
    The first column of x corresponds to the intercept 𝑏₀. The second column contains the original values of x.
    """
    x = sm.add_constant(x)
    model = sm.Logit(y, x).fit(method='newton') # or, if you want to apply L1 regularization, with .fit_regularized():
    print(f"params: {model.params}")
    probabilities = model.predict(x) # probabilities of the predicted outputs being equal to one or p(x)
    print(f"prediction probabilities: {probabilities}")
    predictions = (probabilities >= 0.5).astype(int)
    print(f"predictions: {predictions}")
    confusion = model.pred_table()
    print(f"confusion matrix: {confusion}")
    ShowConfusionMatrix(confusion)
    print("summary:")
    print(model.summary())
    print(model.summary2())

def ShowConfusionMatrix(confusion):
    """
    Creates a heatmap that represents the confusion matrix.
    Different colors represent different numbers and similar colors represent similar numbers
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(confusion)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, confusion[i, j], ha='center', va='center', color='red')
    plt.show()

if __name__ == "__main__":
    single_variate_binary_classification()
    single_variate_binary_classification(10.0)
    single_variate_binary_classification_statsmodels()