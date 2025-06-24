import math, numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

"""
https://realpython.com/logistic-regression-python/

Logistic regression is a linear classifier, so you’ll use a linear function 𝑓(𝐱) = 𝑏₀ + 𝑏₁𝑥₁ + ⋯ + 𝑏ᵣ𝑥ᵣ, also called the logit (https://en.wikipedia.org/wiki/Logit).
The logistic regression function 𝑝(𝐱) is the sigmoid function of 𝑓(𝐱): 𝑝(𝐱) = 1 / (1 + exp(−𝑓(𝐱)). As such, it’s often close to either 0 or 1. 
The function 𝑝(𝐱) is often interpreted as the predicted probability that the output for a given 𝐱 is equal to 1. Therefore, 1 − 𝑝(𝑥) is the probability that the output is 0.
To get the best weights, you usually maximize the log-likelihood function (LLF) for all observations 𝑖 = 1, …, 𝑛. This method is called the maximum likelihood estimation and is represented by the equation LLF = Σᵢ(𝑦ᵢ log(𝑝(𝐱ᵢ)) + (1 − 𝑦ᵢ) log(1 − 𝑝(𝐱ᵢ))).
When 𝑦ᵢ = 0, the LLF for the corresponding observation is equal to log(1 − 𝑝(𝐱ᵢ)). If 𝑝(𝐱ᵢ) is close to 𝑦ᵢ = 0, then log(1 − 𝑝(𝐱ᵢ)) is close to 0. This is the result you want. If 𝑝(𝐱ᵢ) is far from 0, then log(1 − 𝑝(𝐱ᵢ)) drops significantly. 
You don’t want that result because your goal is to obtain the maximum LLF. Similarly, when 𝑦ᵢ = 1, the LLF for that observation is 𝑦ᵢ log(𝑝(𝐱ᵢ)). If 𝑝(𝐱ᵢ) is close to 𝑦ᵢ = 1, then log(𝑝(𝐱ᵢ)) is close to 0. If 𝑝(𝐱ᵢ) is far from 1, then log(𝑝(𝐱ᵢ)) is a large negative number.
There’s one more important relationship between 𝑝(𝐱) and 𝑓(𝐱), which is that log(𝑝(𝐱) / (1 − 𝑝(𝐱))) = 𝑓(𝐱). This equality explains why 𝑓(𝐱) is the logit. It implies that 𝑝(𝐱) = 0.5 when 𝑓(𝐱) = 0 and that the predicted output is 1 if 𝑓(𝐱) > 0 and 0 otherwise.
Other classification techniques:
- k-Nearest Neighbors
- Naive Bayes classifiers
- Support Vector Machines
- Decision Trees
- Random Forests
- Neural Networks

L1 regularization penalizes the LLF with the scaled sum of the absolute values of the weights: |𝑏₀|+|𝑏₁|+⋯+|𝑏ᵣ|.
L2 regularization penalizes the LLF with the scaled sum of the squares of the weights: 𝑏₀²+𝑏₁²+⋯+𝑏ᵣ².
Elastic-net regularization is a linear combination of L1 and L2 regularization.

LogisticRegression:
- penalty is a string ('l2' by default) that decides whether there is regularization and which approach to use. Other options are 'l1', 'elasticnet', and 'none'.
- solver is a string ('liblinear' by default) that decides what solver to use for fitting the model. Other options are 'newton-cg', 'lbfgs', 'sag', and 'saga'.
- C is a positive floating-point number (1.0 by default) that defines the relative strength of regularization. Smaller values indicate stronger regularization.
- random_state is an integer, an instance of numpy.RandomState, or None (default) that defines what pseudo-random number generator to use.
- multi_class is a string ('ovr' by default) that decides the approach to use for handling multiple classes. Other options are 'multinomial' and 'auto'.
  - 'ovr' says to make the binary fit for each class.
  - 'multinomial' says to apply the multinomial loss fit.

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

def HandwritingClassification():
    """
    dataset with 1797 observations, each of which is an image of one handwritten digit. Each image has 64 px, with a width of 8 px and a height of 8 px.
    The inputs (𝐱) are vectors with 64 dimensions or values. Each input vector describes one image. Each of the 64 values represents one pixel of the image. The input values are the integers between 0 and 16, depending on the shade of gray for the corresponding pixel. 
    x is a multi-dimensional array with 1797 rows and 64 columns. It contains integers from 0 to 16.
    """
    x, y = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    """
    Standardization is the process of transforming data in a way such that the mean of each column becomes equal to zero, and the standard deviation of each column is one. This way, you obtain the same scale for all columns. Take the following steps to standardize your data:

    Calculate the mean and standard deviation for each column.
    Subtract the corresponding mean from each element.
    Divide the obtained difference by the corresponding standard deviation.
    It’s a good practice to standardize the input data that you use for logistic regression, although in many cases it’s not necessary. Standardization might improve the performance of your algorithm. It helps if you need to compare and interpret the weights. 
    It’s important when you apply penalization because the algorithm is actually penalizing against the large values of the weights.
    """
    scaler = StandardScaler() # perform z-score normalization
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test) # only transforms the argument, without fitting the scaler.
    model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', random_state=0).fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(f"Training data score: {model.score(x_train, y_train)}, test: {model.score(x_test, y_test)}")
    confusion = confusion_matrix(y_test, predictions)
    ShowMulticlassConfusionMatrix(confusion, 12)
    print("Classification Report:")
    print(classification_report(y, predictions))

def ShowMulticlassConfusionMatrix(confusion, font_size: int):
    """
    You can see that the shades of purple represent small numbers (like 0, 1, or 2), while green and yellow show much larger numbers (27 and above).
    The numbers on the main diagonal (27, 32, …, 36) show the number of correct predictions from the test set. For example, there are 27 images with zero, 32 images of one, and so on that are correctly classified. 
    Other numbers correspond to the incorrect predictions. For example, the number 1 in the third row and the first column shows that there is one image with the number 2 incorrectly classified as 0.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(confusion)
    ax.grid(False)
    ax.set_xlabel('Predicted outputs', fontsize=font_size, color='black')
    ax.set_ylabel('Actual outputs', fontsize=font_size, color='black')
    ax.xaxis.set(ticks=range(10))
    ax.yaxis.set(ticks=range(10))
    ax.set_ylim(9.5, -0.5)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, confusion[i, j], ha='center', va='center', color='white')
    plt.legend()
    plt.show()

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
    plt.legend()
    plt.show()

if __name__ == "__main__":
    single_variate_binary_classification()
    single_variate_binary_classification(10.0)
    single_variate_binary_classification_statsmodels()