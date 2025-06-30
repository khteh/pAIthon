import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.style.use('./data/deeplearning.mplstyle')

RANDOM_STATE = 55 ## We will pass it to every sklearn call so we ensure reproducibility

# https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download

class HeartFailurePrediction():
    _X_train = None
    _X_val = None
    _Y_train = None
    _Y_val = None
    _dt: DecisionTreeClassifier = None
    _rf: RandomForestClassifier = None
    _xgb: XGBClassifier = None
    def __init__(self, path):
        self.PrepareData(path)

    def PrepareData(self, path:str):
        """
        Remove the binary variables, because one-hot encoding them would do nothing to them. To achieve this we will just count how many different values there are in each categorical variable and consider only the variables with 3 or more values.
        one-hot encoding aims to transform a categorical variable with n outputs into n binary variables.

        Pandas has a built-in method to one-hot encode variables, it is the function pd.get_dummies. There are several arguments to this function, but here we will use only a few. They are:

        data: DataFrame to be used
        prefix: A list with prefixes, so we know which value we are dealing with
        columns: the list of columns that will be one-hot encoded. 'prefix' and 'columns' must have the same length.
        """
        # Load the dataset using pandas
        df = pd.read_csv(path)
        cat_variables = ['Sex',
            'ChestPainType',
            'RestingECG',
            'ExerciseAngina',
            'ST_Slope'
        ]
        # This will replace the columns with the one-hot encoded ones and keep the columns outside 'columns' argument as it is.
        df = pd.get_dummies(data = df, prefix = cat_variables, columns = cat_variables)
        features = [x for x in df.columns if x not in 'HeartDisease'] ## Removing our target variable
        # We will keep the shuffle = True since our dataset has not any time dependency.    
        self._X_train, self._X_val, self._Y_train, self._Y_val = train_test_split(df[features], df['HeartDisease'], train_size = 0.8, random_state = RANDOM_STATE)
        print(f'train samples: {len(self._X_train)}')
        print(f'validation samples: {len(self._X_val)}')
        print(f'target proportion: {sum(self._Y_train)/len(self._Y_train):.4f}')
    
    def BuildDecisionTreeModel(self):
        """
        There are several hyperparameters in the Decision Tree object from Scikit-learn. Only use some of them.

        The hyperparameters we will use and investigate here are:

        min_samples_split: The minimum number of samples required to split an internal node.
        Choosing a higher min_samples_split can reduce the number of splits and may help to reduce overfitting.
        max_depth: The maximum depth of the tree.
        Choosing a lower max_depth can reduce the number of splits and may help to reduce overfitting.

        Improvements:
        (1) perform feature selection
        (2) hyperparameter tuning
        """
        min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
        max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
        accuracy_list_train = []
        accuracy_list_val = []
        for min_samples_split in min_samples_split_list:
            # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
            self._dt = DecisionTreeClassifier(min_samples_split = min_samples_split,
                                        random_state = RANDOM_STATE).fit(self._X_train, self._Y_train) 
            predictions_train = self._dt.predict(self._X_train) ## The predicted values for the train dataset
            predictions_val = self._dt.predict(self._X_val) ## The predicted values for the test dataset
            accuracy_train = accuracy_score(predictions_train, self._Y_train)
            accuracy_val = accuracy_score(predictions_val, self._Y_val)
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)
        plt.title('Train x Validation metrics')
        plt.xlabel('min_samples_split')
        plt.ylabel('accuracy')
        plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
        plt.plot(accuracy_list_train)
        plt.plot(accuracy_list_val)
        plt.legend(['Train','Validation'])

        accuracy_list_train = []
        accuracy_list_val = []
        for max_depth in max_depth_list:
            # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
            self._dt = DecisionTreeClassifier(max_depth = max_depth,
                                        random_state = RANDOM_STATE).fit(self._X_train, self._Y_train) 
            predictions_train = self._dt.predict(self._X_train) ## The predicted values for the train dataset
            predictions_val = self._dt.predict(self._X_val) ## The predicted values for the test dataset
            accuracy_train = accuracy_score(predictions_train, self._Y_train)
            accuracy_val = accuracy_score(predictions_val, self._Y_val)
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)
        plt.title('Train x Validation metrics')
        plt.xlabel('max_depth')
        plt.ylabel('accuracy')
        plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
        plt.plot(accuracy_list_train)
        plt.plot(accuracy_list_val)
        plt.legend(['Train','Validation'])
        # Choose min_samples_split and max_depth based on the plots above which yield highest accuracy and lowest variance / overfitting, i.e., training and test accuracy should be as close to one another as possible
        self._dt = DecisionTreeClassifier(min_samples_split = 50,
                                             max_depth = 3,
                                             random_state = RANDOM_STATE).fit(self._X_train, self._Y_train)
        print(f"Metrics train:\n\tAccuracy score: {accuracy_score(self._dt.predict(self._X_train), self._Y_train):.4f}")
        print(f"Metrics validation:\n\tAccuracy score: {accuracy_score(self._dt.predict(self._X_val), self._Y_val):.4f}")

    def BuildRandomForestModel(self):
        """
        All of the hyperparameters found in the decision tree model will also exist in this algorithm, since a random forest is an ensemble of many Decision Trees.
        One additional hyperparameter for Random Forest is called n_estimators (default=100) which is the number of Decision Trees that make up the Random Forest.
        Remember that for a Random Forest, we randomly choose a subset of the features AND randomly choose a subset of the training examples to train each individual tree.

        if n is the number of features, we will randomly select sqrt(n) of these features to train each individual tree.
        set the max_features parameter to n

        n_jobs parameter can be used to speed up training jobs.

        Since the fitting of each tree is independent of each other, it is possible fit more than one tree in parallel.
        So setting n_jobs higher will increase how many CPU cores it will use. Note that the numbers very close to the maximum cores of your CPU may impact on the overall performance of your PC and even lead to freezes.
        Changing this parameter does not impact on the final result but can reduce the training time.        
        """
        min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]  ## If the number is an integer, then it is the actual quantity of samples,
                                                    ## If it is a float, then it is the percentage of the dataset
        max_depth_list = [2, 4, 8, 16, 32, 64, None]
        n_estimators_list = [10,50,100,500]
        accuracy_list_train = []
        accuracy_list_val = []
        for min_samples_split in min_samples_split_list:
            # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
            self._rf = RandomForestClassifier(min_samples_split = min_samples_split,
                                        random_state = RANDOM_STATE).fit(self._X_train, self._Y_train) 
            predictions_train = self._rf.predict(self._X_train) ## The predicted values for the train dataset
            predictions_val = self._rf.predict(self._X_val) ## The predicted values for the test dataset
            accuracy_train = accuracy_score(predictions_train, self._Y_train)
            accuracy_val = accuracy_score(predictions_val, self._Y_val)
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)

        plt.title('Train x Validation metrics')
        plt.xlabel('min_samples_split')
        plt.ylabel('accuracy')
        plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list) 
        plt.plot(accuracy_list_train)
        plt.plot(accuracy_list_val)
        plt.legend(['Train','Validation'])
        # Choose min_samples_split and max_depth based on the plots above which yield highest accuracy and lowest variance / overfitting, i.e., training and test accuracy should be as close to one another as possible
        self._rf = RandomForestClassifier(n_estimators = 100,
                                             max_depth = 16, 
                                             min_samples_split = 10).fit(self._X_train, self._Y_train)
        print(f"Metrics train:\n\tAccuracy score: {accuracy_score(self._rf.predict(self._X_train), self._Y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(self._rf.predict(self._X_val), self._Y_val):.4f}")

    def BuildXGBoost(self):
        """
        Gradient Boosting model, called XGBoost. The boosting methods train several trees, but instead of them being uncorrelated to each other, now the trees are fit one after the other in order to minimize the error.

        The model has the same parameters as a decision tree, plus the learning rate.

        The learning rate is the size of the step on the Gradient Descent method that the XGBoost uses internally to minimize the error on each train step.
        One interesting thing about the XGBoost is that during fitting, it can take in an evaluation dataset of the form (X_val,y_val).

        On each iteration, it measures the cost (or evaluation metric) on the evaluation datasets.
        Once the cost (or metric) stops decreasing for a number of rounds (called early_stopping_rounds), the training will stop.
        More iterations lead to more estimators, and more estimators can result in overfitting.
        By stopping once the validation metric no longer improves, we can limit the number of estimators created, and reduce overfitting.        
        """
        # define a subset of our training set (we should not use the test set here).
        n = int(len(self._X_train)*0.8) ## Let's use 80% to train and 20% to eval
        X_train_fit, X_train_eval, y_train_fit, y_train_eval = self._X_train[:n], self._X_train[n:], self._Y_train[:n], self._Y_train[n:]
        """
        We can then set a large number of estimators, because we can stop if the cost function stops decreasing.

        Note some of the .fit() parameters:

        eval_set = [(X_train_eval,y_train_eval)]:Here we must pass a list to the eval_set, because you can have several different tuples ov eval sets.
        early_stopping_rounds: This parameter helps to stop the model training if its evaluation metric is no longer improving on the validation set. It's set to 10.
        The model keeps track of the round with the best performance (lowest evaluation metric). For example, let's say round 16 has the lowest evaluation metric so far.
        Each successive round's evaluation metric is compared to the best metric. If the model goes 10 rounds where none have a better metric than the best one, then the model stops training.
        The model is returned at its last state when training terminated, not its state during the best round. For example, if the model stops at round 26, but the best round was 16, the model's training state at round 26 is returned, not round 16.
        Note that this is different from returning the model's "best" state (from when the evaluation metric was the lowest).
        """
        self._xgb = XGBClassifier(n_estimators = 500, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE)
        self._xgb.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)], early_stopping_rounds = 10)
        print(f"Best iteration with lowest evaluation metric: {self._xgb.best_iteration}")
        print(f"Metrics train:\n\tAccuracy score: {accuracy_score(self._xgb.predict(self._X_train), self._Y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(self._xgb.predict(self._X_val), self._Y_val):.4f}")

if __name__ == "__main__":
    heart = HeartFailurePrediction("data/heart.csv") # https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download
    heart.BuildDecisionTreeModel()
    heart.BuildRandomForestModel()
    heart.BuildXGBoost()