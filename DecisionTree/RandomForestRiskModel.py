import argparse, shap, sklearn, itertools, lifelines, pydotplus, numpy, pandas as pd, seaborn as sb, matplotlib.pyplot as plt, pickle
from pathlib import Path
from sklearn.tree import export_graphviz
from six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from xgboost import XGBClassifier
from utils.DecisionTree import PlotDecisionTree

class RandomForestRiskModel():
    _threshold:float = None
    _path:str = None
    _X_train = None
    _X_val = None
    _X_test = None
    _Y_train = None
    _Y_val = None
    _Y_test = None
    _best_hyperparams = None
    _imputer_iterations: int = None
    _imputer: IterativeImputer = None
    _rf: RandomForestClassifier = None
    _xgb: XGBClassifier = None
    def __init__(self, path:str, threshold:float, imputer_iterations:int):
        self._path = path
        self._imputer_iterations = imputer_iterations
        self._threshold = threshold
        self._PrepareData()
        if self._path and len(self._path) and Path(self._path).exists() and Path(self._path).is_file():
            print(f"Using saved model {self._path}...")
            with open(self._path, 'rb') as file:
                self._rf = pickle.load(file)
            self._trained = True

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
        print(f"\n=== {self.BuildRandomForestModel.__name__} ===")
        if not self._rf:
            self._random_forest_grid_search()
        # Best hyperparameters:
        #{'n_estimators': 456, 'max_depth': 13, 'min_samples_leaf': 5, 'random_state': 10}
        #Train C-Index: 0.9582820140564089
        #Val C-Index: 0.7794067462921643
        print(f"classes: {self._rf.classes_}") # classes: [False  True]
        cindex, subgroup = self._bad_subset(self._X_train, self._Y_train)
        print(f"Train dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        cindex, subgroup = self._bad_subset(self._X_val, self._Y_val)
        print(f"Validation dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        cindex, subgroup = self._bad_subset(self._X_test, self._Y_test)
        print(f"Test dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        #PlotDecisionTree(self._rf, self._X_test.columns, ['neg', 'pos'], "HeartDiseasePredictionDecisionTree") # AttributeError: 'RandomForestClassifier' object has no attribute 'tree_'

    def _random_forest_grid_search(self):
        print(f"\n=== {self._random_forest_grid_search.__name__} ===")
        # Define ranges for the chosen random forest hyperparameters 
        hyperparams = {
            
            # how many trees should be in the forest (int)
            'n_estimators': [456, 789],

            # the maximum depth of trees in the forest (int)
            'max_depth': [11,13,15],
            
            # the minimum number of samples in a leaf as a fraction
            # of the total number of samples in the training set
            # Can be int (in which case that is the minimum number)
            # or float (in which case the minimum is that fraction of the
            # number of training set samples)
            'min_samples_leaf': [3,5,7,9],
        }
        fixed_hyperparams = {
            'random_state': 10,
        }
        rf = RandomForestClassifier
        best_rf, best_hyperparams = self._holdout_grid_search(rf, self._X_train, self._Y_train,
                                                        self._X_val, self._Y_val, hyperparams,
                                                        fixed_hyperparams)
        print(f"Best hyperparameters:\n{best_hyperparams}")
        y_train_best = best_rf.predict_proba(self._X_train)[:, 1]
        print(f"Train C-Index: {self._cindex(self._Y_train, y_train_best)}")

        y_val_best = best_rf.predict_proba(self._X_val)[:, 1]
        print(f"Val C-Index: {self._cindex(self._Y_val, y_val_best)}")
        
        # add fixed hyperparamters to best combination of variable hyperparameters
        best_hyperparams.update(fixed_hyperparams)
        self._rf = best_rf
        self._best_hyperparams = best_hyperparams
        if self._path:
            with open(self._path, 'wb') as f:
                pickle.dump(self._rf, f)
            print(f"Model saved to {self._path}.")

    def _holdout_grid_search(self, rf, X_train_hp, y_train_hp, X_val_hp, y_val_hp, hyperparams, fixed_hyperparams={}):
        '''
        Conduct hyperparameter grid search on hold out validation set. Use holdout validation.
        Hyperparameters are input as a dictionary mapping each hyperparameter name to the
        range of values they should iterate over. Use the cindex function as your evaluation
        function.

        Input:
            clf: sklearn classifier
            X_train_hp (dataframe): dataframe for training set input variables
            y_train_hp (dataframe): dataframe for training set targets
            X_val_hp (dataframe): dataframe for validation set input variables
            y_val_hp (dataframe): dataframe for validation set targets
            hyperparams (dict): hyperparameter dictionary mapping hyperparameter
                                names to range of values for grid search
            fixed_hyperparams (dict): dictionary of fixed hyperparameters that
                                    are not included in the grid search

        Output:
            best_estimator (sklearn classifier): fitted sklearn classifier with best performance on
                                                validation set
            best_hyperparams (dict): hyperparameter dictionary mapping hyperparameter
                                    names to values in best_estimator
        '''
        best_estimator = None
        best_hyperparams = {}
        
        # hold best running score
        best_score = 0.0

        # get list of param values
        lists = hyperparams.values()
        
        # get all param combinations
        param_combinations = list(itertools.product(*lists))
        total_param_combinations = len(param_combinations)

        # iterate through param combinations
        for i, params in enumerate(param_combinations, 1):
            # fill param dict with params
            param_dict = {}
            for param_index, param_name in enumerate(hyperparams):
                param_dict[param_name] = params[param_index]
                
            # create estimator with specified params
            estimator = rf(**param_dict, **fixed_hyperparams)

            # fit estimator
            estimator.fit(X_train_hp, y_train_hp)
            
            # get predictions on validation set
            preds = estimator.predict_proba(X_val_hp)
            #print(f"prediction: shape: {preds.shape} {preds}") # Matches with self._rf.classes_: [probablity of NOT dying within 10 years, probablity of dying within 10 years]
            # compute cindex for predictions
            estimator_score = self._cindex(y_val_hp, preds[:,1])

            print(f'[{i}/{total_param_combinations}] {param_dict}')
            print(f'Val C-Index: {estimator_score}\n')

            # if new high score, update high score, best estimator
            # and best params 
            if estimator_score >= best_score:
                    best_score = estimator_score
                    best_estimator = estimator
                    best_hyperparams = param_dict

        # add fixed hyperparamters to best combination of variable hyperparameters
        best_hyperparams.update(fixed_hyperparams)
        return best_estimator, best_hyperparams
    
    def _PrepareData(self):
        """
        NHANES I epidemiology dataset. This dataset contains various features of hospital patients as well as their outcomes, i.e. whether or not they died within 10 years (self._threshold).
        """
        print(f"\n=== {self._PrepareData.__name__} ===")
        X = pd.read_csv("data/NHANESI_subset_X.csv") # (9932, 19)
        y = pd.read_csv("data/NHANESI_subset_y.csv")["y"]
        print(f"X: {X.shape}, Y: {y.shape}")
        print(f"X columns: {X.columns}")
        df = X.drop([X.columns[0]], axis=1)
        df.loc[:, 'time'] = y
        df.loc[:, 'death'] = numpy.ones(len(X))
        df.loc[df.time < 0, 'death'] = 0
        df.loc[:, 'time'] = numpy.abs(df.time)
        mask = (df.time > self._threshold) | (df.death == 1)
        df = df[mask]
        X = df.drop(['time', 'death'], axis='columns')
        y = df.time < self._threshold
        print(f"X columns: {X.columns}")
        print("X:")
        print(X.head())
        print("Y:")
        print(y.head())

        # Get 80% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
        self._X_train, x_, self._Y_train, y_ = train_test_split(X, y, test_size=0.20, random_state=1)

        # Split the 40% subset above into two: one half for cross validation and the other for the test set
        self._X_val, self._X_test, self._Y_val, self._Y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

        # Delete temporary variables
        del x_, y_

        print(f"X_train: {self._X_train.shape}, Y_train: {self._Y_train.shape}, X_val: {self._X_val.shape}, Y_val: {self._Y_val.shape}, X_test: {self._X_test.shape}, Y_test: {self._Y_test.shape}")
        # Impute using regression on other covariates
        self._imputer = IterativeImputer(random_state=0, sample_posterior=False, max_iter=self._imputer_iterations, min_value=0)
        self._imputer.fit(self._X_train)
        self._X_train = pd.DataFrame(self._imputer.transform(self._X_train), columns=self._X_train.columns, index=self._X_train.index)
        self._X_val = pd.DataFrame(self._imputer.transform(self._X_val), columns=self._X_val.columns, index=self._X_val.index)
        self._X_test = pd.DataFrame(self._imputer.transform(self._X_test), columns=self._X_test.columns, index=self._X_test.index)

    def _bad_subset(self, X, Y):
        # define mask to select large subset with poor performance
        # currently mask defines the entire set
        print(f"\n=== {self._bad_subset.__name__} ===")
        mask = X["Age"] > 65
        #print(f"X: {X.shape}, Y: {Y.shape}, mask: {mask}")
        X_subgroup = X[mask]
        y_subgroup = Y[mask]
        subgroup_size = len(X_subgroup)

        y_subgroup_preds = self._rf.predict_proba(X_subgroup)[:, 1]
        performance = self._cindex(y_subgroup.values, y_subgroup_preds)
        return performance, subgroup_size
    
    def _cindex(self, y_true, scores):
        """
        C-index = (#concordance + 0.5 * #risk_ties) / #permissible
        - The C-index measures the discriminatory power of a risk score.
        - Intuitively, a higher c-index indicates that the model's prediction is in agreement with the actual outcomes of a pair of patients.
        """
        return lifelines.utils.concordance_index(y_true, scores)

    def _prob_drop(self, age):
        return 1 - (numpy.exp(0.25 * age - 5) / (1 + numpy.exp(0.25 * age - 5)))

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Resnet Signs Language multi-class classifier')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    parser.add_argument('-g', '--grayscale', action='store_true', help='Use grayscale model')
    args = parser.parse_args()

    risk = RandomForestRiskModel(f"models/RandomForestRiskModel.pkl", 10, 10)
    risk.BuildRandomForestModel()