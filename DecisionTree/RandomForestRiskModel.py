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
from .DecisionTree import DecisionTree
from utils.CIndex import cindex
class RandomForestRiskModel(DecisionTree):
    """
    Tree based models by predicting the 10-year risk of death of individuals from the NHANES | epidemiology dataset (https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx/)
    """
    _threshold:float = None
    _imputer_iterations: int = None
    _imputer: IterativeImputer = None
    _best_hyperparams = None
    _rf: RandomForestClassifier = None
    _xgb: XGBClassifier = None
    _shap = None
    def __init__(self, path:str, threshold:float, imputer_iterations:int):
        super().__init__(path)
        self._imputer_iterations = imputer_iterations
        self._threshold = threshold
        self._PrepareData()
        self._rf = self._LoadModel()
        if self._rf is not None:
            self._shap = shap.TreeExplainer(self._rf)

    def BuildRandomForestModel(self, retrain:bool = False):
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
        if not self._rf or retrain:
            hyperparams = {
                
                # how many trees should be in the forest (int)
                'n_estimators': [456, 789],

                # the maximum depth of trees in the forest (int). If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                'max_depth': [11,13,15, None],
                
                # the minimum number of samples in a leaf as a fraction
                # of the total number of samples in the training set
                # Can be int (in which case that is the minimum number)
                # or float (in which case the minimum is that fraction of the
                # number of training set samples)
                'min_samples_leaf': [1,3,5,7,9],
            }
            fixed_hyperparams = {
                'random_state': 10,
            }
            self._rf, self._best_hyperparams = self._random_forest_grid_search(RandomForestClassifier, self._X_train, self._Y_train, self._X_val, self._Y_val, hyperparams, fixed_hyperparams)
            self._shap = shap.TreeExplainer(self._rf)
        # Best hyperparameters:
        # {'n_estimators': 456, 'max_depth': None, 'min_samples_leaf': 5, 'random_state': 10}
        # Train C-Index: 0.9872592913256227
        # Val C-Index: 0.7800695629347684
        # classes: [False  True]
        print(f"Best hyperparameters:\n{self._best_hyperparams}")
        y_train_best = self._rf.predict_proba(self._X_train)[:, 1]
        print(f"Train C-Index: {self._cindex(self._Y_train, y_train_best)}")

        y_val_best = self._rf.predict_proba(self._X_val)[:, 1]
        print(f"Val C-Index: {self._cindex(self._Y_val, y_val_best)}")
        print(f"classes: {self._rf.classes_}") # classes: [False  True]
        cindex, subgroup = self._bad_subset(self._X_train, self._Y_train)
        print(f"Train dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        cindex, subgroup = self._bad_subset(self._X_val, self._Y_val)
        print(f"Validation dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        cindex, subgroup = self._bad_subset(self._X_test, self._Y_test)
        print(f"Test dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        #PlotDecisionTree(self._rf, self._X_test.columns, ['neg', 'pos'], "HeartDiseasePredictionDecisionTree") # AttributeError: 'RandomForestClassifier' object has no attribute 'tree_'

    def ExplainModelPrediction(self):
        """
        You choose to apply **SHAP (SHapley Additive exPlanations)**, a cutting edge method that explains predictions made by black-box machine learning models (i.e. models which are too complex to be understandable by humans as is).
        Given a prediction made by a machine learning model, SHAP values explain the prediction by quantifying the additive importance of each feature to the prediction. SHAP values have their roots in cooperative game theory, where Shapley values are used to quantify the contribution of each player to the game.
        Although it is computationally expensive to compute SHAP values for general black-box models, in the case of trees and forests there exists a fast polynomial-time algorithm. For more details, see the [TreeShap paper](https://arxiv.org/pdf/1802.03888.pdf).
        Use the [shap library](https://github.com/slundberg/shap) to do this for our random forest model.

        How to read this chart:
        - The red sections on the left are features which push the model towards the final prediction in the positive direction (i.e. a higher Age increases the predicted risk).
        - The blue sections on the right are features that push the model towards the final prediction in the negative direction (if an increase in a feature leads to a lower risk, it will be shown in blue).
        - Note that the exact output of your chart will differ depending on the hyper-parameters that you choose for your model.

        """
        print(f"\n=== {self.ExplainModelPrediction.__name__} ===")
        i = 0
        if not self._X_test_risk:
            self._X_test_risk = self._X_test.copy(deep=True)
            self._X_test_risk.loc[:, 'risk'] = self._rf.predict_proba(self._X_test_risk)[:, 1]
            self._X_test_risk = self._X_test_risk.sort_values(by='risk', ascending=False)
            #print(self._X_test_risk.head())
        print(f"self._X_test_risk.index[i]: {self._X_test_risk.index[i]}")
        shap_values = self._shap.shap_values(self._X_test.loc[self._X_test_risk.index[i], :])
        shap_value = shap_values[:,1]
        print(f"shap_values: {shap_values.shape}, {shap_values}")
        print(f"shap_value: {shap_value.shape}, {shap_value}")
        shap.force_plot(self._shap.expected_value[1], shap_value, feature_names=self._X_test.columns, matplotlib=True, figsize=(20, 10))
        shap_values = self._shap.shap_values(self._X_test)[:,:,1] # (992, 18, 2),
        print(f"shap_values: {shap_values.shape}, {shap_values}")
        shap.summary_plot(shap_values, self._X_test)
        shap.dependence_plot('Age', shap_values, self._X_test, interaction_index='Sex')
        shap.dependence_plot('Poverty index', shap_values, self._X_test, interaction_index='Age')
   
    def _PrepareData(self):
        """
        NHANES | epidemiology dataset. This dataset contains various features of hospital patients as well as their outcomes, i.e. whether or not they died within 10 years (self._threshold).
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
    
    def _Evaluate(self, Y, predictions):
        return cindex(Y, predictions[:,1])
    
    def _prob_drop(self, age):
        return 1 - (numpy.exp(0.25 * age - 5) / (1 + numpy.exp(0.25 * age - 5)))

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Random Forest Risk Model')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    parser.add_argument('-g', '--grayscale', action='store_true', help='Use grayscale model')
    args = parser.parse_args()

    risk = RandomForestRiskModel("models/RandomForestRiskModel.pkl", 10, 10)
    risk.BuildRandomForestModel(args.retrain)
    risk.ExplainModelPrediction()