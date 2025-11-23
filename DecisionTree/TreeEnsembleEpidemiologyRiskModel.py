import argparse, shap, sklearn, itertools, lifelines, pydotplus, numpy, pandas as pd, seaborn as sb, matplotlib.pyplot as plt, pickle
from pathlib import Path
from sklearn.tree import export_graphviz
from six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from xgboost import XGBClassifier, DMatrix
from utils.DecisionTreeViz import PlotDecisionTree
from .DecisionTree import DecisionTree
from utils.CIndex import CIndex
class TreeEnsembleEpidemiologyRiskModel(DecisionTree):
    """
    Tree based models by predicting the 10-year risk of death of individuals from the NHANES | epidemiology dataset (https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx/)
    """
    _threshold:float = None
    _imputer_iterations: int = None
    _imputer: IterativeImputer = None
    _rf: RandomForestClassifier = None
    _xgb: XGBClassifier = None
    _shap = None
    def __init__(self, path:str, threshold:float, imputer_iterations:int):
        self._imputer_iterations = imputer_iterations
        self._threshold = threshold
        self._PrepareData()

    def BuildRandomForestModel(self, model_path:str, retrain:bool = False):
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
        self._model_path = model_path
        self._rf = self._LoadModel()
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
            self._rf = self._random_forest_grid_search(RandomForestClassifier, self._X_train, self._Y_train, self._X_val, self._Y_val, hyperparams, fixed_hyperparams)
            # Best hyperparameters:
            # {'n_estimators': 456, 'max_depth': None, 'min_samples_leaf': 5, 'random_state': 10}
            # Train C-Index: 0.9872592913256227
            # Val C-Index: 0.7800695629347684
            # classes: [False  True]
            print(f"Best hyperparameters:\n{self._best_hyperparams}")
            self._shap = shap.TreeExplainer(self._rf)

        y_train_best = self._rf.predict_proba(self._X_train)[:, 1]
        print(f"Train C-Index: {CIndex(self._Y_train, y_train_best)}")

        y_val_best = self._rf.predict_proba(self._X_val)[:, 1]
        print(f"Val C-Index: {CIndex(self._Y_val, y_val_best)}")
        print(f"classes: {self._rf.classes_}") # classes: [False  True]
        cindex, subgroup = self._bad_subset(self._rf, self._X_train, self._Y_train)
        print(f"Train dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        cindex, subgroup = self._bad_subset(self._rf, self._X_val, self._Y_val)
        print(f"Validation dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        cindex, subgroup = self._bad_subset(self._rf, self._X_test, self._Y_test)
        print(f"Test dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        #PlotDecisionTree(self._rf, self._X_test.columns, ['neg', 'pos'], "HeartDiseasePredictionDecisionTree") # AttributeError: 'RandomForestClassifier' object has no attribute 'tree_'
        self._ExplainModelPrediction(self._rf)

    def BuildXGBoost(self, model_path:str, retrain:bool = False):
        """
        Gradient Boosting model, called XGBoost. The boosting methods train several trees, but instead of them being uncorrelated to each other, now the trees are fit one after the other in order to minimize the error.

        The model has the same parameters as a decision tree, plus the learning rate.

        The learning rate is the size of the step on the Gradient Descent method that the XGBoost uses internally to minimize the error on each train step.
        One interesting thing about the XGBoost is that during fitting, it can take in an evaluation dataset of the form (X_val,y_val).

        On each iteration, it measures the cost (or evaluation metric) on the evaluation datasets.
        Once the cost (or metric) stops decreasing for a number of rounds (called early_stopping_rounds), the training will stop.
        More iterations lead to more estimators, and more estimators can result in overfitting.
        By stopping once the validation metric no longer improves, we can limit the number of estimators created, and reduce overfitting.

        We can then set a large number of estimators, because we can stop if the cost function stops decreasing.

        Note some of the .fit() parameters:

        eval_set = [(X_train_eval,y_train_eval)]:Here we must pass a list to the eval_set, because you can have several different tuples ov eval sets.
        early_stopping_rounds: This parameter helps to stop the model training if its evaluation metric is no longer improving on the validation set. It's set to 10.
        The model keeps track of the round with the best performance (lowest evaluation metric). For example, let's say round 16 has the lowest evaluation metric so far.
        Each successive round's evaluation metric is compared to the best metric. If the model goes 10 rounds where none have a better metric than the best one, then the model stops training.
        The model is returned at its last state when training terminated, not its state during the best round. For example, if the model stops at round 26, but the best round was 16, the model's training state at round 26 is returned, not round 16.
        Note that this is different from returning the model's "best" state (from when the evaluation metric was the lowest).
        """
        print(f"\n=== {self.BuildXGBoost.__name__} ===")
        self._model_path = model_path
        self._xgb = self._LoadModel()
        # define a subset of our training set (we should not use the test set here).
        #n = int(len(self._X_train)*0.8) ## Let's use 80% to train and 20% to eval
        #X_train_fit, X_train_eval, y_train_fit, y_train_eval = self._X_train[:n], self._X_train[n:], self._Y_train[:n], self._Y_train[n:]
        if not self._xgb or retrain:
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
                'min_samples_leaf': [3,5,7,9],
            }
            fixed_hyperparams = {
                'random_state': 11,
                "early_stopping_rounds": 10
            }
            self._xgb = self._random_forest_grid_search(XGBClassifier, self._X_train, self._Y_train, self._X_val, self._Y_test, hyperparams, fixed_hyperparams)
            print(f"Best hyperparameters:\n{self._best_hyperparams}")
        y_train_best = self._xgb.predict_proba(self._X_train)[:, 1]
        print(f"Train C-Index: {CIndex(self._Y_train, y_train_best)}")

        y_val_best = self._xgb.predict_proba(self._X_val)[:, 1]
        print(f"Val C-Index: {CIndex(self._Y_test, y_val_best)}")
        print(f"classes: {self._xgb.classes_}") # classes: [False  True]
        cindex, subgroup = self._bad_subset(self._xgb, self._X_train, self._Y_train)
        print(f"Train dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        cindex, subgroup = self._bad_subset(self._xgb, self._X_val, self._Y_val)
        print(f"Validation dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        cindex, subgroup = self._bad_subset(self._xgb, self._X_test, self._Y_test)
        print(f"Test dataset Subgroup size: {subgroup}, C-Index: {cindex}")
        #PlotDecisionTree(self._xgb, self._features, ['neg', 'pos'], "XGBoostHeartDiseasePrediction") AttributeError: 'XGBClassifier' object has no attribute 'tree_'
        self._ExplainModelPrediction(self._xgb)

    def _ExplainModelPrediction(self, model):
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
        print(f"\n=== {self._ExplainModelPrediction.__name__} ===")
        i = 0
        if self._X_test_risk is None:
            self._X_test_risk = self._X_test.copy(deep=True)
            self._Y_test_risk = self._Y_test.copy(deep=True)
            self._X_test_risk.loc[:, 'risk'] = model.predict_proba(self._X_test_risk)[:, 1]
            self._X_test_risk = self._X_test_risk.sort_values(by='risk', ascending=False)
            self._Y_test_risk = self._Y_test_risk.reindex(self._X_test_risk.index)
            #print(self._X_test_risk.head())
        print(f"self._X_test_risk.index[{i}]: {self._X_test_risk.index[i]}")
        print(f"self._Y_test_risk.index[{i}]: {self._Y_test_risk.index[i]}")
        print(self._X_test_risk.head())
        self._shap = shap.TreeExplainer(model)
        X = self._X_test.loc[self._X_test_risk.index[i], :]
        Y = self._Y_test.loc[self._Y_test_risk.index[i]]
        print(f"X: {X}")
        print(f"Y: {Y}")
        print(f"index: {self._X_test_risk.index[i]}, X: {X.shape}, Y: {Y.shape}")
        # https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
        if isinstance(model, XGBClassifier):
            # Avoid pandas Series which will have the original DF columns as its index and the values of that specific row as its data. 
            # The "name" column is actually the name attribute of the resulting Series, which automatically gets assigned the index label used to retrieve the row.
            # ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, the experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:371: object
            X = self._X_test.loc[self._X_test_risk.index[i], :].to_numpy()
            X = X[numpy.newaxis, ...] # Add a single grayacale channel
            Y = Y[numpy.newaxis, ...] # Add a single grayacale channel
            dmatrix = DMatrix(data=X, label=Y, enable_categorical=False)
            #dmatrix = DMatrix(data=X, enable_categorical=True)
            print(f"dmatrix: {dmatrix}")
            shap_values = self._shap.shap_values(dmatrix)
            shap_value = shap_values[0]
            print(f"shap_values: {shap_values.shape}, {shap_values}")
            print(f"shap_value: {shap_value.shape}, {shap_value}")
            shap.force_plot(self._shap.expected_value, shap_value, feature_names=self._X_test.columns, matplotlib=True, figsize=(20, 10))
        else:
            shap_values = self._shap.shap_values(X)
            shap_value = shap_values[:,1]
            print(f"shap_values: {shap_values.shape}, {shap_values}")
            print(f"shap_value: {shap_value.shape}, {shap_value}")
            shap.force_plot(self._shap.expected_value[1], shap_value, feature_names=self._X_test.columns, matplotlib=True, figsize=(20, 10))
        if isinstance(model, XGBClassifier):
            test_data_dm = DMatrix(data = self._X_test, label = self._Y_test, enable_categorical=False)
            shap_values = self._shap.shap_values(test_data_dm)
        else:
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

    def _bad_subset(self, model, X, Y):
        # define mask to select large subset with poor performance
        # currently mask defines the entire set
        print(f"\n=== {self._bad_subset.__name__} ===")
        mask = X["Age"] > 65
        #print(f"X: {X.shape}, Y: {Y.shape}, mask: {mask}")
        X_subgroup = X[mask]
        y_subgroup = Y[mask]
        subgroup_size = len(X_subgroup)

        y_subgroup_preds = model.predict_proba(X_subgroup)[:, 1]
        performance = CIndex(y_subgroup.values, y_subgroup_preds)
        return performance, subgroup_size
    
    def _Evaluate(self, Y, probabilities):
        return CIndex(Y, probabilities[:,1])
    
    def _prob_drop(self, age):
        return 1 - (numpy.exp(0.25 * age - 5) / (1 + numpy.exp(0.25 * age - 5)))

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Random Forest Risk Model')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    risk = TreeEnsembleEpidemiologyRiskModel(None, 10, 10)
    risk.BuildRandomForestModel("models/RandomForestEpidemiologyRiskModel.pkl", args.retrain)
    risk.BuildXGBoost("models/XGBoostEpidemiologyRiskModel.pkl", args.retrain)