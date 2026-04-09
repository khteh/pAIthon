import pickle, itertools, shap, numpy, pandas as pd
from abc import ABC
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, DMatrix
class DecisionTree(ABC):
    _model_path:str = None
    _X_train = None
    _X_val = None
    _X_test = None
    _X_test_risk = None
    _Y_train = None
    _Y_val = None
    _Y_test = None
    _best_hyperparams = None
    _rf: RandomForestClassifier = None
    _xgb: XGBClassifier = None
    _shap = None

    def __init__(self, path:str):
        self._model_path = path

    def _LoadModel(self, model_path:str = None):
        if model_path:
            self._model_path = model_path
        if self._model_path and len(self._model_path) and Path(self._model_path).exists() and Path(self._model_path).is_file():
            print(f"Using saved model {self._model_path}...")
            with open(self._model_path, 'rb') as file:
                return pickle.load(file)

    def _random_forest_grid_search(self, model, X_train, Y_train, X_val, Y_val, hyperparams:dict, fixed_hyperparams:dict, model_path:str = None):
        print(f"\n=== {self._random_forest_grid_search.__name__} ===")
        # Define ranges for the chosen random forest hyperparameters 
        best_model, self._best_hyperparams = self._holdout_grid_search(model, X_train, Y_train, X_val, Y_val, hyperparams, fixed_hyperparams)
               
        # add fixed hyperparamters to best combination of variable hyperparameters
        self._best_hyperparams.update(fixed_hyperparams)
        if model_path:
            self._model_path = model_path
        if self._model_path:
            with open(self._model_path, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"Model saved to {self._model_path}.")
        return best_model

    def _holdout_grid_search(self, model, X_train, Y_train, X_val, Y_val, hyperparams:dict, fixed_hyperparams:dict):
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
        for i, params in enumerate(tqdm(param_combinations), 1):
            # fill param dict with params
            param_dict = {}
            for param_index, param_name in enumerate(hyperparams):
                param_dict[param_name] = params[param_index]

            # create estimator with specified params
            estimator = model(**param_dict, **fixed_hyperparams)

            # fit estimator
            if "early_stopping_rounds" in fixed_hyperparams:
                estimator.fit(X_train, Y_train, eval_set = [(X_val, Y_val)])
            else:
                estimator.fit(X_train, Y_train)
            
            # get predictions on validation set
            preds = estimator.predict_proba(X_val)
            #print(f"prediction: shape: {preds.shape} {preds}") # Matches with self._rf.classes_: [probablity of NOT dying within 10 years, probablity of dying within 10 years]
            # compute cindex for predictions
            estimator_score = self._Evaluate(Y_val, preds)

            print(f'[{i}/{total_param_combinations}] {param_dict}')
            print(f'Val score: {estimator_score}\n')

            # if new high score, update high score, best estimator
            # and best params 
            if estimator_score >= best_score:
                    best_score = estimator_score
                    best_estimator = estimator
                    best_hyperparams = param_dict

        # add fixed hyperparamters to best combination of variable hyperparameters
        best_hyperparams.update(fixed_hyperparams)
        return best_estimator, best_hyperparams

    def _Evaluate(self, predictions):
        pass

    def _ExplainXGBoostPrediction(self, dependencies):
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
        print(f"\n=== {self._ExplainXGBoostPrediction.__name__} ===")
        # features: ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N', 'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']
        if self._xgb is None:
            raise Exception(f"Invalid XGBClassifier!")
        i = 0       
        if self._X_test_risk is None:
            self._X_test_risk = self._X_test.copy(deep=True)
            self._Y_test_risk = self._Y_test.copy(deep=True)
            self._X_test_risk.loc[:, 'risk'] = self._xgb.predict_proba(self._X_test_risk)[:, 1]
            self._X_test_risk = self._X_test_risk.sort_values(by='risk', ascending=False)
            self._Y_test_risk = self._Y_test_risk.reindex(self._X_test_risk.index)
        print(f"X_test: {type(self._X_test)} {self._X_test.shape}") # (92, 20)
        print(f"self._X_test_risk.index[{i}]: {self._X_test_risk.index[i]}")
        print(f"self._Y_test_risk.index[{i}]: {self._Y_test_risk.index[i]}")
        print(f"_X_test_risk:\n{self._X_test_risk.head()}")
        prediction = self._xgb.predict_proba(self._X_test) # This needs to be done before adding the "risk" column below
        print(f"prediction: {prediction.shape}")
        # https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
        # https://github.com/shap/shap/issues/4225
        # https://github.com/shap/shap/pull/4254
        # X: A matrix of samples (# samples x # features) on which to explain the model’s output.
        # Avoid pandas Series which will have the original DF columns as its index and the values of that specific row as its data. 
        # The "name" column is actually the name attribute of the resulting Series, which automatically gets assigned the index label used to retrieve the row.
        # https://github.com/shap/shap/issues/4214
        # https://github.com/shap/shap/issues/4224
        X = self._X_test.loc[self._X_test_risk.index[i], :].to_numpy() # This also works
        X = X[numpy.newaxis, ...] # Add axis-0 as samples  # This also works
        Y = self._Y_test.loc[self._Y_test_risk.index[i]]
        Y = Y[numpy.newaxis, ...] # Add axis-0 as samples # This also works
        # enable_categorical:
        # If you pass a DataFrame (like Pandas, Polars, or cuDF) that contains columns explicitly typed as category, XGBoost will raise a ValueError. To use these columns without manual encoding, you must set the parameter to True.
        # When set to False, you are responsible for converting categorical data into numeric formats (such as one-hot encoding or label encoding) before passing the data into the DMatrix.
        dmatrix = DMatrix(data=X, label=Y, enable_categorical=True) # This also works.
        dmatrix = DMatrix(data=X, enable_categorical=True) # (#samples, #features) This also works
        print(f"dmatrix: ({dmatrix.num_row()}, {dmatrix.num_col()})") # (1, 20)
        shap_values = self._shap.shap_values(dmatrix) # This also works
        #shap_values = self._shap.shap_values(X)  # This also works. X must be a Dataframe which has proper dtype info.
        shap_value = shap_values[0]
        print(f"shap_values: {shap_values.shape}, {shap_values}") # shap_values: (1, 20)
        print(f"shap_value: {shap_value.shape}, {shap_value}") # shap_value: (20,)
        print(f"expected_value: {self._shap.expected_value.shape}, {self._shap.expected_value}") # expected_value: scalar
        #assert (shap_values[0, :].sum() + self._shap.expected_value == prediction[0]).all(), f"{shap_values[0, :].sum()} + {self._shap.expected_value} = {shap_values[0, :].sum() + self._shap.expected_value} != {prediction[0]}"
        # shap.plots.force(base_value, shap_values=None, features=None, feature_names=None, out_names=None, link='identity', plot_cmap='RdBu', matplotlib=False, show=True, figsize=(20, 3), ordering_keys=None, ordering_keys_time_format=None, text_rotation=0, contribution_threshold=0.05)
        shap.force_plot(self._shap.expected_value, shap_value, feature_names=self._X_test.columns, matplotlib=True, figsize=(20, 10))
        test_data_dm = DMatrix(data = self._X_test, label = self._Y_test, enable_categorical=True)
        shap_values = self._shap.shap_values(test_data_dm) # shap_values: (92, 20)
        #assert (shap_values[0, :].sum() + self._shap.expected_value == prediction[0]).all(), f"{shap_values[0, :].sum()} + {self._shap.expected_value} = {shap_values[0, :].sum() + self._shap.expected_value} != {prediction[0]}"            
        print(f"Summary Plot shap_values: {shap_values.shape}, {shap_values}") # (92, 20)
        shap.summary_plot(shap_values, self._X_test)
        if dependencies is not None and len(dependencies) > 0:
            for dep in dependencies:
                print(f"dependence_plot: {dep[0]} {dep[1]}")
                shap.dependence_plot(dep[0], shap_values, self._X_test, interaction_index=dep[1])

    def _ExplainRandomForestPrediction(self, dependencies):
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
        print(f"\n=== {self._ExplainRandomForestPrediction.__name__} ===")
        # features: ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N', 'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']
        if self._rf is None:
            raise Exception(f"Invalid RandomForestClassifier!")
        i = 0
        if self._X_test_risk is None:
            self._X_test_risk = self._X_test.copy(deep=True)
            self._Y_test_risk = self._Y_test.copy(deep=True)
            self._X_test_risk.loc[:, 'risk'] = self._rf.predict_proba(self._X_test_risk)[:, 1]
            self._X_test_risk = self._X_test_risk.sort_values(by='risk', ascending=False)
            self._Y_test_risk = self._Y_test_risk.reindex(self._X_test_risk.index)
        print(f"X_test: {type(self._X_test)} {self._X_test.shape}") # (92, 20)
        print(f"self._X_test_risk.index[{i}]: {self._X_test_risk.index[i]}")
        print(f"self._Y_test_risk.index[{i}]: {self._Y_test_risk.index[i]}")
        print(f"_X_test_risk:\n{self._X_test_risk.head()}")
        #self._shap = shap.TreeExplainer(self._rf)
        # https://github.com/shap/shap/issues/4224
        X = self._X_test.loc[[self._X_test_risk.index[i]], :] # Need to maintain the pandas DataFrame dtype
        X = pd.concat([X], keys=['#samples']) # Add axis-0 as samples

        Y = self._Y_test.loc[self._Y_test_risk.index[i]]
        Y = Y[numpy.newaxis, ...] # Add axis-0 as samples
        print(f"X: {type(X)}, {X.shape}\n{X}") # (20,)
        print(f"Y: {type(Y)}, {Y.shape}\n{Y}") # 1 scalar value
        print(f"index: {self._X_test_risk.index[i]}") # index: 414, X: (20,), Y: ()

        prediction = self._rf.predict_proba(self._X_test) # This needs to be done before adding the "risk" column below
        print(f"prediction: {prediction.shape}")
        # https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
        # https://github.com/shap/shap/issues/4225
        # https://github.com/shap/shap/pull/4254
        # X: A matrix of samples (# samples x # features) on which to explain the model’s output.
        shap_values = self._shap.shap_values(X)
        shap_value = shap_values[:,:,1]
        print(f"shap_values: {shap_values.shape}, {shap_values}") # shap_values: (1, 20, 2)
        print(f"shap_value: {shap_value.shape}, {shap_value}") # shap_value: (1, 20)
        print(f"expected_value: {self._shap.expected_value.shape}, {self._shap.expected_value}") # expected_value: (2,)
        #assert shap_values[0, :, 0].sum() + self._shap.expected_value[0] == prediction[0, 0], f"{shap_values[0, :, 0].sum()} + {self._shap.expected_value[0]} = {shap_values[0, :, 0].sum() + self._shap.expected_value[0]} != {prediction[0, 0]}"
        #assert shap_values[0, :, 1].sum() + self._shap.expected_value[1] == prediction[0, 1], f"{shap_values[0, :, 1].sum()} + {self._shap.expected_value[1]} = {shap_values[0, :, 1].sum() + self._shap.expected_value[1]} != {prediction[0, 1]}"
        shap.force_plot(self._shap.expected_value[1], shap_value, feature_names=self._X_test.columns, matplotlib=True, figsize=(20, 10))
        shap_values = self._shap.shap_values(self._X_test)[:,:,1] # shap_values: (92, 20)
        #assert shap_values[0, :, 0].sum() + self._shap.expected_value[0] == prediction[0, 0], f"{shap_values[0, :, 0].sum()} + {self._shap.expected_value[0]} = {shap_values[0, :, 0].sum() + self._shap.expected_value[0]} != {prediction[0, 0]}"
        #assert shap_values[0, :, 1].sum() + self._shap.expected_value[1] == prediction[0, 1], f"{shap_values[0, :, 1].sum()} + {self._shap.expected_value[1]} = {shap_values[0, :, 1].sum() + self._shap.expected_value[1]} != {prediction[0, 1]}"
        print(f"Summary Plot shap_values: {shap_values.shape}, {shap_values}") # (92, 20)
        shap.summary_plot(shap_values, self._X_test)
        if dependencies is not None and len(dependencies) > 0:
            for dep in dependencies:
                print(f"dependence_plot: {dep[0]} {dep[1]}")
                shap.dependence_plot(dep[0], shap_values, self._X_test, interaction_index=dep[1])