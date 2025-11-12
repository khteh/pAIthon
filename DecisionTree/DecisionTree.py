import pickle, itertools
from abc import ABC
from tqdm import tqdm

class DecisionTree(ABC):
    _model_path:str = None
    _X_train = None
    _X_val = None
    _X_test = None
    _X_test_risk = None
    _Y_train = None
    _Y_val = None
    _Y_test = None
    _hyperparams:dict = None
    _fixed_hyperparams:dict = None
    _best_hyperparams = None

    def __init__(self, path:str, hyperparams:dict, fixed_hyperparams:dict):
        self._fixed_hyperparams = fixed_hyperparams
        self._hyperparams = hyperparams
        self._model_path = path

    def _random_forest_grid_search(self, model):
        print(f"\n=== {self._random_forest_grid_search.__name__} ===")
        # Define ranges for the chosen random forest hyperparameters 
        best_model, self._best_hyperparams = self._holdout_grid_search(model)
               
        # add fixed hyperparamters to best combination of variable hyperparameters
        self._best_hyperparams.update(self._fixed_hyperparams)
        if self._model_path:
            with open(self._model_path, 'wb') as f:
                pickle.dump(self._rf, f)
            print(f"Model saved to {self._model_path}.")
        return best_model

    def _holdout_grid_search(self, rf):
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
        lists = self._hyperparams.values()
        
        # get all param combinations
        param_combinations = list(itertools.product(*lists))
        total_param_combinations = len(param_combinations)

        # iterate through param combinations
        for i, params in enumerate(tqdm(param_combinations), 1):
            # fill param dict with params
            param_dict = {}
            for param_index, param_name in enumerate(self._hyperparams):
                param_dict[param_name] = params[param_index]
                
            # create estimator with specified params
            estimator = rf(**param_dict, **self._fixed_hyperparams)

            # fit estimator
            estimator.fit(self._X_train, self._Y_train)
            
            # get predictions on validation set
            preds = estimator.predict_proba(self._X_val if self._X_val is not None and len(self._X_val) > 0 else self._X_test)
            #print(f"prediction: shape: {preds.shape} {preds}") # Matches with self._rf.classes_: [probablity of NOT dying within 10 years, probablity of dying within 10 years]
            # compute cindex for predictions
            estimator_score = self._Evaluate(preds)

            print(f'[{i}/{total_param_combinations}] {param_dict}')
            print(f'Val score: {estimator_score}\n')

            # if new high score, update high score, best estimator
            # and best params 
            if estimator_score >= best_score:
                    best_score = estimator_score
                    best_estimator = estimator
                    best_hyperparams = param_dict

        # add fixed hyperparamters to best combination of variable hyperparameters
        best_hyperparams.update(self._fixed_hyperparams)
        return best_estimator, best_hyperparams
