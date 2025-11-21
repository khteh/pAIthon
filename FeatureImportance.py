import pandas as pd, pickle, numpy, matplotlib.pyplot as plt, shap
from utils.CIndex import CIndex

class FeatureImportance():
    _path:str = None
    _model_path:str = None
    _X_test = None
    _X_test_risky = None
    _Y_test = None
    _model = None
    def __init__(self, path, model_path:str):
        self._path = path
        self._model_path = model_path
        self._PrepareData()
        
    def ShowTestDataset(self):
        self._X_test_risky = self._X_test.copy(deep=True)
        self._X_test_risky.loc[:, 'risk'] = self._model.predict_proba(self._X_test)[:, 1] # Predicting our risk.
        self._X_test_risky = self._X_test_risky.sort_values(by='risk', ascending=False) # Sorting by risk value.
        print("Riskiest individual:")
        print(self._X_test_risky.head())
        importances = self.permutation_importance(self._X_test, self._y_test, cindex, num_samples=100)
        print("Importances:")
        print(importances)
        importances.T.plot.bar()
        plt.ylabel("Importance")
        plt.show()

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
        explainer = shap.TreeExplainer(self._model)
        i = 0 # Picking an individual
        shap_value = explainer.shap_values(self._X_test.loc[self._X_test_risky.index[i], :])[1]
        shap.force_plot(explainer.expected_value[1], shap_value, feature_names=self._X_test.columns, matplotlib=True)
        shap_values = shap.TreeExplainer(self._model).shap_values(self._X_test)[1]
        shap.summary_plot(shap_values, self._X_test)
        shap.dependence_plot('Age', shap_values, self._X_test, interaction_index = 'Sex')
        shap.dependence_plot('Poverty index', shap_values, self._X_test, interaction_index='Age')

    def _PrepareData(self):
        self._model = pickle.load(open(self._model_path, 'rb')) # Loading the model ModuleNotFoundError: No module named 'sklearn.ensemble.forest'
        test_df = pd.read_csv(self._path)
        test_df = test_df.drop(test_df.columns[0], axis=1)
        self._X_test = test_df.drop('y', axis=1)
        self._Y_test = test_df.loc[:, 'y']
        cindex_test = CIndex(self._Y_test, self._model.predict_proba(self._X_test)[:, 1])
        print("Model C-index on test: {}".format(cindex_test))

    def _permute_feature(self, df, feature):
        """
        Given dataset, returns version with the values of
        the given feature randomly permuted. 

        Args:
            df (dataframe): The dataset, shape (num subjects, num features)
            feature (string): Name of feature to permute
        Returns:
            permuted_df (dataframe): Exactly the same as df except the values
                                    of the given feature are randomly permuted.
        """
        permuted_df = df.copy(deep=True) # Make copy so we don't change original df

        # Permute the values of the column 'feature'
        permuted_features = numpy.random.permutation(df[feature])
        
        # Set the column 'feature' to its permuted values.
        permuted_df[feature] = permuted_features
        return permuted_df

    def permutation_importance(self, X, y, metric, num_samples = 100):
        """
        Compute permutation importance for each feature.

        Args:
            X (dataframe): Dataframe for test data, shape (num subject, num features)
            y (numpy.array): Labels for each row of X, shape (num subjects,)
            model (object): Model to compute importances for, guaranteed to have
                            a 'predict_proba' method to compute probabilistic 
                            predictions given input
            metric (function): Metric to be used for feature importance. Takes in ground
                            truth and predictions as the only two arguments
            num_samples (int): Number of samples to average over when computing change in
                            performance for each feature
        Returns:
            importances (dataframe): Dataframe containing feature importance for each
                                    column of df with shape (1, num_features)
        """
        importances = pd.DataFrame(index = ['importance'], columns = X.columns)
        
        # Get baseline performance (note, you'll use this metric function again later)
        baseline_performance = metric(y, self._model.predict_proba(X)[:, 1])

        print(f"model.classes: {self._model.classes_}")
        # Iterate over features (the columns in the importances dataframe)
        for feature in importances: # complete this line
            print(f"feature: {feature}")
            # Compute 'num_sample' performances by permutating that feature
            
            # You'll see how the model performs when the feature is permuted
            # You'll do this num_samples number of times, and save the performance each time
            # To store the feature performance,
            # create a numpy array of size num_samples, initialized to all zeros
            feature_performance_arr = numpy.zeros(num_samples)
            
            # Loop through each sample
            for i in range(num_samples): # complete this line
                
                # permute the column of dataframe X
                perm_X = self._permute_feature(X, feature)
                
                # calculate the performance with the permuted data
                # Use the same metric function that was used earlier
                feature_performance_arr[i] = metric(y, self._model.predict_proba(perm_X)[:,1])
        
        
            # Compute importance: absolute difference between 
            # the baseline performance and the average across the feature performance
            importances[feature]['importance'] = baseline_performance - numpy.mean(feature_performance_arr)
        return importances
    
if __name__ == "__main__":
    featureImportance = FeatureImportance('data/nhanest_test.csv', 'models/nhanes_rf.sav')
    featureImportance.ShowTestDataset()
    featureImportance.ExplainModelPrediction()