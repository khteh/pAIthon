import argparse, shap, sklearn, itertools, lifelines, pydotplus, numpy, pandas as pd, seaborn as sb, matplotlib.pyplot as plt, pickle, random
from pathlib import Path
from pandas import DataFrame
from sklearn.tree import export_graphviz
from six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from xgboost import XGBClassifier
from utils.DecisionTreeViz import PlotDecisionTree
from .DecisionTree import DecisionTree
from utils.CIndex import CIndex
class MedicalTreatmentTLearner(DecisionTree):
    """
    Examine data from an RCT (Randomized Control Trial), measuring the effect of a particular drug combination on colon cancer. 
    Specifically, we'll be looking the effect of [Levamisole](https://en.wikipedia.org/wiki/Levamisole) and [Fluorouracil](https://en.wikipedia.org/wiki/Fluorouracil) on patients who have had surgery to remove their colon cancer. 
    After surgery, the curability of the patient depends on the remaining residual cancer. In this study, it was found that this particular drug combination had a clear beneficial effect, when compared with [Chemotherapy](https://en.wikipedia.org/wiki/Chemotherapy). 
    """
    _treatment_path: str = None
    _control_path:str = None
    _data: DataFrame = None
    _X_treat_train = None
    _Y_treat_train = None
    _X_treat_val = None
    _Y_treat_val = None
    _X_control_train = None
    _Y_control_train = None
    _X_control_val = None
    _Y_control_val = None
    _threshold:float = None
    _imputer_iterations: int = None
    _imputer: IterativeImputer = None
    _treatment_model: RandomForestClassifier = None
    _control_model: RandomForestClassifier = None
    _best_hyperparam_treat = None
    _best_hyperparam_ctrl = None
    _shap = None
    def __init__(self, treatment_path:str, control_path:str, threshold:float, imputer_iterations:int):
        super().__init__(None)
        self._treatment_path = treatment_path
        self._control_path = control_path
        self._imputer_iterations = imputer_iterations
        self._threshold = threshold
        self._PrepareData()
        self._treatment_model = self._LoadModel(self._treatment_path)
        self._control_model = self._LoadModel(self._control_path)

    def BuildRandomForestModel(self, retrain:bool = False):
        """
        Two-Tree method (T-learner)
        - miu1(x): Prognostic model for Yi(1)
        - miu0(x): Prognostic model for Yi(0)
        - miu1 - miu0

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
        if not self._treatment_model or not self._control_model or retrain:
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
                'random_state': 12,
            }
            print(f"X_treat_train: {self._X_treat_train.shape}, Y_treat_train: {self._Y_treat_train.shape}")
            # perform grid search with the treatment data to find the best model 
            self._treatment_model, self._best_hyperparam_treat  = self._random_forest_grid_search(RandomForestClassifier,
                                                self._X_treat_train, self._Y_treat_train,
                                                self._X_treat_val, self._Y_treat_val, hyperparams, fixed_hyperparams, self._treatment_path)
            # perform grid search with the control data to find the best model 
            self._control_model, self._best_hyperparam_ctrl = self._random_forest_grid_search(RandomForestClassifier,
                                                self._X_control_train, self._Y_control_train,
                                                self._X_control_val, self._Y_control_val, hyperparams, fixed_hyperparams, self._control_path)
            #self._shap = shap.TreeExplainer(self._rf)
        # Best treatment hyperparameters:
        # {'n_estimators': 789, 'max_depth': None, 'min_samples_leaf': 3, 'random_state': 12}
        # Best control hyperparameters:
        # {'n_estimators': 789, 'max_depth': 13, 'min_samples_leaf': 3, 'random_state': 12}
        # X_val num of patients 61
        print(f"Best treatment hyperparameters:\n{self._best_hyperparam_treat}")
        print(f"Best control hyperparameters:\n{self._best_hyperparam_ctrl}")

    def EvaluateValidationDataset(self):
        # Use the t-learner to predict the risk reduction for patients in the validation set
        print(f"\n=== {self.EvaluateValidationDataset.__name__} ===")
        rr_t_val = self.Predict(self._X_val.drop(['TRTMT'], axis=1))
        print(f"X_val num of patients {self._X_val.shape[0]}")
        print(f"rr_t_val num of patient predictions {rr_t_val.shape[0]}")
        print("rrt_val:")
        print(rr_t_val[:10])
        plt.hist(rr_t_val, bins='auto')
        plt.title("Histogram of Predicted ARR, T-Learner, validation set", fontsize=22, fontweight="bold")
        plt.xlabel('predicted risk reduction', fontsize="xx-large")
        plt.ylabel('count of patients', fontsize="xx-large")
        plt.legend(loc='upper right')
        plt.show()
        print(f"X_val: {self._X_val.shape}, Y_val: {self._Y_val.shape}, rr_t_val: {rr_t_val.shape}")
        empirical_benefit, avg_benefit = self._quantile_benefit(self._X_val, self._Y_val, rr_t_val)
        self._plot_empirical_risk_reduction(empirical_benefit, avg_benefit, 'T Learner [val set]')
        c_for_benefit_tlearner_val_set = self._c_statistic(rr_t_val, self._Y_val, self._X_val.TRTMT)
        print(f"C-for-benefit statistic of T-learner on val set: {c_for_benefit_tlearner_val_set:.4f}")
    
    def EvaluateTestDataset(self):
        # Use the t-learner to predict the risk reduction for patients in the test set
        print(f"\n=== {self.EvaluateTestDataset.__name__} ===")
        rr_t_test = self.Predict(self._X_test.drop(['TRTMT'], axis=1))
        print(f"X_test num of patients {self._X_test.shape[0]}")
        print(f"rr_t_test num of patient predictions {rr_t_test.shape[0]}")
        plt.hist(rr_t_test, bins='auto')
        plt.title("Histogram of Predicted ARR, T-Learner, test set", fontsize=22, fontweight="bold")
        plt.xlabel('predicted risk reduction', fontsize="xx-large")
        plt.ylabel('count of patients', fontsize="xx-large")
        plt.legend(loc='upper right')
        plt.show()
        print(f"_X_test: {self._X_test.shape}, _Y_test: {self._Y_test.shape}")
        empirical_benefit, avg_benefit = self._quantile_benefit(self._X_test, self._Y_test, rr_t_test)
        self._plot_empirical_risk_reduction(empirical_benefit, avg_benefit, 'T Learner [test set]')
        c_for_benefit_tlearner_test_set = self._c_statistic(rr_t_test, self._Y_test, self._X_test.TRTMT)
        print(f"C-for-benefit statistic of T-learner on test set: {c_for_benefit_tlearner_test_set:.4f}")

    def Predict(self, data):
        """
        Return predicted risk reduction for treatment for given data matrix.
        Notice when viewing the histogram that predicted risk reduction can be negative.
        - This means that for some patients, the T-learner predicts that treatment will actually increase their risk (negative risk reduction). 
        - The T-learner is more flexible compared to the logistic regression model, which only predicts non-negative risk reduction for all patients (view the earlier histogram of the 'predicted ARR' histogram for the logistic regression model, and you'll see that the possible values are all non-negative).
        Args:
          X (dataframe): dataframe containing features for each subject
    
        Returns:
          preds (np.array): predicted risk reduction for each row of X
        """
        print(f"\n=== {self.Predict.__name__} ===")
        print(f"data: {data.shape}")
        # predict the risk of death using the control estimator
        risk_control = self._control_model.predict_proba(data)[:, 1]
        print(f"risk_control: {risk_control.shape}")
        # predict the risk of death using the treatment estimator
        risk_treatment = self._treatment_model.predict_proba(data)[:, 1]
        print(f"risk_treatment: {risk_treatment.shape}")
        # the predicted risk reduction is control risk minus the treatment risk
        pred_risk_reduction =  risk_control - risk_treatment
        return pred_risk_reduction
            
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
        Below is a description of all the fields (one-hot means a different field for each level):
        - `sex (binary): 1 if Male, 0 otherwise`
        - `age (int): age of patient at start of the study`
        - `obstruct (binary): obstruction of colon by tumor`
        - `perfor (binary): perforation of colon`
        - `adhere (binary): adherence to nearby organs`
        - `nodes (int): number of lymphnodes with detectable cancer`
        - `node4 (binary): more than 4 positive lymph nodes`
        - `outcome (binary): 1 if died within 5 years`
        - `TRTMT (binary): treated with levamisole + fluoroucil`
        - `differ (one-hot): differentiation of tumor`
        - `extent (one-hot): extent of local spread`
        In particular pay attention to the `TRTMT` and `outcome` columns. Our primary endpoint for our analysis will be the 5-year survival rate, which is captured in the `outcome` variable.
        """
        print(f"\n=== {self._PrepareData.__name__} ===")
        self._data = pd.read_csv("data/levamisole_data.csv", index_col=0)
        print(f"Data Dimensions: {self._data.shape}")
        treated_prob, control_prob = self._event_rate()
        print(f"Death rate for treated patients: {treated_prob:.4f} ~ {int(treated_prob*100)}%")
        print(f"Death rate for untreated patients: {control_prob:.4f} ~ {int(control_prob*100)}%")

        #data = data.dropna(axis=0)
        y = self._data.outcome
        # notice we are dropping a column here. Now our total columns will be 1 less than before
        X = self._data.drop('outcome', axis=1) 
        self._X_train, x_, self._Y_train, y_ = train_test_split(X, y, test_size=0.20, random_state=1)

        # Split the 40% subset above into two: one half for cross validation and the other for the test set
        self._X_val, self._X_test, self._Y_val, self._Y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

        # Delete temporary variables
        del x_, y_

        print(f"X_train: {self._X_train.shape}, Y_train: {self._Y_train.shape}, X_val: {self._X_val.shape}, Y_val: {self._Y_val.shape}, X_test: {self._X_test.shape}, Y_test: {self._Y_test.shape}")
        print(f"X columns: {X.columns}")
        print("X:")
        print(X.head())
        print("Y:")
        print(y.head())

        # Impute using regression on other covariates
        self._imputer = IterativeImputer(random_state=0, sample_posterior=False, max_iter=self._imputer_iterations, min_value=0)
        self._imputer.fit(self._X_train)
        self._X_train = pd.DataFrame(self._imputer.transform(self._X_train), columns=self._X_train.columns, index=self._X_train.index)
        self._X_val = pd.DataFrame(self._imputer.transform(self._X_val), columns=self._X_val.columns, index=self._X_val.index)
        self._X_test = pd.DataFrame(self._imputer.transform(self._X_test), columns=self._X_test.columns, index=self._X_test.index)
        self._treatment_dataset_split()

    def _treatment_dataset_split(self):
        """
        Separate treated and control individuals in training
        and testing sets. Remember that returned
        datasets should NOT contain the 'TRTMT' column!

        Args:
            X_train (dataframe): dataframe for subject in training set
            self._Y_train (np.array): outcomes for each individual in X_train
            X_val (dataframe): dataframe for subjects in validation set
            self._Y_val (np.array): outcomes for each individual in X_val
        
        Returns:
            X_treat_train (df): training set for treated subjects
            y_treat_train (np.array): labels for X_treat_train
            X_treat_val (df): validation set for treated subjects
            y_treat_val (np.array): labels for X_treat_val
            X_control_train (df): training set for control subjects
            y_control_train (np.array): labels for X_control_train
            X_control_val (np.array): validation set for control subjects
            y_control_val (np.array): labels for X_control_val
        """
        # From the training set, get features of patients who received treatment
        self._Y_train = pd.DataFrame(self._Y_train, index = self._X_train.index)
        self._Y_val = pd.DataFrame(self._Y_val, index = self._X_val.index)
        print(f"X_train: {self._X_train.shape}, Y_train: {self._Y_train.shape}")
        print(self._X_train.head())
        print(self._Y_train[:10])
        #print(f"columns: {X_train.columns}")
        self._X_treat_train = self._X_train[self._X_train.TRTMT == True]
        
        # drop the 'TRTMT' column
        self._X_treat_train.drop("TRTMT", axis=1, inplace=True)
        
        # From the training set, get the labels of patients who received treatment
        self._Y_treat_train = self._Y_train[self._X_train.TRTMT == True].to_numpy().ravel()

        # From the validation set, get the features of patients who received treatment
        self._X_treat_val = self._X_val[self._X_val.TRTMT == True]
                            
        # Drop the 'TRTMT' column
        self._X_treat_val.drop("TRTMT", axis=1, inplace=True)
                            
        # From the validation set, get the labels of patients who received treatment
        self._Y_treat_val = self._Y_val[self._X_val.TRTMT == True].to_numpy().ravel()
                            
        # --------------------------------------------------------------------------------------------
                            
        # From the training set, get the features of patients who did not received treatment
        self._X_control_train = self._X_train[self._X_train.TRTMT == False]
                            
        # Drop the TRTMT column
        self._X_control_train.drop("TRTMT", axis=1, inplace=True)
                            
        # From the training set, get the labels of patients who did not receive treatment
        self._Y_control_train = self._Y_train[self._X_train.TRTMT == False].to_numpy().ravel()
        
        # From the validation set, get the features of patients who did not receive treatment
        self._X_control_val = self._X_val[self._X_val.TRTMT == False]
        
        # drop the 'TRTMT' column
        self._X_control_val.drop("TRTMT", axis=1, inplace=True)

        # From the validation set, get teh labels of patients who did not receive treatment
        self._Y_control_val = self._Y_val[self._X_val.TRTMT == False].to_numpy().ravel()

    def _Evaluate(self, Y, predictions):
        # Evaluate the model's performance using the regular concordance index
        return CIndex(Y, predictions[:, 1])

    def _event_rate(self):
        '''
        Compute empirical rate of death within 5 years
        for treated and untreated groups.

        Args:
            df (dataframe): dataframe containing trial results. 
                            'TRTMT' column is 1 if patient was treated, 0 otherwise. 
                                'outcome' column is 1 if patient died within 5 years, 0 otherwise.
    
        Returns:
            treated_prob (float64): empirical probability of death given treatment
            untreated_prob (float64): empirical probability of death given control
        '''
        treated_prob = 0.0
        control_prob = 0.0
        
        treated_prob = numpy.float64(sum(self._data[self._data.TRTMT == 1].outcome) / sum(self._data.TRTMT))
        control_prob = numpy.float64(sum(self._data[self._data.TRTMT == 0].outcome) / sum(self._data.TRTMT == 0))
        return treated_prob, control_prob
    
    def _quantile_benefit(self, X, y, arr_hat):
        df = X.copy(deep=True)
        df.loc[:, 'y'] = y
        df.loc[:, 'benefit'] = arr_hat
        benefit_groups = pd.qcut(arr_hat, 10)
        df.loc[:, 'benefit_groups'] = benefit_groups
        empirical_benefit = df.loc[df.TRTMT == 0, :].groupby('benefit_groups').y.mean() - df.loc[df.TRTMT == 1].groupby('benefit_groups').y.mean()
        avg_benefit = df.loc[df.TRTMT == 0, :].y.mean() - df.loc[df.TRTMT==1, :].y.mean()
        return empirical_benefit, avg_benefit

    def _plot_empirical_risk_reduction(self, emp_benefit, av_benefit, model):
        """
        Recall that the predicted risk reduction is along the horizontal axis and the vertical axis is the empirical (actual risk reduction).
        A good model would predict a lower risk reduction for patients with actual lower risk reduction.  Similarly, a good model would predict a higher risk reduction for patients with actual higher risk reduction (imagine a diagonal line going from the bottom left to the top right of the plot).
        The T-learner seems to be doing a bit better (compared to the logistic regression model) at differentiating between the people who would benefit most treatment and the people who would benefit least from treatment.
        """
        plt.scatter(range(len(emp_benefit)), emp_benefit)
        plt.xticks(range(len(emp_benefit)), range(1, len(emp_benefit) + 1))
        plt.title("Empirical Risk Reduction vs. Predicted ({})".format(model), fontsize=22, fontweight="bold")
        plt.ylabel("Empirical Risk Reduction", fontsize="xx-large")
        plt.xlabel("Predicted Risk Reduction Quantile", fontsize="xx-large")
        plt.plot(range(10), [av_benefit]*10, linestyle='--', label='average RR')
        plt.legend(loc='lower right', fontsize="xx-large")
        plt.show()

    def _c_statistic(self, pred_rr, y, w):
        """
        Return concordance-for-benefit, the proportion of all matched pairs with
        unequal observed benefit, in which the patient pair receiving greater
        treatment benefit was predicted to do so.

        Args: 
            pred_rr (array): array of predicted risk reductions
            y (array): array of true outcomes
            w (array): array of true treatments 
        
        Returns: 
            cstat (float): calculated c-stat-for-benefit
        """
        assert len(pred_rr) == len(w) == len(y)
        
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        # Collect pred_rr, y, and w into tuples for each patient
        tuples = list(zip(pred_rr, y, w))
        
        # Collect untreated patient tuples, stored as a list
        untreated = [t for t in tuples if t[2] == 0]
        
        # Collect treated patient tuples, stored as a list
        treated = [t for t in tuples if t[2] == 1]

        # randomly subsample to ensure every person is matched
        
        # if there are more untreated than treated patients,
        # randomly choose a subset of untreated patients, one for each treated patient (length of treated patients).

        if len(untreated) > len(treated):
            untreated = random.sample(untreated, len(treated))
            
        # if there are more treated than untreated patients,
        # randomly choose a subset of treated patients, one for each untreated patient (length of untreated patients).
        if len(treated) > len(untreated):
            treated = random.sample(treated, len(untreated))
            
        assert len(untreated) == len(treated)

        # Sort the untreated patients by their predicted risk reduction
        untreated = sorted(untreated, key=lambda x: x[0])
        
        # Sort the treated patients by their predicted risk reduction
        treated = sorted(treated, key=lambda x: x[0])
        
        # match untreated and treated patients to create pairs together
        pairs = [(untreated[i], treated[i]) for i in range(len(treated))]

        # calculate the c-for-benefit using these pairs (use the function that you implemented earlier)
        cstat = self._c_for_benefit_score(pairs)
        return cstat
    
    def _c_for_benefit_score(self, pairs):
        """
        Compute c-statistic-for-benefit given list of
        individuals matched across treatment and control arms. 

        Args:
            pairs (list of tuples): each element of the list is a tuple of individuals,
                                    the first from the control arm and the second from
                                    the treatment arm. Each individual 
                                    p = (pred_outcome, actual_outcome) is a tuple of
                                    their predicted outcome and actual outcome.
        Result:
            cstat (float): c-statistic-for-benefit computed from pairs.
        """
        # mapping pair outcomes to benefit
        obs_benefit_dict = {
            (0, 0): 0,
            (0, 1): -1,
            (1, 0): 1,
            (1, 1): 0,
        }

        # compute observed benefit for each pair
        obs_benefit = [obs_benefit_dict[(i[1],j[1])] for (i,j) in pairs]

        # compute average predicted benefit for each pair
        pred_benefit = [(i[0]+j[0])/2 for (i,j) in pairs]

        concordant_count, permissible_count, risk_tie_count = 0, 0, 0

        # iterate over pairs of pairs
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                
                # if the observed benefit is different, increment permissible count
                if obs_benefit[i] != obs_benefit[j]:

                    # increment count of permissible pairs
                    permissible_count += 1
                    
                    # if concordant, increment count
                    # Higher ITE estimate has higher outcome.
                    if (pred_benefit[i] > pred_benefit[j] and obs_benefit[i] > obs_benefit[j]) or (pred_benefit[i] < pred_benefit[j] and obs_benefit[i] < obs_benefit[j]): # change to check for concordance
                        concordant_count += 1

                    # if risk tie, increment count
                    # Same ITE estiame with diff outcomes
                    if pred_benefit[i] == pred_benefit[j]: #change to check for risk ties
                        risk_tie_count += 1
        # compute c-statistic-for-benefit
        # The behaviour when permissible_count is zero is undefined or N/A.
        cstat = (concordant_count + 0.5 * risk_tie_count) / permissible_count
        return cstat    

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Random Forest Medical Treatment')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    parser.add_argument('-g', '--grayscale', action='store_true', help='Use grayscale model')
    args = parser.parse_args()

    treatment = MedicalTreatmentTLearner("models/MedicalTreatmentTLearner_treatment.pkl", "models/MedicalTreatmentTLearner_control.pkl", 10, 10)
    treatment.BuildRandomForestModel(args.retrain)
    treatment.EvaluateValidationDataset()
    treatment.EvaluateTestDataset()