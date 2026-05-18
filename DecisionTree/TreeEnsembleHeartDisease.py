import argparse, pandas as pd, matplotlib.pyplot as plt, shap, numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay, auc
from xgboost import XGBClassifier, DMatrix
from utils.DecisionTreeViz import PlotDecisionTree
from utils.Plots import plot_roc_curve, CorrelationMatrixHeatMap
from utils.ConfusionMatrix import ConfusionMatrix
from .DecisionTree import DecisionTree

# https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download
class HeartDisease(DecisionTree):
    _path:str = None
    _features = None
    _dt: DecisionTreeClassifier = None
    def __init__(self, path):
        self._path = path
        self._PrepareData()

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
        print(f"\n=== {self.BuildDecisionTreeModel.__name__} ===")
        min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
        max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
        accuracy_list_train = []
        accuracy_list_val = []
        for min_samples_split in min_samples_split_list:
            # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
            self._dt = DecisionTreeClassifier(min_samples_split = min_samples_split,
                                        random_state = 11).fit(self._X_train, self._Y_train) 
            predictions_train = self._dt.predict(self._X_train) ## The predicted values for the train dataset
            predictions_val = self._dt.predict(self._X_val) ## The predicted values for the test dataset
            accuracy_train = accuracy_score(predictions_train, self._Y_train)
            accuracy_val = accuracy_score(predictions_val, self._Y_val)
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)
        plt.figure(figsize=(10, 10), constrained_layout=True)
        plt.title('Train x Validation metrics (min_samples_split)', fontsize=22, fontweight="bold")
        plt.xlabel('min_samples_split')
        plt.ylabel('accuracy')
        plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
        plt.plot(accuracy_list_train)
        plt.plot(accuracy_list_val)
        plt.legend(['Train','Validation'], fontsize='x-large')
        plt.savefig(f"output/DecisionTree_samples_split.png")
        plt.clf()
        plt.close()

        accuracy_list_train = []
        accuracy_list_val = []
        for max_depth in max_depth_list:
            # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
            self._dt = DecisionTreeClassifier(max_depth = max_depth,
                                        random_state = 11).fit(self._X_train, self._Y_train) 
            predictions_train = self._dt.predict(self._X_train) ## The predicted values for the train dataset
            predictions_val = self._dt.predict(self._X_val) ## The predicted values for the test dataset
            accuracy_train = accuracy_score(predictions_train, self._Y_train)
            accuracy_val = accuracy_score(predictions_val, self._Y_val)
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)
        plt.figure(figsize=(10, 10), constrained_layout=True)
        plt.title('Train x Validation metrics (max_depth)', fontsize=22, fontweight="bold")
        plt.xlabel('max_depth')
        plt.ylabel('accuracy')
        plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
        plt.plot(accuracy_list_train)
        plt.plot(accuracy_list_val)
        plt.legend(['Train','Validation'], fontsize='x-large')
        plt.savefig(f"output/DecisionTree_max_depth.png")
        plt.clf()
        plt.close()
        # Choose min_samples_split and max_depth based on the plots above which yield highest accuracy and lowest variance / overfitting, i.e., training and test accuracy should be as close to one another as possible
        self._dt = DecisionTreeClassifier(min_samples_split = 50,
                                             max_depth = 3,
                                             random_state = 11).fit(self._X_train, self._Y_train)
        print(f"classes: {self._dt.classes_}") # classes: [False  True]
        print(f"Metrics train:\n\tAccuracy score: {accuracy_score(self._dt.predict(self._X_train), self._Y_train):.4f}")
        print(f"Metrics validation:\n\tAccuracy score: {accuracy_score(self._dt.predict(self._X_val), self._Y_val):.4f}")
        PlotDecisionTree(self._dt, self._features, ['neg', 'pos'], "HeartDiseasePredictionDecisionTree") # Matches with self._dt.classes_
        plt.clf()
        plt.close()
        self._EvaluateTest(self._dt)

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
                'min_samples_leaf': [3,5,7,9],
            }
            fixed_hyperparams = {
                'random_state': 11,
            }
            self._rf = self._random_forest_grid_search(RandomForestClassifier, self._X_train, self._Y_train, self._X_val, self._Y_val, hyperparams, fixed_hyperparams)
            print(f"Best hyperparameters:\n{self._best_hyperparams}")
        self._shap = shap.TreeExplainer(self._rf) # self._shap.expected_value is calculated during shap.TreeExplainer(model) initialization from the model's internal structure
        print(f"expected_value: {self._shap.expected_value.shape}, {self._shap.expected_value}")
        print(f"classes: {self._rf.classes_}") # classes: [False  True]
        print(f"Metrics train:\n\tAccuracy score: {accuracy_score(self._rf.predict(self._X_train), self._Y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(self._rf.predict(self._X_val), self._Y_val):.4f}")
        #PlotDecisionTree(self._rf, self._features, ['neg', 'pos'], "RandomForestHeartDiseasePrediction") AttributeError: 'RandomForestClassifier' object has no attribute 'tree_'
        self._ExplainRandomForestPrediction([("Age", "Sex_F"), ("Age", "Sex_M")])
        self._EvaluateTest(self._rf)

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
            self._xgb = self._random_forest_grid_search(XGBClassifier, self._X_train, self._Y_train, self._X_val, self._Y_val, hyperparams, fixed_hyperparams)
            print(f"Best hyperparameters:\n{self._best_hyperparams}")
        self._shap = shap.TreeExplainer(self._xgb)
        print(f"expected_value: {self._shap.expected_value.shape}, {self._shap.expected_value}")
        print(f"Best iteration with lowest evaluation metric: {self._xgb.best_iteration}")
        print(f"Metrics train:\n\tAccuracy score: {accuracy_score(self._xgb.predict(self._X_train), self._Y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(self._xgb.predict(self._X_val), self._Y_val):.4f}")
        #PlotDecisionTree(self._xgb, self._features, ['neg', 'pos'], "XGBoostHeartDiseasePrediction") AttributeError: 'XGBClassifier' object has no attribute 'tree_'
        self._ExplainXGBoostPrediction([("Age", "Sex_F"), ("Age", "Sex_M")])
        self._EvaluateTest(self._xgb)

    def _PrepareData(self):
        """
        Remove the binary variables, because one-hot encoding them would do nothing to them. To achieve this we will just count how many different values there are in each categorical variable and consider only the variables with 3 or more values.
        one-hot encoding aims to transform a categorical variable with n outputs into n binary variables.

        Pandas has a built-in method to one-hot encode variables, it is the function pd.get_dummies. There are several arguments to this function, but here we will use only a few. They are:

        data: DataFrame to be used
        prefix: A list with prefixes, so we know which value we are dealing with
        columns: the list of columns that will be one-hot encoded. 'prefix' and 'columns' must have the same length.
        """
        print(f"\n=== {self._PrepareData.__name__} ===")
        # Load the dataset using pandas
        df = pd.read_csv(self._path)
        self._PlotData(df)
        CorrelationMatrixHeatMap(df.corr())
        cat_variables = ['Sex',
            'ChestPainType',
            'RestingECG',
            'ExerciseAngina',
            'ST_Slope'
        ]
        # This will replace the columns with the one-hot encoded ones and keep the columns outside 'columns' argument as it is.
        df = pd.get_dummies(data = df, prefix = cat_variables, columns = cat_variables)
        self._features = [x for x in df.columns if x not in 'HeartDisease'] ## Removing our target variable
        # features: ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N', 'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']
        print(f"features: {self._features}")
        # We will keep the shuffle = True since our dataset has not any time dependency.    
        #self._X_train, self._X_val, self._Y_train, self._Y_val = train_test_split(df[self._features], df['HeartDisease'], train_size = 0.8)

        # Get 80% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
        self._X_train, x_, self._Y_train, y_ = train_test_split(df[self._features], df['HeartDisease'], test_size=0.20, random_state=1)

        # Split the 40% subset above into two: one half for cross validation and the other for the test set
        self._X_val, self._X_test, self._Y_val, self._Y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

        # Delete temporary variables
        del x_, y_

        print(f"X_train: {self._X_train.shape}, Y_train: {self._Y_train.shape}, X_val: {self._X_val.shape}, Y_val: {self._Y_val.shape}, X_test: {self._X_test.shape}, Y_test: {self._Y_test.shape}")
        print(f'train samples: {len(self._X_train)}')
        print(f'validation samples: {len(self._X_val)} {type(self._X_val)}, Y_val: {self._Y_val.shape} {type(self._Y_val)}')
        print(f'target proportion: {sum(self._Y_train)/len(self._Y_train):.4f}')
        print(f"Y: shape: {self._Y_train.shape}, {self._Y_train[:10]}")
        print(f"X:")
        print(self._X_train.head())

    def _PlotData(self, data):
        print(f"\n=== {self._PlotData.__name__} ===")
        ages = numpy.sort(data.Age.unique())
        type(ages)
        males = []
        females = []
        for age in ages:
            males.append(len(data[(data.Age == age) & (data.Sex == 'M') & (data.HeartDisease == 1)]))
            females.append(len(data[(data.Age == age) & (data.Sex == 'F') & (data.HeartDisease == 1)]))

        #data.plot(kind='scatter', x='age', y='chol', c='target')
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,6), sharex=True)

        age_cholesterol = ax[0].scatter(x=data['Age'], y=data['Cholesterol'], c=data['HeartDisease'])
        ax[0].set(title="Heart Disease (Age and Cholesterol Levels)", ylabel="Cholesterol")
        ax[0].legend(*age_cholesterol.legend_elements(), title="Diagnosis") # legend_elements() will look for the c=data['target']
        ax[0].axhline(data.Cholesterol.mean(), linestyle="--")

        genders = {
            'Male': males,
            'Female': females
        }

        x = ages #np.arange(len(species))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        for gender, count in genders.items():
            offset = width * multiplier
            rects = ax[1].bar(x + offset, count, width, label=gender)
            ax[1].bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[1].set(title='Heart Disease (Age and Gender)', xlabel="Age", ylabel="Count")
        ax[1].set_xticks(x + width, ages)
        ax[1].legend(loc='upper left', ncols=3)
        ax[1].set_ylim(0, 10)
        fig.suptitle("Heart Disease Analysis", fontsize=16, fontweight="bold")
        plt.legend(fontsize='x-large')
        plt.show()

    def _EvaluateTest(self, model):
        """
        # https://scikit-learn.org/stable/modules/model_evaluation.html
        Estimator score method: Estimators have a score method providing a default evaluation criterion for the problem they are designed to solve. 
        Most commonly this is accuracy for classifiers and the coefficient of determination (R^2) for regressors. Details for each estimator can be found in its documentation.
        """
        predictions = model.predict(self._X_test)

        # 1. Accuracy
        model.score(self._X_train, self._Y_train)
        accuracy = model.score(self._X_test, self._Y_test)
        print(f"Accuracy: {accuracy}")

        # 2. ROC curve
        # Make predictions with probabilities
        y_probs = model.predict_proba(X_test) # predict the class probabilities. The returned estimates for all classes are ordered by the label of classes.
        # y_probs[0]: Probabilities of class-0 (false)
        # y_probs[1]: Probabilities of class-1 (true)
        y_probs_positives = y_probs[:, 1]
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        fpr, tpr, thresholds = roc_curve(self._Y_test, y_probs_positives)
        auc_score = roc_auc_score(self._Y_test, y_probs_positives)
        print(f"AUC Score: {auc_score}") # 
        plot_roc_curve(tpr, fpr)
        roc_auc = auc(fpr, tpr)
        rocCurveDisplay = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name="Heart Disease Classifier")
        rocCurveDisplay.plot()
        plt.show()

        # 3. Confusion Matrix
        confusion = ConfusionMatrix(self._Y_test, predictions)
        print("Confusion matrix:")
        print(confusion)
        ConfusionMatrixDisplay.from_predictions(self._Y_test.values.reshape([1, -1]), predictions.reshape([1,-1]), display_labels=["No Disease", "Disease"])
        plt.title("Heart Disease Confusion Matrix")
        plt.show()

        # 4. Classification Report
        report = classification_report(self._Y_test, predictions)
        print("Classification Report:")
        print(report)
        
    def _Evaluate(self, Y, probabilities):
        """
        Method used by the DecisionTree base class to score the model selection process.
        Evaluate the model's performance using the accuracy score
        Args:
            Y: Y_val
            probabilities: the predicted probabilities of the classes
        """
        predicted_labels = (probabilities[:, 1] > 0.5).astype(int)
        return accuracy_score(Y, predicted_labels)

if __name__ == "__main__":
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Heart Disease Prediction Tree Ensemble')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    parser.add_argument('-g', '--grayscale', action='store_true', help='Use grayscale model')
    args = parser.parse_args()

    heart = HeartDisease("data/heart.csv") # https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download
    heart.BuildDecisionTreeModel()
    heart.BuildRandomForestModel("models/RandomForestHeartDisease.pkl", args.retrain)
    heart.BuildXGBoost("models/XGBoostHeartDisease.pkl", args.retrain)