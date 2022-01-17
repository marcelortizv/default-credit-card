import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve
from sklearn.preprocessing import StandardScaler

# models

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
xgb.set_config(verbosity=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def preprocess(df: pd.DataFrame, target_string: str, scaler: bool):
    """
    Process the raw dataset into to sets, predictors and target
    :param
    df: raw dataframe
    target_string: name of target columns
    scaler: True or False if you want to apply Scaling
    :return:
    X: predictors / features
    y: outcome / target
    """
    if target_string in list(df.columns):
        df = clean_dataset(df)
        y = df[str(target_string)].values
        X = df.drop([str(target_string)], axis=1)

        if scaler:
            sc = StandardScaler()
            X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

            return X, y

        return X, y
    else:
        raise ValueError("Error, target_string do not belong to this dataset")


def report(y_true, y_pred):
    """
    Classification report based on scikit-learn
    :param y_true: ground truth
    :param y_pred: prediction
    :return: Classication Report
    """
    target_names = ['No Default', 'Default']  # you can change this passing it as parameter of the function ;)
    return print(classification_report(y_true, y_pred, target_names=target_names))


def print_ml_score(y_test, y_pred, clf):
    """
    Print report in console
    :param y_test: Testing outcome
    :param y_pred: Predictions
    :param clf: Classifier
    :return:
    """
    print('Classifier in testing: ', clf.__class__.__name__)
    report(y_test, y_pred)



def save_model(model, model_path, name, version):
    pickle.dump(model, open(f"{model_path}/{name}-{version}.pkl", 'wb'))
    print(f"Model {name}-{version}.pkl saved successfully")


def save_plot(plot, output_path, name, version):
    try:
        os.makedirs(f"{output_path}/plots_roc_curve")
    except FileExistsError:
        pass

    plt.savefig(f"{output_path}/plots_roc_curve/{name}-{version}.png")
    print(f"ROC Curve of model {name}-{version}.png saved successfully")


def train_model(classifier, param_grid, feature_vector_train, label_train, feature_vector_test, label_test, model_path, output_path, version):

    # name of model
    model_name = classifier.__class__.__name__
    grid = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='f1')
    grid_result = grid.fit(feature_vector_train, label_train)
    best_score = grid_result.best_score_
    best_params = grid_result.best_params_
    print(f"Training {model_name}-{version}:")
    print(f"This model uses {feature_vector_train.shape[1]} features")
    print(f"F1 score in training is: {best_score}")
    print(f"Best params in training are: {best_params}")

    # testing process
    model = grid_result.best_estimator_
    predictions = model.predict(feature_vector_test)
    # print classification report in console
    print_ml_score(label_test, predictions, classifier)
    plot_roc = plot_roc_curve(model, feature_vector_test, label_test)

    # save model in pickle format
    save_model(model, model_path, model_name, version)
    # save ROC Curve
    save_plot(plot_roc, output_path, model_name, version)
    print("---------------------------------------------------------\n\n")

    try:
        y_proba = model.predict_proba(feature_vector_test)[::,1]
        fpr, tpr, _ = roc_curve(label_test, y_proba)
        auc = roc_auc_score(label_test, y_proba)
        return fpr, tpr, auc, best_params, best_score

    except:
        fpr = 0
        tpr = 0
        auc = 0
        return fpr, tpr, auc, best_params, best_score


def get_columns_model(model_name, FEATURESDIR):
    features_file = glob.glob(f'{FEATURESDIR}/{model_name}*.txt')[0]
    with open(features_file) as f:
        lines = f.readlines()

    columns = []
    for line in lines:
        columns.append(line.strip())

    return columns


def select_k_variables(classifier, features, target, n_columns, min_cols):

    try:
        kbest = SelectKBest(f_classif)
        pipe = Pipeline([
            ('kbest', kbest),
            ('clf', classifier)
        ])

        # grid search to select the best k! It consider total columns of dataframe until the min_value of columns required
        grid_search = GridSearchCV(pipe, {'kbest__k': list(reversed(range(n_columns + 1)))[:min_cols - 3]})
        grid_search.fit(features, target)
        k_selected = grid_search.best_params_['kbest__k']

        fs = SelectKBest(score_func=f_classif, k=k_selected)

        # features selected
        _ = fs.fit_transform(features, target)
        features_names = list(fs.get_feature_names_out())
        # building dataframe to save artifacts of models
        scores = pd.DataFrame(fs.scores_)
        column_names = pd.DataFrame(features.columns)
        # concat two dataframes for better visualization
        features_scores = pd.concat([column_names, scores], axis=1)
        features_scores.columns = ['features_selected', 'score']  # naming the dataframe columns

        # keep only the best k features for model
        features_scores = features_scores.sort_values('score', ascending=False).head(k_selected).reset_index(drop=True)

        # artifact name
        artifact_name = f"{classifier.__class__.__name__}_{round(grid_search.best_score_, 2)}"

        print("----------------------------------------------------------------")
        print(f"Model {classifier.__class__.__name__} works with {k_selected} features")

        return features_names, features_scores, artifact_name

    except :
        print(f"Model {classifier.__class__.__name__} was omitted.")


def save_features_name(features_list, models_path, artifact_name):
    textfile = open(f"{models_path}/features_by_models/{artifact_name}.txt", "w")
    for element in features_list:
        textfile.write(element + "\n")
    textfile.close()

    return print(f"{artifact_name} saved successfully!")
