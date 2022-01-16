import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# models

from sklearn.svm import SVC
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


def preprocess(df: pd.DataFrame, target_string) -> (np.ndarray, np.ndarray):
    """
    Process the raw dataset into to sets, predictors and target
    :param df: raw dataframe
    :return:
    X: predictors / features
    y: outcome / target
    """
    if target_string not in list(df.columns):
        raise ValueError("Error, target_string do not belong to this dataset")
    else:
        df = clean_dataset(df)
        y = df[str(target_string)].values
        X = df.drop([str(target_string)], axis=1)

        return X, y


def report(y_true, y_pred):
    """
    Classification report based on scikit-learn
    :param y_true: ground truth
    :param y_pred: prediction
    :return: Classication Report
    """
    target_names = ['No Fraud', 'Fraud']  # you can change this passing it as parameter of the function ;)
    return print(classification_report(y_true, y_pred, target_names=target_names))


def print_ml_score(y_test, y_pred, clf):
    print('Classifier: ', clf.__class__.__name__)
    report(y_test, y_pred)
    print("---------------------------------------------------------")


def train_model(classifier, feature_vector_train, label_train, feature_vector_test, label_test):

    resample = BorderlineSMOTE(random_state=593)
    pipeline = Pipeline([('SMOTE', resample), (str(classifier.__class__.__name__), classifier)])

    pipeline.fit(feature_vector_train, label_train)
    predictions = pipeline.predict(feature_vector_test)
    return print_ml_score(label_test, predictions, classifier)


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

        print(f"Model {classifier.__class__.__name__} works with {k_selected} features")
        print("---------------------------------------")
        return features_names, features_scores, artifact_name

    except :
        print(f"Model {classifier.__class__.__name__} was omitted.")


def save_features_name(features_list, models_path, artifact_name):
    textfile = open(f"{models_path}/features_by_models/{artifact_name}.txt", "w")
    for element in features_list:
        textfile.write(element + "\n")
    textfile.close()

    return print(f"{artifact_name} saved successfully!")
