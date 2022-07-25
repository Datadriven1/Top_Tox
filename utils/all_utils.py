import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
from sklearn.decomposition import PCA
import os
import pickle
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SMOTE
from sklearn import metrics

def read_config(config_path):
    with open(config_path) as config_file:
       content = yaml.safe_load(config_file)

    return content

def read_file(filename):
    """
    Loads the data from a csv or excel file.
    """
    if filename.endswith(".csv"):
        return pd.read_csv(filename)
    elif filename.endswith(".xlsx"):
        return pd.read_excel(filename)
    else:
        raise Exception("File format not supported.")

def remove_unnecessary_columns(df, unnecessary_columns):
    """
    Removes the unnecessary columns from the dataframe.
    """
    df = df.drop(unnecessary_columns , axis=1)
    return df

def drop_nan_rows(df):
    return df.dropna()

## create a function that handles inbalanced dataset by removing the 30% majority class
def remove_inbalanced_data(df, label_column, class_value):
    """
    Removes the inbalanced data from the dataframe.
    """
    df = df.drop(df[df[label_column] == class_value].sample(frac=0.4).index)
    return df

def drop_missing_values(df, threshold):
    missing_values = df.isnull().sum()
    missing_values_percentage = (missing_values / df.shape[0]) * 100
    missing_values_percentage = missing_values_percentage.sort_values(ascending=False)
    return df.drop(missing_values_percentage[missing_values_percentage > threshold].index, axis=1)

## creat a function that drop rows which have nan values
def drop_nan_rows(df):
    return df.dropna()


def drop_missing_columns(df):
    return df.drop(missing_columns(df), axis=1)

def all_zero(df):
    return df.columns[df.apply(lambda x: x == 0).all()].tolist()

def drop_zero_columns(df):
    return df.drop(all_zero(df), axis=1)

def missing_columns(df):
    return df.columns[df.isnull().any()].tolist()

def convert_to_binary(df, column, value_1, value_2):
    df[column] = df[column].map({value_1: 0, value_2: 1})
    return df

def remove_correlated_columns(df, threshold):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1)

## create a function that apply PCA transformation on x dataset
def pca_transformation(x, n_components):
    """
    Applies PCA transformation on the feature columns.
    """
    pca = PCA(n_components=n_components)
    x = pca.fit_transform(x)
    return x

def standard_scaling_data(x, y):
    scaler = StandardScaler()
    scaler.fit(x)
    y = scaler.transform(y)
    return y

def split_data(df, label_column):
    """
    Splits the data into train and test sets.
    """
    x = df.drop(columns=[label_column])
    y = pd.DataFrame(df[label_column])
    return x, y

def split_train_test_data(x, y, test_size):
    """
    Splits the data into train and test sets.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

def save_model(model, model_name, model_dir_path):
    """
    Saves the model into a pickle file.
    """
    model_path = os.path.join(model_dir_path, f"{model_name}.pkl")
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)



def oversample_data_minor_class(x, y):
        sm = SMOTE()
        x, y = sm.fit_resample(x, y)
        return x, y

## create a function that remove perticular value of data from dataframe
def remove_value_from_dataframe(df, column, value):
    df = df[df[column] != value]
    return df


## creat a function that remove null value from dataframe from specified column
def remove_null_value_from_dataframe(df, column):
    df = df[df[column].notnull()]
    return df

## creat a function that convert column data type from string to integer
def convert_column_data_type(df, column, data_type):
    df[column] = df[column].astype(data_type)
    return df

def stats(y_train, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_train, y_pred)
    accuracy = metrics.accuracy_score(y_train, y_pred)
#    roc_auc_score = metrics.roc_auc_score(y_train, y_pred)
    Kappa = metrics.cohen_kappa_score(y_train, y_pred, weights='linear')
    # True and false values
    TN, FP, FN, TP = confusion_matrix.ravel()  
    # Sensitivity, hit rate, recall, or true positive rate
    SE = TP/(TP+FN)
    # Specificity or true negative rate
    SP = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Correct classification rate
    CCR = (SE + SP)/2
    d = dict({'Accuracy': accuracy,
#         'AUC': roc_auc_score,
         'Kappa': Kappa,
         'CCR': CCR,
         'Sensitivity': SE,
         'PPV': PPV,
         'Specificity': SP,
         'NPV': NPV})
    return pd.DataFrame(d, columns=d.keys(), index=[0]).round(2)


## create a funcation that compare mean and standard deviation columns from dataframe and drop the row which have standard deviation heigher then mean and retain row if there is no value for standard deviation
def remove_outlier_from_dataframe(df, mean_column, std_column):
    df = df[(df[std_column] < df[mean_column])]
    return df

## create a funcation that replace nan value with zero
def replace_nan_with_zero(df, column):
    df[column] = df[column].fillna(0)
    return df