from utils.all_utils import *
import argparse
from utils.plot import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from utils.model import *
import os


args = argparse.ArgumentParser()

args.add_argument("--config", "-c", help="Path to config file", default="config.yaml")
args.add_argument("--input_file", help="Path to input file", type=str, default="Clean_mordred_descriptors_with_binary_class.csv")
args.add_argument("--predication_input_file", help="Path to predication input file", type=str, default="Clean_mordred_descriptors_with_binary_class.csv")
parsed_args = args.parse_args()

def training(config_path):
    config = read_config(config_path)
    print(config)

    df = read_file(parsed_args.input_file)
    print(f"original dataset shape \n {df.shape}")

    df_p = read_file(parsed_args.predication_input_file)

    print(f"Value counts of classes: \n {df[config['params']['label_column']].value_counts()}")

    label_column = config['params']['label_column']
    unnecessary_columns = config['params']['unnecessary_columns']
    df_new = remove_unnecessary_columns(df, unnecessary_columns)
    print(f"Dataset shape after removing unnecessary_columns: {df_new.shape}")

    df_new = drop_missing_columns(df_new)
    print(f"Dataset shape after removing missing values: {df_new.shape}")
    

    df_new = drop_zero_columns(df_new)
    print(f"Dataset shape after removing zero columns: {df_new.shape}")

    # value_1 = config['params']['value_1']
    # value_2 = config['params']['value_2']


    # df_new = convert_to_binary(df_new, label_column, value_1, value_2)
    
    # df_new = remove_inbalanced_data(df_new, label_column, 1)
    # print(df_new[label_column].value_counts())
    
    x, y = split_data(df_new, label_column)
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
 
    x_p, y_p = split_data(df_p, label_column)

    # x = remove_correlated_columns(x, 0.9)
    # print(f"x shape after removing correlated columns {x.shape}")


    x = standard_scaling_data(x, x)

    x_p = standard_scaling_data(x, x_p)    

    x, y = oversample_data_minor_class(x, y)

    print(y[label_column].value_counts())

    test_size = config['params']['test_size']
    x_train, x_test, y_train, y_test = split_train_test_data(x,y, test_size)
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape} \n")

    model_list = [ExtraTreesClassifier(),AdaBoostClassifier(),LogisticRegression(),SVC(),GaussianNB(),RandomForestClassifier()]
    
    pred = all_classification_model(model_list, x_train, y_train, x_test, y_test, parsed_args.input_file.split("/")[-1].split(".")[-2], parsed_args.input_file.split("/")[-2], x, y)

    
if __name__ == "__main__":

    training(config_path=parsed_args.config)