from utils.all_utils import *
import argparse
from utils.plot import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from utils.model import *
import os
from imblearn.over_sampling import SMOTE


args = argparse.ArgumentParser()

args.add_argument("--config", "-c", help="Path to config file", default="config.yaml")
args.add_argument("--input_file", help="Path to input file", type=str, default="Clean_mordred_descriptors_with_binary_class.csv")
parsed_args = args.parse_args()

def training(config_path):
    config = read_config(config_path)
    print(config)

    df = read_file(parsed_args.input_file)
    print(f"original dataset shape \n {df.shape}")

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
 

    # x = remove_correlated_columns(x, 0.9)
    # print(f"x shape after removing correlated columns {x.shape}")


    x = standard_scaling_data(x, x)

    x, y = oversample_data(x, y)

    print(y[label_column].value_counts())

    test_size = config['params']['test_size']
    x_train, x_test, y_train, y_test = split_train_test_data(x,y, test_size)
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape} \n")

    model_list = [ExtraTreesClassifier(),AdaBoostClassifier(),LogisticRegression(),SVC(),GaussianNB(),KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier()]
    
    pred = all_classification_model(model_list, x_train, y_train, x_test, y_test, parsed_args.input_file.split("/")[-1].split(".")[-2], parsed_args.input_file.split("/")[-2])

    model_list_1 = []
    for i in model_list:
        model_list_1.append(str(type(i)).split(".")[-1].replace("'>",""))

    artifacts_dir = config["artifacts"]["artifacts_dir"]

    plots_dir = config["artifacts"][f"plots_dir"]
    plots_dir = (f"{plots_dir}/{parsed_args.input_file.split('/')[-2]}")
    plot_name = config["artifacts"]["plot"]
    plot_name = (f'{parsed_args.input_file.split("/")[-2]}_{parsed_args.input_file.split("/")[-1].split(".")[-2]}')
    plot_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plot_dir_path, exist_ok=True)

    save_plot(model_list_1 ,pred,plot_name, plot_dir_path, parsed_args.input_file.split("/")[-2], parsed_args.input_file.split("/")[-1].split(".")[-2])


if __name__ == "__main__":

    training(config_path=parsed_args.config)