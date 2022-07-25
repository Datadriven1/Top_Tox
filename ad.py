from sklearn.model_selection import permutation_test_score, StratifiedKFold, GridSearchCV
from utils.all_utils import *
import argparse
from utils.plot import *
from utils.model import *
import os
from sklearn.ensemble import RandomForestClassifier


args = argparse.ArgumentParser()

args.add_argument("--config", "-c", help="Path to config file", default="config.yaml")
args.add_argument("--input_file", help="Path to input file", type=str, default="Data/DILI_Data/ECFP4_Fingerprints.xlsx")
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


    # x = standard_scaling_data(x, x)

    # x, y = oversample_data_minor_class(x, y)

    # print(y[label_column].value_counts())

    test_size = config['params']['test_size']
    X_train, x_test, y_train, y_test = split_train_test_data(x,y, test_size)
    print(f"x_train shape: {X_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape} \n")

    # Number of trees in random forest
    n_estimators = [750, 1000]
    max_features = ['auto']
    criterion = ['gini', 'entropy']

    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'criterion': criterion}

    # setup model building
    rf = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1, cv=5, verbose=1)
    rf.fit(X_train, y_train)
    print()
    print('Best params: %s' % rf.best_params_)
    print('Score: %.2f' % rf.best_score_)

    rf_best = RandomForestClassifier(**rf.best_params_)
    rf_best.fit(X_train, y_train)

    # Params
    pred = []
    ad = []
    index = []
    cross_val = StratifiedKFold(n_splits=4)

    # Do 5-fold loop
    for train_index, test_index in cross_val.split(X_train, y_train):
        print(train_index)
        print(test_index)
        print(len(train_index))
        print(len(test_index))
        print(train_index.shape)
        print(test_index.shape)
        print(type(train_index))
        print(type(test_index))
    #    fold_model = rf_best.fit(X_train.iloc[train_index], y_train[train_index])
        fold_pred = rf_best.predict(X_train.iloc[test_index])
        fold_ad = rf_best.predict_proba(X_train.iloc[test_index])
        pred.append(fold_pred)
        ad.append(fold_ad)
        index.append(test_index)

    threshold_ad = 0.6

    # Prepare results to export    
    fold_index = np.concatenate(index)    
    fold_pred = np.concatenate(pred)
    fold_ad = np.concatenate(ad)
    fold_ad = (np.amax(fold_ad, axis=1) >= threshold_ad).astype(str)
    five_fold = pd.DataFrame({'Prediction': fold_pred,'AD': fold_ad}, index=list(fold_index))
    five_fold.AD[five_fold.AD == 'False'] = np.nan
    five_fold.AD[five_fold.AD == 'True'] = five_fold.Prediction
    five_fold.sort_index(inplace=True)
    five_fold['y_train'] = pd.DataFrame(y_train)
    five_fold_ad = five_fold.dropna().astype(int)
    coverage_5f = len(five_fold_ad) / len(five_fold)

    # Print stats
    cross_val_stats = pd.DataFrame(stats(five_fold.y_train, five_fold.Prediction))
    cross_val_stats['Coverage'] = 1.0
    print('Five-fold external cross validation: \n', cross_val_stats.to_string(index=False), '\n')
    cross_val_stats_ad = pd.DataFrame(stats(five_fold_ad.y_train, five_fold_ad.AD))
    cross_val_stats_ad['Coverage'] = round(coverage_5f, 2)
    print('Five-fold external cross validation with 60% AD cutoff: \n', cross_val_stats_ad.to_string(index=False), '\n')


if __name__ == "__main__":

    training(config_path=parsed_args.config)


