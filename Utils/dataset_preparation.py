import pandas as pd
import os
import numpy as np
from pyspark.sql import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def model_input_preparation(
    df: pd.DataFrame,
    categorical_feature_list: list,
    numerical_feature_list: list,
    feature_one_hot_list: list,
    normal_feature_list: list,
    one_hot: bool,
    scaler: bool,
    test_size: float,
    seed: int,
    selected_feature:bool,
    IS_SMOTE:bool
):
    if one_hot:
        encoder = OneHotEncoder(sparse_output=False, drop=None)
        encoded = encoder.fit_transform(df[categorical_feature_list])
        df_encoded = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_feature_list),
            index=df.index
        )
        df_final = pd.concat([df.drop(categorical_feature_list, axis=1), df_encoded], axis=1)
        if selected_feature:
            df_final = df_final[feature_one_hot_list + ["label"]]
            numerical_feature_list = list(set(numerical_feature_list).intersection(feature_one_hot_list))
    else:
        df_final = df.copy()
        if selected_feature:
            df_final = df_final[normal_feature_list + ["label"]]
            numerical_feature_list = list(set(numerical_feature_list).intersection(normal_feature_list))

    df_final["label"] = df_final["label"].map({"Yes": 1, "No": 0})

    X = df_final.drop(columns=['label'])
    y = df_final["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    if IS_SMOTE and one_hot:
        smote=SMOTE() 
        X_train,y_train=smote.fit_resample(X_train,y_train)
    elif IS_SMOTE and not one_hot:
        return "SMOTE need all numerical data, please use one hot with SMOTE"

    if scaler:
        std_scaler = StandardScaler()
        X_train_num = std_scaler.fit_transform(X_train[numerical_feature_list])
        X_test_num = std_scaler.transform(X_test[numerical_feature_list])

        X_train.loc[:, numerical_feature_list] = X_train_num
        X_test.loc[:, numerical_feature_list] = X_test_num

    return X_train, y_train, X_test, y_test
