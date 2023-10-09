from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
#from sqlalchemy import distinct


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2.Encode string categorical features (dytpe `object`):
        
    categorical_columns = working_train_df.select_dtypes(include=['object']).columns    
    binary_cols =[col for col in categorical_columns if working_train_df[col].nunique() == 2]    
    one_hot_cols = [col for col in categorical_columns if working_train_df[col].nunique() > 2]
   
    ordinal_enc = OrdinalEncoder()
    ordinal_enc.fit(working_train_df[binary_cols])
    
    working_train_df[binary_cols] = ordinal_enc.transform(working_train_df[binary_cols])
    working_val_df[binary_cols] = ordinal_enc.transform(working_val_df[binary_cols])
    working_test_df[binary_cols] = ordinal_enc.transform(working_test_df[binary_cols])
    
    one_hot_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')    
    one_hot_enc.fit(working_train_df[one_hot_cols])
    
    train_one_hot_encoded = one_hot_enc.transform(working_train_df[one_hot_cols])
    val_one_hot_encoded = one_hot_enc.transform(working_val_df[one_hot_cols])
    test_one_hot_encoded = one_hot_enc.transform(working_test_df[one_hot_cols])
    
    train_one_hot_encoded_df = pd.DataFrame(train_one_hot_encoded, columns=one_hot_enc.get_feature_names_out(one_hot_cols), index=working_train_df.index)    
    val_one_hot_encoded_df = pd.DataFrame(val_one_hot_encoded, columns=one_hot_enc.get_feature_names_out(one_hot_cols), index=working_val_df.index)    
    test_one_hot_encoded_df = pd.DataFrame(test_one_hot_encoded, columns=one_hot_enc.get_feature_names_out(one_hot_cols), index=working_test_df.index)
 
    working_train_df = pd.concat([working_train_df.drop(columns=one_hot_cols), train_one_hot_encoded_df], axis=1)
    working_val_df = pd.concat([working_val_df.drop(columns=one_hot_cols), val_one_hot_encoded_df], axis=1)
    working_test_df = pd.concat([working_test_df.drop(columns=one_hot_cols), test_one_hot_encoded_df], axis=1)  
               
    # 3. Impute values for all columns with missing data or, just all the columns.
       
    imputer = SimpleImputer(strategy='median')
    
    working_train_df = imputer.fit_transform(working_train_df)
    working_val_df = imputer.transform(working_val_df)
    working_test_df = imputer.transform(working_test_df)
          
    # 4.Feature scaling with Min-Max scaler. 
    
    scaler = MinMaxScaler()
    
    working_train_df = scaler.fit_transform(working_train_df)
    working_val_df = scaler.transform(working_val_df)
    working_test_df = scaler.transform(working_test_df)
            
    return working_train_df, working_val_df, working_test_df


