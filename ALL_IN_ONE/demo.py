from source_code.hyper_parameter import hyper_parameter_classifier
from source_code.detect_outlierANDremove import detect_remove_outliers
from source_code.handle_imbalanced_dataset import handle_imbalanced_data
from source_code.diamensionalityReduction import diamensionality_reduction
from source_code.remove_unwntedColumns import remove_col
from source_code.find_Corr_remove import find_correlation
from source_code.transformation import transformation
from source_code.classifierTrainer import non_hyper_parameter_classifier_model
from source_code.replace_NaN import replace_nan
from sklearn.datasets import load_iris
from source_code.handle_missing_value_in_catData import replace_nan_categorical_data
import pandas as pd
import numpy as np
import joblib
import sys,os

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
feature = pd.DataFrame(np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]))
label = pd.DataFrame(raw_df.values[1::2, 2])

from sklearn.linear_model import LinearRegression
df=pd.read_csv('/config/workspace/ALL_IN_ONE/all_datasets/demo.csv')
x=df.drop(columns=['label'])
y=df['label']
from all_models.combine_all import combine_all_functions
cf=combine_all_functions()
path='/config/workspace/ALL_IN_ONE/internet_service_churn.csv'
df=pd.read_csv(path)
# feature=df.drop(columns=['Survived'])
# label=df['Survived']
# nan=replace_nan_categorical_data()
# df=nan.combine_all(feature)
# print(df.head())

# dc=detect_remove_outliers()
# dc.remove_outlier(feature,out_)
dic=cf._combine_all_data_preprocessing(path,'churn')
#train=non_hyper_parameter_classifier_model()

# train.split_data_training(a,b,hyper_parameter=True)


#Index(['0', '1', '2', '4', '5', '6', '7', '9', '10', '11', '12']