# from source_code.hyper_parameter import hyper_parameter_classifier
# from source_code.detect_outlierANDremove import detect_remove_outliers
# from source_code.handle_imbalanced_dataset import handle_imbalanced_data
# from source_code.diamensionalityReduction import diamensionality_reduction
# from source_code.remove_unwntedColumns import remove_col
# from source_code.find_Corr_remove import find_correlation
# from source_code.transformation import transformation
# from source_code.classifierTrainer import non_hyper_parameter_classifier_model
# from source_code.replace_NaN import replace_nan
# from sklearn.datasets import load_iris
# from source_code.handle_missing_value_in_catData import replace_nan_categorical_data
# import pandas as pd
# import numpy as np
# import joblib
# import sys, os

# # data_url = "http://lib.stat.cmu.edu/datasets/boston"
# # raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# # x = pd.DataFrame(np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]))
# # y = pd.DataFrame(raw_df.values[1::2, 2])
# df=pd.read_csv('E:\ALL_IN_ONE\ALL_IN_ONE\diabetes.csv')
# x=df.drop(columns=['Outcome'])
# y=df['Outcome']
# non=non_hyper_parameter_classifier_model()
# #non.split_data_training(x,y,hyper_parameter=True)
# out,df=non.model_predicted(x)
# non.classification_model_score(out,y)

li=[]

for i in range (2):
    print(i)
    a='demo'
    b='d'
    li.extend((a,b))


