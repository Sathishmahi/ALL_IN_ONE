import pandas as pd
import numpy as np
from source_code.classifierTrainer import non_hyper_parameter_classifier_model
from source_code.handle_missing_value_in_catData import replace_nan_categorical_data

from source_code.handle_categorical_features import cat_value

from all_models.combine_all import combine_all_functions

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
feature = pd.DataFrame(np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]))

label = pd.DataFrame(raw_df.values[1::2, 2])
label=label.set_axis(['label'],axis=1)
final=pd.concat((feature,label),axis=1)
final.to_csv('all_datasets/demo.csv')

# data=pd.read_csv('/config/workspace/melb_data.csv')
# cf=combine_all_functions()
# d=cf.combine('/config/workspace/melb_data.csv','Propertycount',False)
# # cv=replace_nan_categorical_data()
# d=cv.combine_all(data)
# print(d.isna().sum())

# train=non_hyper_parameter_classifier_model()
# # a,b=train.model_predicted(feature)
# # print(len(a))
# # print(len(b))
# train.split_data_training(feature,label,hyper_parameter=True)
