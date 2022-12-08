import pandas as pd
import numpy as np
from source_code.classifierTrainer import non_hyper_parameter_classifier_model
from source_code.handle_missing_value_in_catData import replace_nan_categorical_data
from source_code.handle_categorical_features import cat_value
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
feature = pd.DataFrame(np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]))
label = pd.DataFrame(raw_df.values[1::2, 2])

data=pd.read_csv('/config/workspace/melb_data.csv')
print(data.isna().sum())
print('-----------------------------------------------------------------------')
cv=replace_nan_categorical_data()
d=cv.combine_all(data)
print(d.isna().sum())

# train=non_hyper_parameter_classifier_model()
# a,b=train.model_predicted(feature)
# print(len(a))
# print(len(b))
#train.split_data_training(feature,label,hyper_parameter=True)
