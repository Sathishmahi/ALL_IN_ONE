# import pandas as pd
# df=pd.read_csv('/config/workspace/melb_data.csv')
# print(df.shape)
# print('===================================================================================')
# print(df.head(1))
# print('===================================================================================')
# print(df.isna().sum())
# print('===================================================================================')
# print(df.dtypes)
# print('===================================================================================')
# print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
# df=pd.read_csv('/config/workspace/ALL_IN_ONE/all_datasets/after_stdScaler_transform_data.csv')

# print(df.shape)
# print('===================================================================================')
# print(df.head(1))
# print('===================================================================================')
# print(df.isna().sum())
# print('===================================================================================')
# print(df.dtypes)
# print('===================================================================================')
# print('===================================================================================')
from source_code.classifierTrainer import non_hyper_parameter_classifier_model
train=non_hyper_parameter_classifier_model()
from all_models.combine_all import combine_all_functions
cf=combine_all_functions()

x,y=cf._combine_all_data_preprocessing('/config/workspace/ALL_IN_ONE/all_datasets/demo.csv','label',False)
print(x.isna().any())
print(x.head(1))
# x.to_csv('all_datasets/x.csv')
# y.to_csv('all_datasets/y.csv')
# print('DONE')
df = x.loc[:, ~x.columns.str.contains('^Unnamed')]
#label= y.loc[:, ~y.columns.str.contains('^Unnamed')]
train.split_data_training(df,y,hyper_parameter=True)
# print(x.head())
# print(x.isna().sum())