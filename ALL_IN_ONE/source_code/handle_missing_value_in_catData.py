from sklearn.linear_model import LogisticRegression
import pandas as pd
from path_name_provoiders.all_names import handle_miss_val_cat_dict as dic
class replace_nan_categorical_data:
  def __init__(self):
    pass
  def replace_nan_most_frequncy_data(self,data:pd.DataFrame):
    replace_nan_col=[col for col in data.columns if ((data[col].nunique()/len(data))>0.15) and ((data[col].nunique()/len(data))<=0.55) and data[col].isna().any() and (str(data[col].dtypes).lower().startswith('o'))]
    for col in replace_nan_col:
        data[col].fillna(data[col].mode().values[0],inplace=True)
    return data
  
  def handle_miss_values_catogorical_features(self,data:pd.DataFrame):
    all_nan_col=[col for col in data.columns if (data[col].nunique()/len(data))<=0.10 and (data[col].isna().sum()/len(data))<=0.15 and data[col].isna().any()]
    feature_col=[col for col in data.columns if not data[col].isna().any() and (str(data[col].dtypes).startswith('int') or str(data[col].dtypes).startswith('float'))]
    feature=data[feature_col]
    for label in all_nan_col:
        if label in all_nan_col and (str(data[label].dtypes).lower().startswith('o') or str(data[label].dtypes).lower().startswith('bool') or str(data[label].dtypes).lower().startswith('cat')):
            print(f'---{label}----')
            [dic.update({j:i} )for i,j in enumerate(data[label].value_counts().index.tolist())]
            data[label]=data[label].map(dic)
        label_data=pd.DataFrame(data[label])
        predict_in=data[data[label].isnull()].index.tolist()

        feature_ind=[ind for ind in data.index if ind not in predict_in]
        x_train=feature.iloc[feature_ind]
        y_train=label_data.iloc[feature_ind]
        x_test=feature.iloc[predict_in]
        log=LogisticRegression()
        log.fit(x_train,y_train)
        predict_val=log.predict(x_test)
        data[label][predict_in]=predict_val
    return data
  def remove_col(self,data:pd.DataFrame):
    most_nan_col=[col for col in data.columns if (data[col].nunique()/len(data))>0.55 and data[col].isna().any() and (str(data[col].dtyes).lower().startswith('o') or str(data[col].dtyes).lower().startswith('cat') or str(data[col].dtyes).lower().startswith('bool'))]
    data.drop(columns=most_nan_col,inplace=True)
    return data
  def combine_all(self,data:pd.DataFrame):
    data=self.remove_col(data)
    data=self.handle_miss_values_catogorical_features(data)
    data=self.replace_nan_most_frequncy_data(data)
    return data
    