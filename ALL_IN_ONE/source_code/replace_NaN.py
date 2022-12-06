from sklearn.impute import KNNImputer
import pandas as pd

class replace_nan:
  def __init__(self):
    pass
  def replace_nan_knnimpute(self,data:pd.DataFrame)->pd.DataFrame:
    try:
      imputer = KNNImputer(n_neighbors=3)
      columns=data.columns
      After_imputation_data = imputer.fit_transform(data)
      After_imputation_data=pd.DataFrame(data=After_imputation_data,columns=columns)
      After_imputation_data.to_csv('all_datasets/after_FillNA_data.csv')
      return After_imputation_data
    except Exception as e:
      raise e
  def mean_median_mode(self,data:pd.DataFrame)->pd.DataFrame:
    try:
      for col in data.columns:
        if data[col].isna().sum()>0:
          if data[col].nunique()/len(data)>=0.1:
            
            data[col].fillna(data[col].mode(),inplace=True)
            
          else:
            print(data[col].mean())
            data[col].fillna(data[col].mean(),inplace=True)
      data.to_csv('all_datasets/after_FillNA_data.csv')
      return data

    except Exception as e:
      raise e