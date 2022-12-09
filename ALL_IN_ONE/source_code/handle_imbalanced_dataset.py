from imblearn.combine import SMOTEENN,SMOTETomek
import pandas as pd
from source_code.exception import CustomException
import sys
class handle_imbalanced_data:
  
  def __init__(self):
    pass
  def using_smoteen(self,feature:pd.DataFrame,label:pd.Series)->pd.DataFrame:
    try:
      smk = SMOTEENN()
      X_res,y_res=smk.fit_resample(feature,label)
      return X_res,y_res
    except:
      raise CustomException(sys)

  def using_smotetomek(self,feature:pd.DataFrame,label:pd.Series)->pd.DataFrame:
    try:
      smk = SMOTETomek()
      X_res,y_res=smk.fit_resample(feature,label)
      x_final=pd.concat((X_res,y_res),axis=1)
      final_df.to_csv('all_datasets/after_handle_imbalanced_data.csv')
      return X_res,y_res
    except:
      raise CustomException(sys)