from sklearn.preprocessing import StandardScaler
import pandas as pd
class transformation:
  def __init__(self):
    pass
  def log_dist(self,data:pd.DataFrame)->pd.DataFrame:
    try:
      final_log_transform_data=data/2.7183
      final_log_transform_data.to_csv('all_dataset/after_log_transform_data.csv')
      return final_log_transform_data
    except  Exception as e:
      raise e

  def std_scaler_dist(self,data:pd.DataFrame)->pd.DataFrame:
    try:
      columns=data.columns
      scaler=StandardScaler()
      np_transform_data=scaler.fit_transform(data)
      final_stdScaler_transform_data=pd.DataFrame(data=np_transform_data,columns=columns)
      final_stdScaler_transform_data.to_csv('all_dataset/after_stdScaler_transform_data.csv')
      return final_stdScaler_transform_data
    except Exception as e:
      raise e