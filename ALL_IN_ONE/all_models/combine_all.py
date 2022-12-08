import pandas as pd
import numpy as np
from source_code.classifierTrainer import non_hyper_parameter_classifier_model
from source_code.handle_missing_value_in_catData import replace_nan_categorical_data
from source_code.hyper_parameter import hyper_parameter_classifier
from source_code.detect_outlierANDremove import detect_remove_outliers
from source_code.handle_imbalanced_dataset import handle_imbalanced_data
from source_code.diamensionalityReduction import diamensionality_reduction
from source_code.remove_unwntedColumns import remove_col
from source_code.find_Corr_remove import find_correlation
from source_code.transformation import transformation
from source_code.replace_NaN import replace_nan
from source_code.handle_categorical_features import cat_value


class combine_all_functions:
    def __init__(self):
        non_hyper_parameter_classifier_model_obj=non_hyper_parameter_classifier_model()
        self.replace_nan_categorical_data_obj=replace_nan_categorical_data()
        self.hyper_parameter_classifier_obj=hyper_parameter_classifier()
        self.detect_remove_outliers_obj=detect_remove_outliers()
        self.handle_imbalanced_data_obj=handle_imbalanced_data()
        self.diamensionality_reduction_obj=diamensionality_reduction()
        self.remove_col_obj=remove_col()
        self.find_correlation_obj=find_correlation()
        self.transformation_obj=transformation()
        self.replace_nan_obj=replace_nan()
        self.cat_value_obj=cat_value()
        

    def is_imbalanced(self,label:pd.DataFrame,cl_name:str)->bool:
        label_li=label[cl_name].values.tolist()
        all_count_li=[]
        is_im=False
        for i in df[0].unique():
            all_count_li.append(label_li.count(i))
        for i in all_count_li:
            for j in all_count_li:
                if  ((i-j)/len(df))>0.20:
                    is_im=True
                    return is_im
        return is_im
    def _combine_all_data_preprocessing(self,path:str,label_column:str,isClassification=True):
        raw_data=pd.read_csv(path)
        feature=raw_data.drop(columns=label_column)
        label=raw_data[label_column]
        replace_nan_cat_data=self.replace_nan_categorical_data_obj.combine_all(feature)
        print(f'===================done replace_nan_CAT=========================')
        handle_cat_data=self.cat_value_obj.combine_all(replace_nan_cat_data)
        print(f'===================done handle_CAT data=========================')
        replace_nan_data=self.replace_nan_obj.replace_nan_knnimpute(handle_cat_data)
        print(f'===================done replace_nan=========================')
        if isClassification:
            TorF=self.is_imbalanced(pd.DataFrame(label,columns=['label']),cl_name='label')
            if TorF:
                replace_nan_data,label=handle_imbalanced_data=self.handle_imbalanced_data_obj.using_smotetomek(replace_nan_data,label)
        
        find_corr_data=self.find_correlation_obj.remove_corr_col(replace_nan_data)
        print(f'===================done find corr data=========================')
        detect_remove_outliers_data=self.detect_remove_outliers_obj.remove_outlier(find_corr_data)
        print(f'===================done remove outliers data=========================')
        transformation_data=self.transformation_obj.std_scaler_dist(detect_remove_outliers_data)
        print(f'===================done transformation data=========================')
        return transformation_data,label

    def model_trainer(self,path:str,label_column=None,isClassification=True):
        final_pre_process_data,label=self._combine_all_data_preprocessing(data)





