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
from source_code.classifierTrainer import non_hyper_parameter_classifier_model
from source_code.replace_NaN import replace_nan


class combine_all_functions:
    def __init__(self):
        pass
    def combine(self,path:str,label_columns:str):
        raw_data=pd.read_csv(data)
        
