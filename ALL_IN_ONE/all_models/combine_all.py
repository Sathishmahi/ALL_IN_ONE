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
from source_code.remove_unwntedColumns import remove_col
from source_code.train_test_split import train_test_split_fn
from source_code.exception import CustomException
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import sys
#from source_code.model_score import return_model_score

from source_code.classifierTrainer import score_dict
all_classification_score=dict()

class combine_all_functions:
    def __init__(self):
        self.kmeans_col_li=[]
        self.non_hyper_parameter_classifier_model_obj=non_hyper_parameter_classifier_model()
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
       # self.return_model_score_obj=return_model_score()
        

    def is_imbalanced(self,label:pd.DataFrame,cl_name:str)->bool:
        label_li=label[cl_name].values.tolist()
        all_count_li=[]
        is_im=False
        for i in label[cl_name].unique():
            all_count_li.append(label_li.count(i))
        for i in all_count_li:
            for j in all_count_li:
                if  ((i-j)/len(df))>0.20:
                    is_im=True
                    return is_im
        return is_im
    def demo(self,feature:pd.DataFrame,label:pd.DataFrame,isClassification=True,predict=False)->pd.DataFrame:
        handle_cat_data=self.cat_value_obj.combine_all(feature)
        print(f'===================done handle_CAT data=========================')
        replace_nan_cat_data=self.replace_nan_categorical_data_obj.combine_all(handle_cat_data)
        print(f'===================done replace_nan_CAT=========================')
        
       
        #replace_nan_data=self.replace_nan_obj.mean_median_mode(feature)
        replace_nan_data=self.replace_nan_obj.replace_nan_knnimpute(replace_nan_cat_data)
        print(f'===================done replace_nan=========================')
        if isClassification:
            TorF=self.is_imbalanced(pd.DataFrame(label,columns=['label']),cl_name='label')
            if TorF:
                replace_nan_data,label=handle_imbalanced_data=self.handle_imbalanced_data_obj.using_smotetomek(replace_nan_data,label)
        
        find_corr_data=self.find_correlation_obj.remove_corr_col(replace_nan_data)
        #find_corr_data=self.find_correlation_obj.remove_corr_col(feature)
        print(f'===================done find corr data=========================')
        # find_corr_data['label']=raw_data[label_column]
        #transformation_data=self.transformation_obj.std_scaler_dist(detect_remove_outliers_data)
        transformation_data=self.transformation_obj.std_scaler_dist(find_corr_data)
        print(f'===================done transformation data=========================')
        final_data,label=self.detect_remove_outliers_obj.remove_outlier(transformation_data,label)

        # transformation_data=self.transformation_obj.std_scaler_dist(feature)
        print(f'===================done for outlier removed data=========================')
        if predict==False:
            final_data=self.remove_col_obj.all_columns_remove(final_data)
            [self.kmeans_col_li.append(col) for col in final_data.columns]
            print(f'===================done for remove unwanted columns=========================')
        else:
            print('True')
            print(self.kmeans_col_li)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            final_data=final_data[self.kmeans_col_li]

            return final_data,label
        return final_data,label
    def model_trainer(self,feature:pd.DataFrame,label:pd.DataFrame,isClassification=True):
        #final_pre_process_data,label=self._combine_all_data_preprocessing(path,label_column,isClassification)
        #print(final_pre_process_data.isna().sum())
        self.non_hyper_parameter_classifier_model_obj.split_data_training(feature,label,hyper_parameter=True)
        print('++++++++++++++++++++++++++++++++ training compleate ++++++++++++++++++++++++++++++++++++++++++')
    def model_predict(self,data:pd.DataFrame):
        df_li,out_li=self.non_hyper_parameter_classifier_model_obj.model_predicted(data)
        return df_li,out_li
    
    def classification_model_score(self,y_pre,y_true):
        print(f'score dict =========>     {score_dict}')

        try:
            for counter,ind in enumerate(score_dict.values()):
                print(ind)
                print('======TRUE=====',y_true)
                
                y_true=y_true['Survived'].iloc[list(ind)]
                print(f' y true =======>   {y_true}')
                accuracy=accuracy_score(y_true,y_pre[counter])
                all_classification_score.update({f"accuracy_score_{counter}":accuracy})
                precision=precision_score(y_true,y_pre[counter])
                all_classification_score.update({f"precision_score_{counter}":precision})
                recall=recall_score(y_true,y_pre[counter])
                all_classification_score.update({f"recall_score_{counter}":recall})
                f1score=f1_score(y_true,y_pre[counter])
                all_classification_score.update({f"recall_score_{counter}":recall})
                confusion_matrix_model=confusion_matrix(y_true,y_pre[counter])
                all_classification_score.update({f"confusion_matrix_{counter}":confusion_matrix_model})
            return all_classification_score
        except:
            CustomException(sys)

    def _combine_all_data_preprocessing(self,path:str,label_column:str,isClassification=True):
        raw_data=pd.read_csv(path)
        feature=raw_data.drop(columns=label_column)
        label=raw_data[label_column]
        x_train,x_test,y_train,y_test=train_test_split_fn(feature=feature,label=label)
        print(x_train.shape)
        data_list=[x_train,x_test,y_train,y_test]

       
        train_feature,train_label=self.demo(x_train,y_train,isClassification)
        print('==================================================')
        print(train_feature.columns)
        print('==================================================')

        self.model_trainer(train_feature,train_label)
        print('+++Train Complete++++')
        test_faeture,test_label=self.demo(x_test,y_test,isClassification,True)
        print('==================================================')
        print(test_faeture.columns)
        print('==================================================')
        df_li,out_li=self.model_predict(test_faeture)
        dic=self.classification_model_score(out_li,test_label)
        print(dic)
        return dic
        # for x in data_list:
        #     for y in data_list:
        #         if x.shape[0]==y.shape[0] and x.shape[1]!=y.shape[1]:
        #             print(x.shape,y.shape)
        #             train_feature,train_label=self.demo(x_train,y_train,isClassification)
        #             self.model_trainer()
        #             print('Train Complete')
        #         elif (x.shape[0]==y.shape[0]) and (x.shape[1]!=y.shape[1]):
        #             test_faeture,test_label=self.demo(x_test,y_test,isClassification)
        #             df_li,out_li=self.model_predict(test_feature)
        #             return df_li,out_li
        #          print('end')

                    

                    


    



