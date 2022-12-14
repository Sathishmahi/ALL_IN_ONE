from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import pandas as pd
from source_code.exception import CustomException
import sys, os


class diamensionality_reduction:
    def __init__(self):
        pass

    def pca_pipe(self, feature: pd.DataFrame) -> pd.DataFrame:
        try:
            scaler = StandardScaler()
            pipe = make_pipeline(scaler, PCA(n_components=0.5))
            x_pca = pipe.fit_transform(feature)
            pa = os.path.join("all_datasets")
            x_pca.to_csv("/after_diamension_reduce_data.csv")
            return x_pca
        except:
            CustomException(sys)
