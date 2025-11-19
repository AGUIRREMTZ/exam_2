"""
Custom pipelines for data preprocessing.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def create_numeric_pipeline():
    """
    Create a pipeline for numeric features.
    
    Returns:
        Pipeline: Sklearn pipeline for numeric data
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('rbst_scaler', RobustScaler()),
    ])


def create_full_pipeline(X_train):
    """
    Create a full preprocessing pipeline for both numeric and categorical features.
    
    Args:
        X_train (pd.DataFrame): Training data to determine feature types
        
    Returns:
        ColumnTransformer: Full preprocessing pipeline
    """
    num_attribs = list(X_train.select_dtypes(exclude=['object']).columns)
    cat_attribs = list(X_train.select_dtypes(include=['object']).columns)
    
    num_pipeline = create_numeric_pipeline()
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
    ])
    
    return full_pipeline
