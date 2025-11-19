"""
Custom transformers for data preprocessing.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder


class DeleteNanRows(BaseEstimator, TransformerMixin):
    """
    Transformer to delete rows with NaN values.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.dropna()


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler that applies RobustScaler to specified attributes.
    """
    def __init__(self, attributes):
        self.attributes = attributes
        self.scaler = None
    
    def fit(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        self.scaler = RobustScaler()
        self.scaler.fit(scale_attrs)
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        X_scaled = self.scaler.transform(scale_attrs)
        X_scaled = pd.DataFrame(
            X_scaled, 
            columns=self.attributes, 
            index=X_copy.index
        )
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        return X_copy


class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    """
    Custom One-Hot Encoder that returns a DataFrame.
    """
    def __init__(self):
        self._oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self._columns = None
        self._cat_columns = None
    
    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._cat_columns = X_cat.columns.tolist()
        self._oh.fit(X_cat)
        # Generate column names
        self._columns = []
        for i, col in enumerate(self._cat_columns):
            categories = self._oh.categories_[i]
            for cat in categories:
                self._columns.append(f"{col}_{cat}")
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy.select_dtypes(include=['object'])
        X_num = X_copy.select_dtypes(exclude=['object'])
        
        if len(X_cat.columns) > 0:
            X_cat_oh = self._oh.transform(X_cat)
            X_cat_oh = pd.DataFrame(
                X_cat_oh, 
                columns=self._columns, 
                index=X_copy.index
            )
            # Drop original categorical columns
            X_copy = X_copy.drop(columns=X_cat.columns)
            # Join with one-hot encoded columns
            X_copy = X_copy.join(X_cat_oh)
        
        return X_copy


class SelectNumericFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer to select only numeric features.
    """
    def __init__(self):
        self.numeric_columns = None
    
    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(exclude=['object']).columns.tolist()
        return self
    
    def transform(self, X, y=None):
        return X[self.numeric_columns]


class SelectCategoricalFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer to select only categorical features.
    """
    def __init__(self):
        self.categorical_columns = None
    
    def fit(self, X, y=None):
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        return self
    
    def transform(self, X, y=None):
        return X[self.categorical_columns]
