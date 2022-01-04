import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['adult_male'] = (X['sex']=='male') & ~(X['age']<16)
        X_transformed['who'] = np.where(X['age']<16, 'child', 
                                        np.where(X['sex']=='female', 'woman', 'man'))
        return X

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, features, method='constant', value='missing'):
        self.features = features
        self.method = method
        self.value = value
    
    def fit(self, X, y=None):
        if self.method=='mean':
            self.value = X[self.features].mean()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = X[self.features].fillna(self.value)
        return X_transformed

class CardinalityReducer(BaseEstimator, TransformerMixin):
    def __init__(self, features, threshold=.1):
        self.features = features
        self.threshold = threshold
        
    def find_top_categories(self, feature):
        if self.threshold>=1:
            counts = feature.value_counts().head(self.threshold)
            categories = counts.index.values
        elif self.threshold>0:
            counts = feature.value_counts(normalize=True)
            categories = counts[counts>=self.threshold].index.values
        return categories
    
    def fit(self, X, y=None):
        self.categories = {}
        for feature in self.features:
            self.categories[feature] = self.find_top_categories(X[feature])
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for feature in self.features:
            X_transformed[feature] = np.where(X[feature].isin(self.categories[feature]), X[feature], 'other')
        return X_transformed

class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop='first'):
        self.features = features
        self.drop = drop
    
    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse=False, drop=self.drop)
        self.encoder.fit(X[self.features])
        return self
    
    def transform(self, X):
        X_transformed = pd.concat([X.drop(columns=self.features).reset_index(drop=True), 
                                   pd.DataFrame(self.encoder.transform(X[self.features]), 
                                                columns=self.encoder.get_feature_names_out(self.features))],
                                  axis=1)
        return X_transformed