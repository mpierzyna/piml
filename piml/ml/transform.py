from sklearn.base import BaseEstimator, TransformerMixin


class DimToPiTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y, **fit_params)

    def fit(self, X, y=None, **fit_params) -> "DimToPiTransformer":
        ...

    def transform(self, X, y=None, **fit_params):
        ...