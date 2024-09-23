"""PREPROCESSING PIPELINE:
For data preprocessing, I will use a pipeline, which, depending on the type of column,
will produce 3 main transformations:

1. CLEANING: removing some of the columns (uninformative or highly correlated with others), counting the number of
missing values in a row (extra feature). It is necessary to separate the first step of the pipeline into a separate
class, so that no algorithms are applied to the data for validation, but simply select those columns that remain in
the data for training.

2. IMPUTING & SCALING: imputing missing values using one of the algorithms: -SimpleImputer - imputing with a constant
value that obviously goes beyond the boundaries of the existing data (a solid choice for decision trees e.g). In our
case, a large negative value is suitable, because the dataset is filled with positive values only -KNNImputer - we
will experiment with the number of neighbors -IterativeImputer  - an algorithm that considers each column in the
dataset as a function of the others, trains a model using a selected algorithm, and imputes missing values using the
prediction

After imputation, the data is optionally normalized (where this is important) and the most informative columns are
selected using PCA

3. ENCODING: for ML algorithms that work only with numbers, produce one-hot encoding or another type of encoding of
categorical columns, depending on the statistics of their unique values

These steps will be implemented as classes equipped with "fit" and "transform" methods.

To assemble a model in which the preprocessing will be the first step before applying the ML algorithm, I will create
a separate class, called "Preprocessor",  that will collect all the transformers described above using Pipeline and
ColumnTransformer. ColumnTransformer will allow to separately deal with each of the three data types, and the fit
and transform methods will allow to use the resulting pipeline during cross-validation. In such case only the
training data will be used to configure the algorithm, and only the resulting transformations will be applied to the
validation data.

To make the ColumnTransformer work one needs to specify a list of columns to which the transformation will be
applied. Since I throw out a certain a priori unknown number of columns at the first step, I need to create the list
of columns for the ColumnTransformer dynamically. To do so, I create a class "make_column_selector" in which I
specify the pattern for selecting columns. I indicate the column type as a pattern, and for the categorical type I
also separate columns with a small and large number of unique values (the exact division boundary is set
experimentally during GridSearch)."""

import numpy as np
import pandas as pd
from collections import defaultdict
import math
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler


def columns_stat(df: pd.DataFrame):
    columns = df.columns.tolist()
    columns_categories = defaultdict(list)
    for col in columns:
        if col == 'release':
            columns_categories['release'].append(col)
        elif str(col)[0].isalpha():
            columns_categories[col[0]].append(col)
    return columns_categories


def columns_catstat(columns_categories):
    return [(key, len(columns_categories[key])) for key in columns_categories.keys()]


class MissRatioRemoval(BaseEstimator, TransformerMixin):
    """removing columns based on the percentage of missing values"""

    def __init__(self, thresh=0.99):
        self.thresh = thresh
        self.feature_names_out = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df_removed, self.feature_names_out = self.miss_ratio_removal(X)
        return df_removed

    def get_feature_names_out(self, df=None):
        return self.feature_names_out

    def get_params(self, deep=True):
        return {"thresh": self.thresh}

    def miss_ratio_removal(self, df):
        columns = df.columns.tolist()
        cols_to_remove = []
        for col in columns:
            miss_ratio = df[col].isnull().sum() / len(df)
            if miss_ratio >= self.thresh:
                cols_to_remove.append(col)
        columns = [col for col in columns if col not in cols_to_remove]
        df_removed = df.drop(columns=cols_to_remove, inplace=False)
        output = ''
        output += '>>>>>MissRatioThresholdRemoval with ' + str(self.thresh * 100) + '%.'
        output += ' Before:' + str(columns_catstat(columns_stat(df)))
        output += ' After:' + str(columns_catstat(columns_stat(df_removed)))
        print(output)
        return df_removed, columns


def find_correlations(df, columns, thresh):
    corr_dict = dict()
    lengths = {col: df[col].count() for col in columns}
    for i, col1 in enumerate(columns):
        len1 = lengths[col1]
        for j, col2 in enumerate(columns[i + 1:]):
            len2 = lengths[col2]
            if df[col1].std() < 0.05 or df[col1].std() < 0.05:
                continue
            if (0.5 < len1 / len2 <= 1) or (0.5 < len2 / len1 <= 1):
                corr = df[col1].corr(df[col2], method='pearson', min_periods=round(0.5 * max(len1, len2)) + 1)
                if corr >= thresh or corr <= -thresh:
                    corr_dict[(col1, col2, len1, len2)] = corr
            else:
                continue
    return corr_dict


class CorrelationRemoval(BaseEstimator, TransformerMixin):
    """removing columns that are highly correlated with others in the dataset"""

    def __init__(self, thresh=0.99):
        self.thresh = thresh
        self.feature_names_out = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df_removed, self.feature_names_out = self.correlation_removal(X)
        return df_removed

    def get_feature_names_out(self, df=None):
        return self.feature_names_out

    def get_params(self, deep=True):
        return {"thresh": self.thresh}

    def correlation_removal(self, df):
        columns = df.columns.tolist()
        cols_to_remove = set()
        cors = find_correlations(df, columns, self.thresh)
        for key in cors:
            cols_to_remove.add(key[int(key[2]) > int(key[3])])
        cols_to_remove = list(cols_to_remove)
        columns = [col for col in columns if col not in cols_to_remove]
        df_removed = df.drop(columns=cols_to_remove, inplace=False)
        return df_removed, columns


class NMissingEstimator(BaseEstimator, TransformerMixin):
    """Let's add the number of empty values in a row as a separate feature.
     I will add a parameter to skip this step and experiment with this functionality during GridSearchCV."""

    def __init__(self):
        self.feature_names_out = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_trans = X.copy()
        self.feature_names_out = X_trans.columns.tolist()
        num_missing = X_trans.isnull().sum(axis=1).tolist()
        X_trans['#missing'] = num_missing
        return X_trans

    def get_feature_names_out(self, df=None):
        return self.feature_names_out


def select_all_columns(df):
    return df.columns.tolist()


class make_column_selector():
    """To make the ColumnTransformer work one needs to specify a list of columns to which the transformation will be applied.
    Since I throw out a certain a priori unknown number of columns at the first step,
    I need to create the list of columns for the ColumnTransformer dynamically.
    To do so, I create a class "make_column_selector" in which I specify the pattern for selecting columns.
    I indicate the column type as a pattern, and for the categorical type I also separate columns
    with a small and large number of unique values (the exact division boundary is set experimentally during GridSearch)."""

    def __init__(self, pattern=None, cardinality=None, cardinality_threshold=2):

        self.pattern = pattern
        self.cardinality = cardinality
        self.cardinality_threshold = cardinality_threshold

    def __call__(self, df):

        cols = df.columns

        if self.pattern:
            cols = cols[cols.str.contains(self.pattern, regex=True)]

        if self.cardinality:
            cols_cardinality = dict(df[cols].nunique(axis=0, dropna=False))
            if self.cardinality == 'high':
                cols = [c for c in cols_cardinality if cols_cardinality[c] > self.cardinality_threshold]
            elif self.cardinality == 'low':
                cols = [c for c in cols_cardinality if cols_cardinality[c] <= self.cardinality_threshold]

        return cols


from sklearn.base import TransformerMixin


class DynamicColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, pattern, cardinality=None, threshold=None):
        self.threshold = threshold
        self.pattern = pattern
        self.cardinality = cardinality
        self.selector = make_column_selector(pattern=pattern, cardinality=cardinality, cardinality_threshold=threshold)
        self.estimator = estimator
        self.feature_names_out = []

    def fit(self, X, y=None):
        cols = self.selector(X)
        self.feature_names_out = cols
        self.estimator.fit(X[cols], y)
        return self

    def transform(self, X, y=None):
        return self.estimator.transform(X)

    def get_feature_names_out(self, df=None):
        return self.feature_names_out


class SkippableCleaner(BaseEstimator, TransformerMixin):
    """let's separate the first step of the pipeline into a separate class, which will delete columns.
    This is necessary so that no algorithms are applied to the data for validation, but simply select
    those columns that remain in the data for training.
    I also add a parameter, so that I can skip the whole cleaning step."""

    def __init__(self, skip=False, corr_remove=False, count_missing=False,
                 num_mrr_thresh=0.99, ord_mrr_thresh=0.99, cat_mrr_thresh=0.99,
                 num_corr_thresh=0.99, ord_corr_thresh=0.99,
                 ):

        self.feature_names_out = []
        self.skip = skip
        self.corr_remove = corr_remove
        self.count_missing = count_missing
        self.num_mrr_thresh = num_mrr_thresh
        self.ord_mrr_thresh = ord_mrr_thresh
        self.cat_mrr_thresh = cat_mrr_thresh
        self.num_corr_thresh = num_corr_thresh
        self.ord_corr_thresh = ord_corr_thresh
        self.mrr_threshes = {'num': num_mrr_thresh, 'ord': ord_mrr_thresh, 'cat': cat_mrr_thresh}
        self.corr_threshes = {'num': num_corr_thresh, 'ord': ord_corr_thresh}

    def fit(self, X, y=None):
        self.feature_names_out = X.columns.tolist()
        steps = [('miss_remove', ColumnTransformer([
            (cat, MissRatioRemoval(thresh=self.mrr_threshes[cat]), make_column_selector(pattern='^' + str(cat[0]))) for
            cat in self.mrr_threshes.keys()
        ], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas'))]
        if not self.skip:
            if self.corr_remove:
                steps.append(('corr_remove', ColumnTransformer([
                    (cat, CorrelationRemoval(thresh=self.corr_threshes[cat]),
                     make_column_selector(pattern='^' + str(cat[0]))) for cat in self.corr_threshes.keys()
                ], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')))
            result = Pipeline(steps).fit_transform(X)
            self.feature_names_out = result.columns.tolist()
        return self

    def transform(self, X, y=None):
        if self.count_missing:
            X_trans = NMissingEstimator().fit_transform(X[self.feature_names_out])
        else:
            X_trans = X[self.feature_names_out]
        return X_trans

    def get_feature_names_out(self, df=None):
        if self.count_missing:
            return self.feature_names_out + ['#missing']
        else:
            return self.feature_names_out


class SkippableDataFrameImputer(BaseEstimator, TransformerMixin):
    """Here I create a class for the imputation step. It can take the imputation strategy as a parameter,
    so that different algorithms can be applied in different steps and the best combination will be chosen during GridSearch.
     This step can also be skipped."""

    def __init__(self, skip=False, strategy='constant', estimator=None,
                 n_neighbors=3, weights='distance', initial_strategy='mean',
                 n_nearest_features_percentege=0.25, max_iter=10, fill_value=-10000,
                 imputation_order='ascending', random_state=42, tol=1e-3):

        self.skip = skip
        self.strategy = strategy
        self.imputer = None
        self.columns = []

        # for knn
        self.n_neighbors = n_neighbors
        self.weights = weights

        # for iterative
        self.estimator = estimator
        self.n_nearest_features_percentege = n_nearest_features_percentege
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.random_state = random_state

    def fit(self, X, y=None):

        self.columns = X.columns.tolist()

        if self.skip:
            return self

        # 'clearly out-of-range' value to fill the missing (solid choice for decision trees)
        if self.strategy == 'constant':
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=self.fill_value)

        if self.strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors, missing_values=np.nan,
                                      weights=self.weights, metric='nan_euclidean', add_indicator=False,
                                      keep_empty_features=True)

        # Multivariate imputer that estimates each feature from all the others.
        if self.strategy == 'iterative':
            n_nearest_features = int(len(self.columns) * self.n_nearest_features_percentege)
            print(n_nearest_features)
            self.imputer = IterativeImputer(estimator=self.estimator, missing_values=np.nan,
                                            max_iter=self.max_iter, ntol=self.tol,
                                            n_nearest_features=n_nearest_features,
                                            initial_strategy=self.initial_strategy,
                                            imputation_order=self.imputation_order,
                                            random_state=self.random_state)

        self.imputer.fit(X)
        return self

    def transform(self, X, y=None):

        if self.skip:
            return X
        return pd.DataFrame(self.imputer.transform(X), columns=self.columns)

    def get_feature_names_out(self, df=None):
        return self.columns


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid_v = np.vectorize(sigmoid)


class SkippableScaler(BaseEstimator, TransformerMixin):
    """I also separate normalization into a separate class, this will allow us to skip it if necessary."""

    def __init__(self, skip=False, apply=None):
        self.columns = []
        self.skip = skip
        self.apply = apply

    def fit(self, df, y=None):
        self.columns = df.columns.tolist()
        return self

    def transform(self, df, y=None):
        if self.skip:
            return df
        else:
            result = StandardScaler().fit_transform(df)
            if self.apply:
                if self.apply == 'sigmoid':
                    result = sigmoid_v(result)

            return pd.DataFrame(result, columns=self.columns)

    def get_feature_names_out(self, df=None):
        return self.columns


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class DataFrameOrdinalEncoder(BaseEstimator, TransformerMixin):
    """customised OrdinalEncoder that can be skipped, returns a pandas dataframe instead of a numpy array,
    and preserves the names of columns, so that the ColumnTransformer can be used afterward"""

    def __init__(self, **kwargs):
        self.encoder = OrdinalEncoder(**kwargs)
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.columns.tolist()
        return self

    def transform(self, X):
        self.encoder.fit(X)
        result = self.encoder.transform(X)

        return pd.DataFrame(result, columns=self.columns)

    def get_feature_names_out(self, df=None):
        return self.columns


from sklearn.decomposition import PCA
from IPython.display import display


class SkippablePCA(BaseEstimator, TransformerMixin):

    def __init__(self, skip=False, display_pca=False, category=None,
                 reduction_factor=2, n_components=None, whiten=True, **kwargs):
        self.skip = skip
        self.category = category
        self.display_pca = display_pca
        self.n_components = n_components
        self.whiten = whiten
        self.reduction_factor = reduction_factor
        self.сolumns = None
        self.feature_names_out = []
        self.pca = PCA(**kwargs)

    def fit(self, df, y=None):

        self.columns = df.columns.tolist()
        self.feature_names_out = self.columns
        if self.skip:
            return self

        params = self.pca.get_params()
        params['whiten'] = self.whiten

        n_columns = len(self.columns)

        # reducing dimension by a constant factor
        if self.n_components == None:
            if self.display_pca:
                print('PCA with redunction factor =', self.reduction_factor)
            self.n_components = int(len(self.columns) // self.reduction_factor)
            params['n_components'] = self.n_components

        elif self.n_components == 'mle':
            if self.display_pca:
                print('PCA with mle')
            params = self.pca.get_params()
            params['n_components'] = 'mle'

        self.pca.set_params(**params)
        self.pca.fit(df)

        if self.display_pca:
            print(self.category, ':', 'BEFORE', len(self.columns), 'AFTER', self.pca.n_components_)
            plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            display(plt.show())
            evr = self.pca.explained_variance_ratio_
            cvr = np.cumsum(self.pca.explained_variance_ratio_)
            pca_df = pd.DataFrame()
            pca_df['Cumulative Variance Ratio'] = cvr
            pca_df['Explained Variance Ratio'] = evr
            display(pca_df.head(10))

        self.feature_names_out = self.pca.get_feature_names_out()
        self.fitted = True
        return self

    def transform(self, df, y=None):
        if self.skip:
            return df

        return pd.DataFrame(self.pca.transform(df), columns=self.pca.get_feature_names_out())

    def get_feature_names_out(self, df=None):
        return self.feature_names_out


def Preprocessor():
    return Pipeline([
        ('cleaner', SkippableCleaner()),
        ('encoder_imputer', ColumnTransformer([

            ('num', DynamicColumnTransformer(Pipeline([
                ('num_impute', SkippableDataFrameImputer().set_output(transform="pandas")),
                ('num_scale', SkippableScaler()),
                ('num_pca', SkippablePCA(category='NUMERIC'))

            ], verbose=False), pattern='^n'), select_all_columns),

            ('ord_oh', DynamicColumnTransformer(
                OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False).set_output(
                    transform="pandas"),
                pattern='^o', cardinality='low'), select_all_columns),

            ('ord_noh', DynamicColumnTransformer(Pipeline([
                ('ord_impute', SkippableDataFrameImputer()),
                ('ord_scale', SkippableScaler()),
                ('ord_pca', SkippablePCA(category='ORDINAL'))
            ], verbose=False).set_output(transform="pandas"), pattern='^o', cardinality='high'), select_all_columns),

            ('cat_oh', DynamicColumnTransformer(
                OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False).set_output(
                    transform="pandas"),
                pattern='^c', cardinality='low'), select_all_columns),

            ('cat_noh', DynamicColumnTransformer(Pipeline([
                ('noh_enc', DataFrameOrdinalEncoder()),
                ('noh_impute', SkippableDataFrameImputer()),
                ('noh_scale', SkippableScaler()),
                ('noh_pca', SkippablePCA(category='CATEGORICAL'))

            ], verbose=False).set_output(transform="pandas"), pattern='^c', cardinality='high'), select_all_columns),

            ('release', Pipeline([
                ('enc', DataFrameOrdinalEncoder().set_output(transform="pandas")),
                ('scale', SkippableScaler(skip=False))
            ]), ['release'])

        ], remainder=SkippableScaler(skip=False), verbose_feature_names_out=False, n_jobs=-1)),

    ])
