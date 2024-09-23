from sklearn.metrics import make_scorer, log_loss
import numpy as np
import pandas as pd

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
from Preprocessing import Preprocessor


def MultiLabelLogLoss(y_true, y_predict):
    """To assess the quality of customised models,
    I use a function that calculates the average LogLoss error value for all target columns."""
    total_loss = 0
    multi = y_true.shape[1]
    y_true = y_true.to_numpy()
    y_predict = np.array(y_predict)
    for label in range(multi):
        if len(y_predict.shape) == 3:
            label_loss = log_loss(labels=[0, 1], y_true=y_true[:, label], y_pred=y_predict[label, :, :])
        if len(y_predict.shape) == 2:
            label_loss = log_loss(labels=[0, 1], y_true=y_true[:, label], y_pred=y_predict[:, label])
        total_loss += label_loss
    return total_loss / multi


multi_log_loss = make_scorer(MultiLabelLogLoss, greater_is_better=False, needs_proba=True)

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import BaseCrossValidator, KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class CustomCV(BaseCrossValidator):
    """Let's also separate cross-validation into a separate class to experiment with stratification."""

    def __init__(self, stratified_X_cols=None, stratified_y_cols=None, n_splits=5, stratified=False,
                 random_state=42):
        self.n_splits = n_splits
        self.stratified = stratified
        self.stratified_X_cols = stratified_X_cols
        self.stratified_y_cols = stratified_y_cols
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if self.stratified:
            msss = MultilabelStratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            if self.stratified_X_cols:
                # it's adivisible to use one-hot-encoding before stratification,
                # since it helps to distribute groups across folds in a more balanced way
                ohe = OneHotEncoder()
                ohe_values = ohe.fit_transform(X[self.stratified_X_cols]).toarray().astype(int)
                labels = pd.DataFrame(ohe_values, columns=list(ohe.get_feature_names_out(self.stratified_X_cols)))

                if self.stratified_y_cols: labels = pd.concat([labels, y[self.stratified_y_cols]])

            elif self.stratified_y_cols:
                labels = y[self.stratified_y_cols]
            else:
                labels = y

            return msss.split(X, labels)

        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            return cv.split(X, y)


from itertools import product


def create_custom_param_grid(param_grid, consistent_groups=None):
    custom_param_grid = []

    # Identify all the skip parameters
    skip_params = {key: value for key, value in param_grid.items() if 'skip' in key}

    # Check consistency in groups
    consistent_values = {}
    if consistent_groups:
        for group in consistent_groups:
            group_values = [param_grid[param] for param in group]
            # Ensure all parameters in a group have the same values
            if not all(group_values[0] == gv for gv in group_values):
                raise ValueError(f"Inconsistent values found in the group {group}")
            consistent_values[frozenset(group)] = group_values[0]

    # Helper function to create the custom grid
    def expand_grid(current_params, remaining_params):
        if not remaining_params:
            custom_param_grid.append(current_params.copy())
            return

        skip_param, values = remaining_params[0]
        prefix = skip_param.split('skip')[0]

        for skip_value in values:
            current_params[skip_param] = [skip_value]  # Wrap value in a list
            if skip_value:
                # When skip is True, set related params to None
                related_params = [key for key in param_grid if key.startswith(prefix) and key != skip_param]
                for rp in related_params:
                    if 'skip' not in rp:  # Avoid modifying other skip flags
                        current_params[rp] = [None]  # Wrap value in a list
                expand_grid(current_params, remaining_params[1:])
            else:
                # When skip is False, explore all combinations of related params
                related_params = [key for key in param_grid if key.startswith(prefix) and key != skip_param]
                related_values = [param_grid[rp] for rp in related_params if 'skip' not in rp]
                product_params = [dict(zip(related_params, v)) for v in product(*related_values)]

                for prod in product_params:
                    for key, value in prod.items():
                        prod[key] = [value]  # Wrap value in a list
                    current_params.update(prod)
                    expand_grid(current_params, remaining_params[1:])

    remaining_skip_params = list(skip_params.items())
    expand_grid({}, remaining_skip_params)

    # Handle multiple consistent groups
    if consistent_groups:
        consistent_products = []
        for group in consistent_groups:
            values = consistent_values[frozenset(group)]
            group_product = [{param: [value] for param in group} for value in values]  # Wrap value in a list
            consistent_products.append(group_product)

        # Generate all combinations of consistent groups
        combined_consistent_products = list(product(*consistent_products))

        final_grid = []
        for base_params in custom_param_grid:
            for consistent_combination in combined_consistent_products:
                combined_params = base_params.copy()
                for group_values in consistent_combination:
                    combined_params.update(group_values)
                final_grid.append(combined_params)
        return final_grid

    return custom_param_grid


from sklearn.model_selection import GridSearchCV, ParameterGrid
import numpy as np


class CustomGridSearchCV(GridSearchCV):
    def __init__(self, estimator, param_grid, cv_param_grid, **kwargs):
        super().__init__(estimator, param_grid, cv=None, **kwargs)
        self.cv_param_grid = cv_param_grid

    def fit(self, X, y=None, **fit_params):
        cv_results = {}
        existing_params = set()
        for cv_params in ParameterGrid(self.cv_param_grid):
            self.cv = CustomCV(**cv_params)
            super().fit(X, y, **fit_params)

            #add current cv_params to the results of grid_search
            for param, value in cv_params.items():
                self.cv_results_[param] = [value] * len(self.cv_results_[next(iter(self.cv_results_))])

            existing_params.update(self.cv_results_.keys())
            left_params = existing_params.copy()

            # #concatenate results of current grid_search with the previous ones
            if cv_results:
                n = len(cv_results[next(iter(cv_results))])
                for param, values in sorted(self.cv_results_.items()):
                    if param in cv_results:
                        cv_results[param] += list(values)
                    else:
                        cv_results[param] = ['-'] * n + list(values)
                    left_params.discard(param)
                for param in left_params:
                    cv_results[param] += ['-'] * len(self.cv_results_[next(iter(self.cv_results_))])
            else:
                for param, values in sorted(self.cv_results_.items()):
                    cv_results[param] = list(values)

        self.cv_results_ = cv_results
        return self