import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression

class LeafModel(BaseEstimator, ClassifierMixin):
    """Гибридная модель: дерево сегментирует пространство признаков, 
    в каждом листе обучается отдельная модель (опционально с отбором признаков)."""

    def __init__(self, tree=None, leaf_model=None, feature_selector=None):
        self.tree = tree
        self.leaf_model = leaf_model
        self.feature_selector = feature_selector

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_all_finite=True, dtype='numeric')
        self.classes_ = unique_labels(y)
        self.feature_names_in_ = getattr(X, 'columns', pd.RangeIndex(X.shape[1])).tolist()

        self.tree_ = clone(self.tree) if self.tree is not None else DecisionTreeClassifier(max_leaf_nodes=5, random_state=42)
        self.tree_.fit(X, y)

        self.leaf_model_ = clone(self.leaf_model) if self.leaf_model is not None else LogisticRegression(random_state=42)

        leaf_indices = self.tree_.apply(X)
        unique_leaves = np.unique(leaf_indices)

        self.leaf_models_ = {}
        self.leaf_selectors_ = {}
        self.leaf_n_samples_ = {}
        self.leaf_pure_classes_ = {}

        for leaf in unique_leaves:
            mask = leaf_indices == leaf
            X_leaf, y_leaf = X[mask], y[mask]
            self.leaf_n_samples_[leaf] = len(y_leaf)

            unique_y = np.unique(y_leaf)
            if len(unique_y) == 1:
                self.leaf_pure_classes_[leaf] = unique_y[0]
                self.leaf_models_[leaf] = None
                self.leaf_selectors_[leaf] = None
                continue

            X_leaf_sel = X_leaf
            if self.feature_selector is not None:
                selector = clone(self.feature_selector)
                try:
                    X_leaf_sel = selector.fit_transform(X_leaf, y_leaf)
                except TypeError:
                    X_leaf_sel = selector.fit_transform(X_leaf)
                self.leaf_selectors_[leaf] = selector
            else:
                self.leaf_selectors_[leaf] = None

            model = clone(self.leaf_model_)
            model.fit(X_leaf_sel, y_leaf)
            self.leaf_models_[leaf] = model

        self.leaf_rules_ = self._extract_tree_rules()
        return self

    def _extract_tree_rules(self):
        tree_ = self.tree_.tree_
        rules = {}
        def recurse(node, path):
            if tree_.children_left[node] == _tree.TREE_LEAF:
                rules[node] = path.copy()
            else:
                feat_name = self.feature_names_in_[tree_.feature[node]]
                threshold = tree_.threshold[node]
                path.append((feat_name, "<=", threshold))
                recurse(tree_.children_left[node], path)
                path.pop()
                path.append((feat_name, ">", threshold))
                recurse(tree_.children_right[node], path)
                path.pop()
        recurse(0, [])
        return rules

    def predict(self, X, return_details=False):
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite=True, dtype='numeric')
        leaf_indices = self.tree_.apply(X)
        predictions = np.empty(X.shape[0], dtype=self.classes_.dtype)
        details = [] if return_details else None

        for leaf in np.unique(leaf_indices):
            mask = leaf_indices == leaf
            if not np.any(mask):
                continue
            if leaf in self.leaf_pure_classes_:
                predictions[mask] = self.leaf_pure_classes_[leaf]
            else:
                X_leaf = X[mask]
                if self.leaf_selectors_[leaf] is not None:
                    X_leaf = self.leaf_selectors_[leaf].transform(X_leaf)
                predictions[mask] = self.leaf_models_[leaf].predict(X_leaf)

            if return_details:
                rule_str = " AND ".join(f"{f} {o} {v:.4f}" for f, o, v in self.leaf_rules_[leaf]) if self.leaf_rules_[leaf] else "Root"
                details.extend([{"leaf_id": leaf, "rule": rule_str, "is_pure": leaf in self.leaf_pure_classes_}] * mask.sum())

        return (predictions, pd.DataFrame(details)) if return_details else predictions

    def predict_proba(self, X, return_details=False):
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite=True, dtype='numeric')
        leaf_indices = self.tree_.apply(X)
        probas = np.zeros((X.shape[0], len(self.classes_)))
        details = [] if return_details else None

        for leaf in np.unique(leaf_indices):
            mask = leaf_indices == leaf
            if not np.any(mask):
                continue
            if leaf in self.leaf_pure_classes_:
                class_idx = list(self.classes_).index(self.leaf_pure_classes_[leaf])
                probas[mask, class_idx] = 1.0
            else:
                X_leaf = X[mask]
                if self.leaf_selectors_[leaf] is not None:
                    X_leaf = self.leaf_selectors_[leaf].transform(X_leaf)
                model = self.leaf_models_[leaf]
                if hasattr(model, "predict_proba"):
                    probas[mask] = model.predict_proba(X_leaf)
                else:
                    preds = model.predict(X_leaf)
                    for i, c in enumerate(self.classes_):
                        probas[mask, i] = (preds == c).astype(float)

            if return_details:
                rule_str = " AND ".join(f"{f} {o} {v:.4f}" for f, o, v in self.leaf_rules_[leaf]) if self.leaf_rules_[leaf] else "Root"
                details.extend([{"leaf_id": leaf, "rule": rule_str, "is_pure": leaf in self.leaf_pure_classes_}] * mask.sum())

        return (probas, pd.DataFrame(details)) if return_details else probas

    def get_leaf_info(self, as_dataframe=False):
        check_is_fitted(self)
        info = {}
        for leaf_id, rule_path in self.leaf_rules_.items():
            rule_str = " AND ".join(f"{f} {o} {v:.4f}" for f, o, v in rule_path) if rule_path else "Root"
            info[leaf_id] = {
                "leaf_id": leaf_id,
                "rule": rule_str,
                "n_samples": self.leaf_n_samples_[leaf_id],
                "is_pure": leaf_id in self.leaf_pure_classes_,
            }
        return pd.DataFrame(info).T if as_dataframe else info