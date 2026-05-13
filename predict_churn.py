import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

class ValueReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, replacements: dict):
        self.replacements = replacements

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.replacements.items():
            if col in X.columns:
                X[col] = X[col].replace(mapping)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features

import __main__; __main__.ValueReplacer = ValueReplacer

@st.cache_resource
def load_predictor():
    """Загрузка предиктора с кэшированием"""
    return ChurnPredictor()

class ChurnPredictor:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.model_path = self.base_dir / 'model' / 'xgb_model.joblib'
        self.pipeline_path = self.base_dir / 'model' / 'prep_ordinal_noscale.joblib'

        self._check_files_exist()

        # Загрузка модели и единого пайплайна
        self.model = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)

        # SHAP-объяснитель (оптимизирован для XGBoost/деревьев)
        self.shap_explainer = shap.TreeExplainer(self.model)

    def _check_files_exist(self):
        files = {
            self.model_path: "Модель XGBoost (xgb_model.joblib)",
            self.pipeline_path: "Пайплайн (prep_ordinal_noscale.joblib)"
        }
        missing = [f"{desc}: {path}" for path, desc in files.items() if not path.exists()]
        if missing:
            st.error("❌ Отсутствуют файлы:\n" + "\n".join(missing))
            st.info(f"Проверьте папку: {self.base_dir / 'model'}")
            raise FileNotFoundError(missing)

    def preprocess_data(self, input_data):
        """Применяет сохранённый пайплайн к новым данным"""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Пайплайн сам делает все замены, импутацию, кодирование
        transformed = self.pipeline.transform(df)
        return np.asarray(transformed)

    def _get_feature_names(self):
        """Безопасно получает названия признаков после пайплайна"""
        if hasattr(self.pipeline, 'get_feature_names_out'):
            return list(self.pipeline.get_feature_names_out())
        n = getattr(self.model, 'n_features_in_', len(self.model.feature_importances_))
        return [f"feature_{i}" for i in range(n)]

    def predict(self, input_data):
        X = self.preprocess_data(input_data)
        pred = self.model.predict(X)
        return pred[0] if len(pred) == 1 else pred

    def predict_proba(self, input_data):
        X = self.preprocess_data(input_data)
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def predict_with_details(self, input_data):
        pred = self.predict(input_data)
        proba = self.predict_proba(input_data)
        pred_val = pred[0] if isinstance(pred, np.ndarray) else pred
        prob_val = proba[0] if isinstance(proba, np.ndarray) else proba
        return {
            'churn_prediction': int(pred_val),
            'churn_probability': float(prob_val),
            'interpretation': '⚠️ The customer is prone to churn' if pred_val == 1 else '✅ The customer is loyal'
        }

    def get_feature_importance(self, top_n=None, return_dataframe=True):
        importances = self.model.feature_importances_
        feature_names = self._get_feature_names()
        df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        df_imp = df_imp.sort_values('importance', ascending=False).reset_index(drop=True)
        df_imp['importance_percent'] = (df_imp['importance'] / df_imp['importance'].sum() * 100).round(2)
        df_imp['cumulative_percent'] = df_imp['importance_percent'].cumsum()
        if top_n:
            df_imp = df_imp.head(top_n)
        return df_imp if return_dataframe else df_imp.to_dict('records')

    def local_explain_shap(self, input_data):
        """Локальная интерпретация через SHAP"""
        X = self.preprocess_data(input_data)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        shap_values = self.shap_explainer.shap_values(X)
        # Для бинарной классификации TreeExplainer может вернуть список [класс_0, класс_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # берём значения для класса Churn (1)
            
        feature_names = self._get_feature_names()
        contributions = list(zip(feature_names, shap_values[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions[:10]