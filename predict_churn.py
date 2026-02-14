import streamlit as st
import pandas as pd
import joblib
import pickle
import json
import numpy as np
import lime
import lime.lime_tabular
from pathlib import Path


# Кэшируем загрузку модели для оптимизации
@st.cache_resource
def load_predictor():
    """Загрузка предиктора с кэшированием"""
    return ChurnPredictor()


class ChurnPredictor:
    def __init__(self):
        # Определяем базовую директорию проекта
        self.base_dir = Path(__file__).parent

        # Определяем пути к файлам
        self.model_path = self.base_dir / 'model' / 'churn_model.pkl'
        self.scaler_path = self.base_dir / 'model' / 'scaler.pkl'
        self.feature_info_path = self.base_dir / 'model' / 'feature_info.pkl'
        self.mappings_path = self.base_dir / 'model' / 'mappings.pkl'
        self.training_data_path = self.base_dir / 'data' / 'training_data_for_lime.npy'
        self.feature_names_path = self.base_dir / 'data' / 'feature_names.json'

        # Проверяем существование файлов перед загрузкой
        self._check_files_exist()

        # Загрузка всех сохраненных компонентов
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.training_data = np.load(self.training_data_path)

        with open(self.feature_info_path, 'rb') as f:
            self.feature_info = pickle.load(f)

        with open(self.mappings_path, 'rb') as f:
            self.mappings = pickle.load(f)

        with open(self.feature_names_path, 'r') as f:
            self.feature_names = json.load(f)

        self.explainer = self._load_and_create_explainer()

    def _check_files_exist(self):
        """Проверяет существование всех необходимых файлов"""
        files_to_check = [
            (self.model_path, "Модель"),
            (self.scaler_path, "Скейлер"),
            (self.feature_info_path, "Информация о признаках"),
            (self.mappings_path, "Маппинги"),
            (self.training_data_path, "Обучающие данные для LIME"),
            (self.feature_names_path, "Названия признаков")
        ]

        missing_files = []
        for file_path, file_desc in files_to_check:
            if not file_path.exists():
                missing_files.append(f"{file_desc}: {file_path}")

        if missing_files:
            error_msg = "❌ Не найдены следующие файлы:\n" + "\n".join(missing_files)
            st.error(error_msg)

            # Показываем текущую директорию для отладки
            st.info(f"Текущая директория: {Path.cwd()}")
            st.info(
                f"Содержимое папки model: {list((self.base_dir / 'model').glob('*')) if (self.base_dir / 'model').exists() else 'папка не найдена'}")
            st.info(
                f"Содержимое папки data: {list((self.base_dir / 'data').glob('*')) if (self.base_dir / 'data').exists() else 'папка не найдена'}")

            raise FileNotFoundError(f"Отсутствуют файлы: {missing_files}")

    def _load_and_create_explainer(self):
        # Создаем explainer заново
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            class_names=['Loyal (0)', 'Churn (1)'],
            mode='classification',
            discretize_continuous=False,
            sample_around_instance=True,
            random_state=42
        )
        return explainer

    # ... остальные методы остаются без изменений ...
    def preprocess_data(self, input_data):
        """
        Предобработка новых данных так же, как при обучении
        input_data: pandas DataFrame или dict
        """
        # Если передан словарь, создаем DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # 1. Применяем маппинги для бинарных признаков
        if 'PreferredLoginDevice' in df.columns:
            df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace(
                {'Mobile Phone': 'Phone', 'Phone': 'Phone', 'Computer': 'Computer'}
            )
            df['PreferredLoginDevice'] = df['PreferredLoginDevice'].map(
                self.mappings['PreferredLoginDevice']
            )

        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map(self.mappings['Gender'])

        # 2. Замены в других категориальных признаках
        if 'PreferredPaymentMode' in df.columns:
            df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(
                {'Credit Card': 'CC'}
            )

        if 'PreferedOrderCat' in df.columns:
            df['PreferedOrderCat'] = df['PreferedOrderCat'].replace(
                {'Mobile Phone': 'Mobile'}
            )

        # 3. One-Hot Encoding
        # Создаем все возможные dummy-колонки
        for col in self.feature_info['one_hot_cols']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                # Добавляем недостающие колонки
                for expected_col in [c for c in self.feature_info['all_features']
                                     if c.startswith(col + '_')]:
                    if expected_col not in dummies.columns:
                        dummies[expected_col] = 0
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        # 4. Масштабирование числовых признаков
        for col in self.feature_info['numeric_cols_to_scale']:
            if col in df.columns:
                # Проверяем, что все значения числовые
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 5. Убеждаемся, что все нужные признаки присутствуют
        for feature in self.feature_info['all_features']:
            if feature not in df.columns:
                df[feature] = 0

        # 6. Масштабируем числовые признаки
        if self.feature_info['numeric_cols_to_scale']:
            df[self.feature_info['numeric_cols_to_scale']] = self.scaler.transform(
                df[self.feature_info['numeric_cols_to_scale']]
            )

        # 7. Упорядочиваем колонки как при обучении
        df = df[self.feature_info['all_features']]

        return df

    def predict(self, input_data):
        """Предсказание оттока"""
        processed_data = self.preprocess_data(input_data)
        prediction = self.model.predict(processed_data)
        return prediction[0] if len(prediction) == 1 else prediction

    def predict_proba(self, input_data):
        """Вероятность оттока"""
        processed_data = self.preprocess_data(input_data)
        probabilities = self.model.predict_proba(processed_data)
        return probabilities[:, 1]  # вероятность класса 1 (Churn)

    def predict_with_details(self, input_data):
        """Предсказание с деталями"""
        prediction = self.predict(input_data)
        probability = self.predict_proba(input_data)

        result = {
            'churn_prediction': int(prediction[0]) if isinstance(prediction, np.ndarray) else int(prediction),
            'churn_probability': float(probability[0]) if isinstance(probability, np.ndarray) else float(probability),
            'interpretation': '⚠️ The customer is prone to churn' if prediction else '✅ The customer is loyal'
        }
        return result

    def get_feature_importance(self, top_n=None, return_dataframe=True):
        """
        Возвращает важность признаков для обученной модели.
        """
        # Проверяем, что модель имеет атрибут feature_importances_ (для деревьев)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        # Проверяем для линейных моделей
        elif hasattr(self.model, 'coef_'):
            importances = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            # Берем абсолютные значения для линейных моделей
            importances = np.abs(importances)
        else:
            raise ValueError("Модель не поддерживает анализ важности признаков. "
                             "Модель должна иметь feature_importances_ или coef_.")

        # Получаем названия признаков
        feature_names = self.feature_info['all_features']

        # Создаем DataFrame с важностью признаков
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Сортируем по убыванию важности
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

        # Добавляем процент от общей важности
        importance_df['importance_percent'] = (
                importance_df['importance'] / importance_df['importance'].sum() * 100).round(2)

        # Добавляем кумулятивный процент
        importance_df['cumulative_percent'] = importance_df['importance_percent'].cumsum()

        # Ограничиваем количество признаков, если указано top_n
        if top_n is not None:
            importance_df = importance_df.head(top_n)

        if return_dataframe:
            return importance_df
        else:
            return importance_df.to_dict('records')

    def local_explain_lime(self, input_data):
        """
        Метод для локальной интерпретации результата.
        """
        processed_data_row = self.preprocess_data(input_data).iloc[0].values.astype(float)
        exp = self.explainer.explain_instance(
            data_row=processed_data_row,
            predict_fn=self.model.predict_proba,
            num_features=8,
            num_samples=500
        )

        return exp.as_list(label=1)
