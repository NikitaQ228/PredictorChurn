import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="E-commerce Churn Predictor",
    page_icon="🛒",
    layout="wide"
)

pages_dir = Path(__file__).parent / "pages"

# Создаем объекты страниц
analysis_page = st.Page(
    str(pages_dir / "analysis.py"),
    title="Analysis",
    icon="📊",
    default=True
)

predictor_page = st.Page(
    str(pages_dir / "predictor.py"),  # функция или модуль
    title="Predictor",
    icon="❓"
)

# Настраиваем навигацию
pg = st.navigation([analysis_page, predictor_page])

# Добавляем информацию в боковое меню
with st.sidebar:

    # Информация об авторе
    st.markdown("### 👨‍💻 Author")
    st.markdown("**Baboshin Nikita**")
    st.markdown("[GitHub](https://github.com/NikitaQ228)")

    st.divider()

    # Ссылка на датасет
    st.markdown("### 🗃️ Dataset")
    st.markdown("[E-commerce Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)")

    st.divider()

    # Дополнительная информация
    st.caption(f"© 2026 | Версия 1.0.0")

pg.run()