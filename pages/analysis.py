import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from predict_churn import load_predictor
from plotly.subplots import make_subplots


def show_feature_importance():
    """Отображение важности признаков в Streamlit"""

    st.markdown("""
        <h2 style='text-align: center;'>
            Random Forest Classifier
        </h2>
        """, unsafe_allow_html=True)

    # Центрированная подпись снизу
    st.markdown("""
        <p style='text-align: center; font-size: 14px;'>
            Accuracy: 98%
        </p>
        """, unsafe_allow_html=True)

    # Загружаем модель
    predictor = load_predictor()

    try:
        # Получаем важность признаков
        importance_df = predictor.get_feature_importance()

        # Создаем две колонки
        col1, col2 = st.columns([1, 1])

        with col1:

            # Топ-10 признаков
            top_10 = importance_df.head(10).copy()
            top_10 = top_10.sort_values('importance_percent', ascending=True)

            fig2 = go.Figure()

            # Добавляем столбцы с оттенками красного
            fig2.add_trace(go.Bar(
                x=top_10['importance_percent'],
                y=top_10['feature'],
                orientation='h',
                marker=dict(
                    color=top_10['importance_percent'],
                    colorscale=[[0, '#ffe5e5'], [1, '#cc0000']],  # Встроенная красная цветовая шкала
                    showscale=False,
                    line=dict(width=1)
                ),
                text=top_10['importance_percent'].round(1),
                texttemplate='%{text}%',
                textposition='auto',
                textfont=dict(size=12),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.1f}%<br>Cumulative %: %{customdata:.1f}%<extra></extra>',
                customdata=top_10['cumulative_percent'],
                width=0.7
            ))

            # Настройка макета
            fig2.update_layout(
                title=dict(
                    text='🏆 Top 10 most important characteristics',
                    font=dict(size=20, family='Arial Black'),
                    x=0.5,
                    xanchor='center'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Importance (%)',
                    title_font=dict(size=14),
                    tickfont=dict(size=12),
                    showgrid=True,
                    zeroline=False,
                    zerolinewidth=1
                ),
                yaxis=dict(
                    title='',
                    tickfont=dict(size=12),
                    showgrid=False,
                    autorange='reversed'
                ),
                margin=dict(l=10, r=10, t=60, b=40),
                height=550,
                shapes=[
                    dict(
                        type='line',
                        x0=top_10['importance_percent'].mean(),
                        y0=-0.5,
                        x1=top_10['importance_percent'].mean(),
                        y1=len(top_10) - 0.5,
                        line=dict(
                            width=2,
                            dash='dash'
                        )
                    )
                ],
                annotations=[
                    dict(
                        x=top_10['importance_percent'].mean(),
                        y=len(top_10) - 1,
                        text=f'{top_10["importance_percent"].mean():.1f}%',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        font=dict(size=11),
                        borderwidth=1,
                        borderpad=4,
                        ax=20,
                        ay=-30
                    )
                ]
            )

            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.subheader("Detailed information")

            # Выбор количества признаков для отображения
            n_features = st.slider(
                "The number of features to display:",
                min_value=5,
                max_value=min(50, len(importance_df)),
                value=15
            )

            # Отображаем таблицу с важностью
            display_df = importance_df.head(n_features).copy()
            display_df['importance'] = display_df['importance'].round(4)
            display_df.index = range(1, len(display_df) + 1)

            st.dataframe(
                display_df[['feature', 'importance', 'importance_percent', 'cumulative_percent']],
                column_config={
                    "feature": "Feature",
                    "importance": "Importance",
                    "importance_percent": "Importance (%)",
                    "cumulative_percent": "Cumulative %"
                },
                use_container_width=True,
                height = 410
            )

    except Exception as e:
        st.error(f"Error when getting the importance of features: {str(e)}")


def display_dataset_info(df):
    """
    Функция для отображения информации о датафрейме в Streamlit
    с фокусом на бизнес-метрики и анализ оттока клиентов
    """

    # Основные метрики датасета
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total customers",
            f"{len(df):,}",
            help="Total number of customers in the dataset"
        )

    with col2:
        n_churned = df['Churn'].sum()
        churn_rate = (n_churned / len(df)) * 100
        st.metric(
            "Customer churn",
            f"{n_churned:,}",
            delta=f"{churn_rate:.1f}%",
            delta_color="inverse",
            help="The number of customers who have left"
        )

    with col3:
        n_retained = len(df) - n_churned
        retention_rate = (n_retained / len(df)) * 100
        st.metric(
            "Retained customers",
            f"{n_retained:,}",
            delta=f"{retention_rate:.1f}%",
            delta_color="normal",
            help="The number of customers who remained"
        )

    with col4:
        missing_values = df.isnull().sum().sum()
        missing_percent = (missing_values / (df.shape[0] * df.shape[1])) * 100
        st.metric(
            "Missing values",
            f"{missing_values:,}",
            delta=f"{missing_percent:.1f}%",
            delta_color="inverse",
            help="Total number of missing values"
        )

    with col5:
        st.metric(
            "Number of signs",
            df.shape[1],
            help="Number of columns in the dataset"
        )

    st.markdown("""
        <style>
            div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
                display: flex;
                justify-content: space-between;  /* Равномерное распределение */
                width: 100%;
            }

            div[data-testid="stTabs"] div[data-baseweb="tab-list"] button {
                flex: 1;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    # Создаем вкладки для разных аспектов анализа
    tab1, tab2, tab3 = st.tabs([
        "📋 Data preview",
        "🔄 Customer churn analysis",
        "📈 Customer characteristics"
    ])

    with tab1:
        st.subheader("The first rows of the dataset")

        # Настройки отображения
        col1, col2 = st.columns([1, 3])
        with col1:
            n_rows = st.selectbox(
                "Number of rows:",
                options=[5, 10, 15, 20, 50],
                index=1
            )

        with col2:
            search_col = st.text_input(
                "🔍 Column search (enter the name):",
                placeholder="for example: CustomerID, Churn, Tenure..."
            )

        # Фильтрация колонок по поиску
        if search_col:
            available_cols = [col for col in df.columns if search_col.lower() in col.lower()]
            if available_cols:
                display_cols = available_cols
            else:
                display_cols = df.columns
                st.info(f"Columns with '{search_col}' were not found. We show all the columns.")
        else:
            display_cols = df.columns

        # Отображаем датафрейм
        st.dataframe(
            df[display_cols].head(n_rows),
            use_container_width=True,
            height=300
        )

        # Статистика по колонкам
        with st.expander("📊 Column statistics"):
            col_stats = pd.DataFrame({
                'Data type': df.dtypes,
                'Non-empty values': df.count(),
                'Unique values': df.nunique(),
                'Skipped %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_stats, use_container_width=True)

    with tab2:
        st.subheader("🔄 Customer churn analysis")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Круговая диаграмма распределения оттока
            churn_counts = df['Churn'].value_counts().reset_index()
            churn_counts.columns = ['Status', 'Count']
            churn_counts['Status'] = churn_counts['Status'].map({0: '✅ Stayed', 1: '⚠️ Churned'})

            colors = ['#00CC96', '#FF4B4B']  # Зеленый для остались, красный для ушли

            fig_pie = go.Figure(data=[go.Pie(
                labels=churn_counts['Status'],
                values=churn_counts['Count'],
                hole=0.4,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=14),
                hovertemplate='<b>%{label}</b><br>Quantity: %{value:,}<br>Percent: %{percent}<extra></extra>'
            )])

            fig_pie.update_layout(
                showlegend=False,
                height=450,
                annotations=[
                    dict(
                        text=f'Total:<br>{len(df):,}',
                        x=0.5,
                        y=0.5,
                        font=dict(size=14, color='gray'),
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Детальная статистика по оттоку
            st.subheader("Detailing customer churn")

            # Создаем метрики
            metrics_df = pd.DataFrame({
                'Indicator': [
                    'Customers who have left',
                    'The customers who stayed',
                    'Churn rate',
                    'Retention rate'
                ],
                'Values': [
                    f"{n_churned:,}",
                    f"{n_retained:,}",
                    f"{churn_rate:.2f}%",
                    f"{retention_rate:.2f}%"
                ]
            })

            st.table(metrics_df)

            # Бизнес-интерпретация
            st.info("""
            **💡  Business interpretation:**
            Churn Rate shows the proportion of customers who
            stopped using the services. The lower this indicator,
            the more effective the customer retention work.
            """)

            # Рекомендации в зависимости от уровня оттока
            if churn_rate < 10:
                st.success("✅ **Low churn rate** is an excellent indicator!")
            elif churn_rate < 20:
                st.warning("⚠️ **Average churn rate** - monitoring required")
            else:
                st.error("🚨 **High level of customer churn - urgent measures are needed")

    with tab3:
        st.subheader("📈 Customer characteristics")

        # Выбор категориального признака для анализа
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['CustomerID']]

        col1, col2 = st.columns([1, 2])

        with col1:
            selected_feature = st.selectbox(
                "Select a feature for analysis:",
                categorical_cols,
                format_func=lambda x: {
                    'PreferredLoginDevice': 'Preferred login device',
                    'PreferredPaymentMode': 'Preferred payment mode',
                    'Gender': 'Gender',
                    'PreferedOrderCat': 'Preferred order category',
                    'MaritalStatus': 'Marital status'
                }.get(x, x)
            )

            # Настройка отображения
            show_numbers = st.checkbox("Show values", value=True)

        with col2:
            # Анализ выбранного признака в разрезе оттока
            churn_by_feature = pd.crosstab(
                df[selected_feature],
                df['Churn'],
                margins=True,
                margins_name='Всего'
            )
            churn_by_feature.columns = ['Stayed', 'Churned', 'Total']
            churn_by_feature['Churn %'] = (churn_by_feature['Churned'] / churn_by_feature['Total'] * 100).round(1)

            # Убираем строку "Всего" для отображения
            display_df = churn_by_feature.iloc[:-1].copy()

            st.dataframe(
                display_df,
                column_config={
                    "Stayed": st.column_config.NumberColumn(format="%d"),
                    "Churned": st.column_config.NumberColumn(format="%d"),
                    "Total": st.column_config.NumberColumn(format="%d"),
                    "Churn %": st.column_config.NumberColumn(format="%.1f%%")
                },
                use_container_width=True
            )

        # Визуализация
        st.subheader(f"Distribution by feature: {selected_feature}")

        fig_bar = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Customer distribution', 'Churn %'],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # График распределения
        counts = df[selected_feature].value_counts()
        fig_bar.add_trace(
            go.Bar(
                x=counts.index,
                y=counts.values,
                name='Quantity',
                marker_color='#636EFA',
                text=counts.values if show_numbers else None,
                textposition='auto',
            ),
            row=1, col=1
        )

        # График процента оттока
        churn_pct = df.groupby(selected_feature)['Churn'].mean() * 100
        fig_bar.add_trace(
            go.Bar(
                x=churn_pct.index,
                y=churn_pct.values,
                name='Churn %',
                marker_color='#FF4B4B',
                text=[f'{v:.1f}%' for v in churn_pct.values] if show_numbers else None,
                textposition='auto',
            ),
            row=1, col=2
        )

        fig_bar.update_layout(
            height=400,
            showlegend=False,
            hovermode='x unified'
        )

        fig_bar.update_xaxes(tickangle=45)

        st.plotly_chart(fig_bar, use_container_width=True)

        # Числовые характеристики
        st.subheader("📊 Statistics based on numerical characteristics")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['CustomerID', 'Churn']]

        selected_numeric = st.multiselect(
            "Select numerical features for analysis:",
            numeric_cols,
            default=numeric_cols[:3],
            format_func=lambda x: {
                'Tenure': 'Tenure',
                'WarehouseToHome': 'Warehouse to home',
                'HourSpendOnApp': 'Hour spend on app',
                'NumberOfDeviceRegistered': 'Number of device registered',
                'SatisfactionScore': 'Satisfaction score',
                'NumberOfAddress': 'Number of address',
                'Complain': 'Complain',
                'OrderAmountHikeFromlastYear': 'Order amount hike from last year',
                'CouponUsed': 'Coupon used',
                'OrderCount': 'Order count',
                'DaySinceLastOrder': 'Day since last order',
                'CashbackAmount': 'Cashback amount'
            }.get(x, x)
        )

        if selected_numeric:
            stats_df = df[selected_numeric].describe().T
            stats_df.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
            stats_df = stats_df.round(2)

            st.dataframe(stats_df, use_container_width=True)


def main():
    """
    Главная функция для работы приложения.
    """
    st.set_page_config(
        page_title="E-commerce Churn Predictor",
        page_icon="🛒",
        layout="wide"
    )

    st.markdown("""
        <style>
        .centered-title {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        </style>

        <div class="centered-title">
            📊 Churn Analysis - E-commerce
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    df = pd.read_excel('data\E Commerce Dataset.xlsx', sheet_name='E Comm', usecols='A:T')

    # Отображение информации о датасете
    display_dataset_info(df)

    st.markdown("---")

    show_feature_importance()


if __name__ == "__main__":
    main()