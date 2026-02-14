import streamlit as st
from predict_churn import load_predictor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd

def create_ecommerce_customer_form():
    """
    Создание формы для ввода данных о клиенте e-commerce.
    Три колонки для лучшей организации информации.
    """
    st.markdown("### 📝 Fill in the information about the customer:")

    # Инициализация состояния для хранения истории предсказаний
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []

    with st.form("ecommerce_customer_form"):
        # Создаем три колонки
        col1, col2, col3 = st.columns(3)

        with col1:
            # 1. Tenure - Срок обслуживания клиента
            tenure = st.number_input(
                "Tenure",
                min_value=0,
                value=10,
                help="Tenure of customer in organization (Months)"
            )

            # 2. CityTier - Уровень города
            city_tier = st.selectbox(
                "CityTier",
                options=[1, 2, 3],
                help="City Tier"
            )

            # 3. Gender - Пол
            gender = st.selectbox(
                "Gender",
                options=['Female', 'Male'],
                help="Gender of customer"
            )

            # 4. MaritalStatus - Семейное положение
            marital_status = st.selectbox(
                "MaritalStatus",
                options=['Single', 'Divorced', 'Married'],
                help="Marital status of customer"
            )

            # 5. WarehouseToHome - Расстояние до склада
            warehouse_to_home = st.number_input(
                "WarehouseToHome",
                min_value=0,
                value=16,
                help="Distance in between warehouse to home of customer (km)"
            )

            # 6. NumberOfDeviceRegistered - Количество устройств
            number_of_device_registered = st.number_input(
                "NumberOfDeviceRegistered",
                min_value=1,
                value=1,
                help="Total number of deceives is registered on particular customer"
            )

        with col2:
            # 7. PreferredLoginDevice - Предпочитаемое устройство для входа
            preferred_login_device = st.selectbox(
                "PreferredLoginDevice",
                options=['Mobile Phone', 'Phone', 'Computer'],
                help="Preferred login device of customer"
            )

            # 8. HourSpendOnApp - Часы в приложении
            hour_spend_on_app = st.number_input(
                "HourSpendOnApp",
                min_value=0.0,
                value=1.0,
                help="Number of hours spend on mobile application or website"
            )

            # 9. NumberOfAddress - Количество адресов
            number_of_address = st.number_input(
                "NumberOfAddress",
                min_value=1,
                value=1,
                help="Total number of added added on particular customer"
            )

            # 10. Complain - Были ли жалобы
            complain = st.selectbox(
                "Complain",
                options=[0, 1],
                help="Any complaint has been raised in last month"
            )

            # 11. PreferredOrderCat - Предпочитаемая категория заказов
            order_categories = [
                'Laptop & Accessory',
                'Mobile',
                'Mobile Phone',
                'Others',
                'Fashion',
                'Grocery'
            ]
            prefered_order_cat = st.selectbox(
                "PreferredOrderCat",
                options=order_categories,
                help="Preferred order category of customer in last month"
            )

            # 12. PreferredPaymentMode - Предпочитаемый способ оплаты
            payment_modes = [
                'Debit Card', 'UPI', 'CC', 'Cash on Delivery',
                'E wallet', 'COD', 'Credit Card'
            ]
            preferred_payment_mode = st.selectbox(
                "PreferredPaymentMode",
                options=payment_modes,
                help="Preferred payment method of customer"
            )

        with col3:
            # 13. SatisfactionScore - Уровень удовлетворенности
            satisfaction_score = st.number_input(
                "SatisfactionScore",
                min_value=1,
                max_value=5,
                value=3,
                help="Satisfactory score of customer on service"
            )

            # 14. CashbackAmount - Сумма кэшбэка
            cashback_amount = st.number_input(
                "CashbackAmount",
                min_value=0.0,
                value=177.0,
                help="Average cashback in last month ($)"
            )

            # 15. OrderAmountHikeFromlastYear - Рост заказов
            order_amount_hike = st.number_input(
                "OrderAmountHikeFromlastYear",
                min_value=0,
                max_value=40,
                value=16,
                help="Percentage increases in order from last year"
            )

            # 16. CouponUsed - Использованные купоны
            coupon_used = st.number_input(
                "CouponUsed",
                min_value=0,
                value=0,
                help="Купонов использовано за месяц"
            )

            # 17. OrderCount - Количество заказов
            order_count = st.number_input(
                "OrderCount",
                min_value=0,
                value=1,
                help="Total number of orders has been places in last month"
            )

            # 18. DaySinceLastOrder - Дней с последнего заказа
            day_since_last_order = st.number_input(
                "DaySinceLastOrder",
                min_value=0,
                value=5,
                help="Day Since last order by customer"
            )

        # Разделитель перед кнопкой
        st.divider()

        # Кнопка отправки формы
        submitted = st.form_submit_button(
            "🔮 Get a prediction of customer churn",
            use_container_width=True
        )

        # Сбор данных формы в словарь при отправке
        if submitted:
            form_data = {
                'Tenure': tenure,
                'PreferredLoginDevice': preferred_login_device,
                'CityTier': city_tier,
                'WarehouseToHome': warehouse_to_home,
                'PreferredPaymentMode': preferred_payment_mode,
                'Gender': gender,
                'HourSpendOnApp': hour_spend_on_app,
                'NumberOfDeviceRegistered': number_of_device_registered,
                'PreferedOrderCat': prefered_order_cat,
                'SatisfactionScore': satisfaction_score,
                'MaritalStatus': marital_status,
                'NumberOfAddress': number_of_address,
                'Complain': complain,
                'OrderAmountHikeFromlastYear': order_amount_hike,
                'CouponUsed': coupon_used,
                'OrderCount': order_count,
                'DaySinceLastOrder': day_since_last_order,
                'CashbackAmount': cashback_amount
            }

            # Делаем предсказание
            predictor = load_predictor()
            prediction_result = predictor.predict_with_details(form_data)
            prediction_explain = predictor.local_explain_lime(form_data)

            # Добавляем новое предсказание в историю
            st.session_state.predictions_history.append({
                'data': form_data,
                'result': prediction_result,
                'explain': prediction_explain,
                'timestamp': len(st.session_state.predictions_history) + 1
            })

    return submitted


def display_prediction(prediction_entry, index):
    """
    Отображение одного предсказания в expander
    """
    data = prediction_entry['data']
    result = prediction_entry['result']
    explain = prediction_entry['explain']

    # Определяем заголовок expander на основе результата
    churn_status = "🔴 HIGH CHURN RISK" if result['churn_prediction'] else "🟢 LOW CHURN RISK"
    expander_title = f"Prediction #{index} - {churn_status} - {result['churn_probability']:.1%}"

    with st.expander(expander_title, expanded=(index == len(st.session_state.predictions_history))):
        # Создаем две колонки для отображения информации

        st.subheader("👤 Customer Information")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write(f"• Gender: {data['Gender']}")
            st.write(f"• Marital status: {data['MaritalStatus']}")
            st.write(f"• City tier: {data['CityTier']}")
            st.write(f"• Tenure: {data['Tenure']} months")
            st.write(f"• Satisfaction score: {data['SatisfactionScore']}/5")
            st.write(f"• Hour spend on app: {data['HourSpendOnApp']}/day")

        with col2:
            st.write("**📍 Location & Device**")
            st.write(f"• Warehouse to home: {data['WarehouseToHome']} km")
            st.write(f"• Devices registered: {data['NumberOfDeviceRegistered']}")
            st.write(f"• Login device: {data['PreferredLoginDevice']}")
            st.write(f"• Payment mode: {data['PreferredPaymentMode']}")

        with col3:
            st.write("**🛍️ Order History**")
            st.write(f"• Order category: {data['PreferedOrderCat']}")
            st.write(f"• Order count: {data['OrderCount']}")
            st.write(f"• Days since last order: {data['DaySinceLastOrder']}")
            st.write(f"• Order amount hike: {data['OrderAmountHikeFromlastYear']}%")

        with col4:
            st.write("**💳 Financial & Support**")
            st.write(f"• Cashback amount: ${data['CashbackAmount']}")
            st.write(f"• Coupon used: {data['CouponUsed']}")
            st.write(f"• Complain: {'Yes' if data['Complain'] == 1 else 'No'}")
            st.write(f"• Number of address: {data['NumberOfAddress']}")

        if result['churn_prediction']:
            st.error(f"**Status:** {result['interpretation']}")
        else:
            st.success(f"**Status:** {result['interpretation']}")
        st.subheader(f"Churn Probability: {result['churn_probability']:.1%}")

        # Детальная информация в дополнительном expander
        with st.expander("🔍 View customer churn factors analysis"):
            # Convert to DataFrame
            df_importance = pd.DataFrame(explain, columns=['feature', 'impact'])
            df_importance['abs_impact'] = abs(df_importance['impact'])
            df_importance = df_importance.sort_values('abs_impact', ascending=True)
            df_importance['color'] = df_importance['impact'].apply(
                lambda x: '#FF4B4B' if x > 0 else '#00CC96'
            )

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('📊 Feature Impact on Decision', '🥧 Relative Feature Importance'),
                specs=[[{'type': 'bar'}, {'type': 'pie'}]],
                horizontal_spacing=0.15
            )

            # Column 1: Horizontal bar chart
            fig.add_trace(
                go.Bar(
                    y=df_importance['feature'],
                    x=df_importance['impact'],
                    orientation='h',
                    marker_color=df_importance['color'],
                    hovertemplate='<b>%{y}</b><br>' +
                                  'Impact: %{x:.4f}<br>' +
                                  '<extra></extra>',
                    name=''
                ),
                row=1, col=1
            )

            # Add vertical line at 0 on first chart
            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray", row=1, col=1)

            # Column 2: Pie chart of importance (absolute values)
            fig.add_trace(
                go.Pie(
                    labels=df_importance['feature'],
                    values=df_importance['abs_impact'],
                    marker=dict(colors=px.colors.qualitative.Set3),
                    textinfo='percent+label',
                    textposition='inside',
                    insidetextorientation='radial',
                    hovertemplate='<b>%{label}</b><br>' +
                                  'Absolute Impact: %{value:.4f}<br>' +
                                  'Relative Importance: %{percent}<br>' +
                                  '<extra></extra>',
                    name=''
                ),
                row=1, col=2
            )

            # Layout configuration
            fig.update_layout(
                height=500,
                width=1200,
                showlegend=False,
                hovermode='y unified',
                margin=dict(t=80, b=50, l=50, r=50)
            )

            # Axes configuration for first chart
            fig.update_xaxes(
                title_text="Impact on Churn Probability",
                gridcolor='lightgray',
                row=1, col=1
            )

            fig.update_yaxes(
                title_text="",
                gridcolor='lightgray',
                row=1, col=1
            )

            # Adjust subplot titles position and font size
            for annotation in fig['layout']['annotations']:
                if annotation['text'] in ['📊 Feature Impact on Decision', '🥧 Relative Feature Importance']:
                    annotation['y'] = 1.05  # Move closer to plots
                    annotation['font']['size'] = 16  # Slightly larger

            st.plotly_chart(fig, use_container_width=True)



def display_predictions_history():
    """
    Отображение всей истории предсказаний
    """
    if st.session_state.predictions_history:
        st.markdown("---")
        st.markdown("## 📜 Prediction History")
        st.markdown("*Each new prediction appears as a new expander above*")

        # Отображаем предсказания в обратном порядке (новые сверху)
        for i, prediction_entry in enumerate(reversed(st.session_state.predictions_history)):
            index = len(st.session_state.predictions_history) - i
            display_prediction(prediction_entry, index)


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
            ❓ Predictor Churn - E-commerce
        </div>
    """, unsafe_allow_html=True)

    # Отображаем форму
    form_submitted = create_ecommerce_customer_form()

    # Отображаем историю предсказаний
    display_predictions_history()

    # Кнопка для очистки истории (опционально)
    if st.session_state.predictions_history:
        st.markdown("---")
        if st.button("🗑️ Clear all predictions history"):
            st.session_state.predictions_history = []
            st.rerun()


if __name__ == "__main__":
    main()