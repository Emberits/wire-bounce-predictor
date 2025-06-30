import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# 1. Физическая модель подскока провода
# ===========================

def compute_wire_bounce(mass_before, mass_after, gravity=9.81, initial_velocity=0.5):
    """Рассчитывает максимальную высоту подскока после сброса массы"""
    delta_mass = mass_before - mass_after
    acceleration = delta_mass * gravity / mass_before
    max_height = (initial_velocity ** 2) / (2 * gravity)
    return max_height


def simulate_wire_oscillations(mass_after_kg, spring_constant=10000, damping=50, duration=5):
    """Симулирует колебания провода после сброса льда"""

    def dynamics(t, y):
        x, v = y
        dxdt = v
        dvdt = -(damping / mass_after_kg) * v - (spring_constant / mass_after_kg) * x
        return [dxdt, dvdt]

    y0 = [0.5, 0]  # начальное отклонение вверх
    t_span = [0, duration]
    t_eval = np.linspace(0, duration, 500)

    sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval)
    return sol.t, sol.y[0]


def plot_oscillations(time, height):
    plt.figure(figsize=(10, 6))
    plt.plot(time, height, label='Высота подскока (м)')
    plt.title("Колебания провода после сброса льда")
    plt.xlabel("Время (с)")
    plt.ylabel("Высота подскока (м)")
    plt.grid(True)
    plt.legend()
    return plt


# ===========================
# 2. Модель машинного обучения
# ===========================

def estimate_ice_thickness(weather_data, k=0.05):
    """Оценка толщины льда на основе метеоистории"""
    ice_thickness = 0
    for hour in weather_data:
        temp = hour['temperature']
        rh = hour['humidity']
        wind_speed = hour['wind_speed']

        if temp < 0 and rh > 70:
            ice_thickness += k * wind_speed * rh * (1 - abs(temp)/10)
    return ice_thickness


def build_ml_model():
    """Создание и обучение модели ML"""
    data = {
        'temperature': [-3, -5, 0, -8, 2, -2, -4, -6, -1, 1],
        'wind_speed': [10, 12, 5, 14, 7, 9, 11, 13, 6, 8],
        'humidity': [85, 90, 60, 95, 70, 80, 88, 92, 75, 82],
        'temp_change_last_6h': [0.5, 1.2, -0.3, 2.0, 0.8, 0.2, 1.0, 1.5, 0.1, 0.6],
        'precipitation': [0.2, 0.5, 0, 1.0, 0.1, 0.3, 0.8, 1.2, 0.05, 0.2],
        'wire_diameter': [12.7, 15.2, 12.7, 15.2, 12.7, 15.2, 12.7, 15.2, 12.7, 15.2],
        'span_length': [200, 300, 250, 300, 200, 250, 300, 200, 250, 300],
        'month': [12, 1, 2, 1, 2, 12, 1, 2, 12, 1],
        'failure': [1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
    }

    df = pd.DataFrame(data)

    # Расчёт estimated_ice_thickness
    for i in range(len(df)):
        hours_before = np.random.randint(6, 24)
        hourly_data = [
            {'temperature': df.loc[i, 'temperature'] + np.random.uniform(-1, 1),
             'humidity': df.loc[i, 'humidity'],
             'wind_speed': df.loc[i, 'wind_speed']} for _ in range(hours_before)]
        ice_thickness = estimate_ice_thickness(hourly_data)
        df.loc[i, 'estimated_ice_thickness'] = ice_thickness

    X = df.drop('failure', axis=1)
    y = df['failure']

    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)

    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_res, y_res)

    return model, df.columns.tolist()


# ===========================
# 3. Гибридная модель прогнозирования
# ===========================

def predict_risk_and_bounce(input_data, ml_model, feature_columns):
    """
    input_data — словарь с входными данными:
        temperature, wind_speed, humidity, temp_change_last_6h,
        precipitation, wire_diameter, span_length, month
    """

    # Добавляем estimated_ice_thickness
    hourly_weather = [{'temperature': input_data['temperature'],
                       'humidity': input_data['humidity'],
                       'wind_speed': input_data['wind_speed']} for _ in range(12)]  # 12 часов намерзания
    input_data['estimated_ice_thickness'] = estimate_ice_thickness(hourly_weather)

    # Подготовка к предсказанию
    input_df = pd.DataFrame([input_data])
    missing_cols = set(feature_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[feature_columns]

    # Предсказание вероятности сброса
    prob_failure = ml_model.predict_proba(input_df)[0][1]

    if prob_failure < 0.5:
        return {
            "risk": "Низкий",
            "message": "Сброс льда маловероятен"
        }

    # Расчёт массы провода до и после сброса
    diameter_mm = input_data['wire_diameter']
    span_m = input_data['span_length']
    ice_thickness_mm = input_data['estimated_ice_thickness']

    radius_wire_m = diameter_mm / 2000  # мм -> метры
    area_wire = np.pi * radius_wire_m ** 2
    volume_wire = area_wire * span_m
    mass_wire_kg = volume_wire * 2700  # плотность алюминия

    radius_total_m = radius_wire_m + ice_thickness_mm / 1000
    volume_ice = (np.pi * radius_total_m ** 2 - area_wire) * span_m
    mass_ice_kg = volume_ice * 917  # плотность льда

    mass_before_kg = mass_wire_kg + mass_ice_kg
    mass_after_kg = mass_wire_kg

    bounce_height = compute_wire_bounce(mass_before_kg, mass_after_kg)
    time, height = simulate_wire_oscillations(mass_after_kg)

    risk_level = "Высокий" if bounce_height > 1.0 else "Средний" if bounce_height > 0.5 else "Низкий"

    return {
        "risk": risk_level,
        "probability_of_ice_shedding": round(prob_failure, 2),
        "estimated_ice_thickness_mm": round(ice_thickness_mm, 2),
        "bounce_height_m": round(bounce_height, 2),
        "max_oscillation_m": round(max(abs(height)), 2),
        "message": f"Вероятный подскок: {bounce_height:.2f} м",
        "plot_time": time,
        "plot_height": height
    }


# ===========================
# 4. Streamlit-приложение
# ===========================

def main():
    st.title("Прогнозирование подскока провода при сбросе льда")

    st.sidebar.header("Входные данные")
    temperature = st.sidebar.number_input("Температура воздуха (°C)", value=-4.0, step=0.1)
    wind_speed = st.sidebar.number_input("Скорость ветра (м/с)", value=12.0, step=0.1)
    humidity = st.sidebar.number_input("Относительная влажность (%)", value=90.0, step=0.1)
    temp_change_last_6h = st.sidebar.number_input("Изменение температуры за последние 6 часов (°C)", value=1.5, step=0.1)
    precipitation = st.sidebar.number_input("Количество осадков за последний час (мм)", value=1.0, step=0.1)
    wire_diameter = st.sidebar.number_input("Диаметр провода (мм)", value=15.2, step=0.1)
    span_length = st.sidebar.number_input("Длина пролёта (м)", value=300, step=1)
    month = st.sidebar.number_input("Месяц года (1-12)", value=1, min_value=1, max_value=12)

    # Обучаем модель
    ml_model, feature_columns = build_ml_model()

    # Входные данные для тестового случая
    test_case = {
        'temperature': temperature,
        'wind_speed': wind_speed,
        'humidity': humidity,
        'temp_change_last_6h': temp_change_last_6h,
        'precipitation': precipitation,
        'wire_diameter': wire_diameter,
        'span_length': span_length,
        'month': month
    }

    # Прогнозируем риск
    result = predict_risk_and_bounce(test_case, ml_model, feature_columns)

    # Выводим результат
    st.subheader("Результат прогнозирования:")
    st.write(f"**Риск:** {result['risk']}")
    st.write(f"**Вероятность сброса льда:** {result['probability_of_ice_shedding'] * 100:.2f}%")
    st.write(f"**Оценённая толщина льда:** {result['estimated_ice_thickness_mm']:.2f} мм")
    st.write(f"**Максимальная высота подскока:** {result['bounce_height_m']:.2f} м")
    st.write(f"**Максимальная амплитуда колебаний:** {result['max_oscillation_m']:.2f} м")
    st.write(f"**Сообщение:** {result['message']}")

    # Визуализация колебаний
    fig = plot_oscillations(result['plot_time'], result['plot_height'])
    st.pyplot(fig)


if __name__ == "__main__":
    main()
