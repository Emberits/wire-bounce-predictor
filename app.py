import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns

# Физическая модель подскока провода
def compute_wire_bounce(mass_before, mass_after, gravity=9.81, initial_velocity=0.5):
    delta_mass = mass_before - mass_after
    acceleration = delta_mass * gravity / mass_before
    max_height = (initial_velocity ** 2) / (2 * gravity)
    return max_height

def simulate_wire_oscillations(mass_after_kg, spring_constant=10000, damping=50, duration=5):
    def dynamics(t, y):
        x, v = y
        dxdt = v
        dvdt = -(damping / mass_after_kg) * v - (spring_constant / mass_after_kg) * x
        return [dxdt, dvdt]

    y0 = [0.5, 0]
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

# Модель машинного обучения
def estimate_ice_thickness(weather_data, k=0.05):
    ice_thickness = 0
    for hour in weather_data:
        temp = hour['temperature']
        rh = hour['humidity']
        wind_speed = hour['wind_speed']

        if temp < 0 and rh > 70:
            ice_thickness += k * wind_speed * rh * (1 - abs(temp)/10)
    return ice_thickness

def build_ml_model():
    data = {
        'temperature': [-3, -5, 0, -8, 2, -2, -4, -6, -1, 1],
        'wind_speed': [10, 12, 5, 14, 7, 9, 11, 13, 6, 8],
        'humidity': [85, 90, 60, 95, 70, 80, 88, 92, 75, 82],
        'temp_change_last_6h': [0.5, 1.2, -0.3, 2.0, 0.8, 0.2, 1.0, 1.5, 0.1, 0.6],
        'precipitation': [0.2, 0.5, 0, 1.0, 0.1, 0.3, 0.8, 1.2, 0.05, 0.2],
        'wire_diameter': [12.7, 15.2, 12.7, 15.2, 12.7, 15.2, 12.7, 15.2, 12.7, 15.2],
        'span_length': [200, 300, 250, 300, 200, 250, 300, 200, 250, 300],
        'month': [12, 1, 2, 1, 2, 12, 1, 2, 12, 1],
        'failure': [1, 1, 0, 1, 0, 1, 1, 1, 0, 0]  # 6 единиц и 4 нуля
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

    # Исправление: установка k_neighbors=3 (меньше, чем минимальное количество примеров в классе)
    smote = SMOTE(k_neighbors=3, random_state=42)  # <--- ДОБАВЬТЕ ЭТУ СТРОКУ
    X_res, y_res = smote.fit_resample(X, y)

    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_res, y_res)

    return model, df.columns.tolist()

# Гибридная модель прогнозирования
def predict_risk_and_bounce(input_data, ml_model, feature_columns):
    # ... (остальной код остаётся без изменений) ...
    pass  # Ваш код здесь

# Streamlit-приложение
def main():
    # ... (остальной код остаётся без изменений) ...
    pass  # Ваш код здесь

if __name__ == "__main__":
    main()
