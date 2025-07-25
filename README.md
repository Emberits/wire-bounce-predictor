# 🔮 Прогнозирование подскока провода после залпового сброса изморози

Это Streamlit-приложение реализует **гибридную модель прогнозирования**:
- Вероятность залпового сброса льда на проводе (модель машинного обучения)
- Амплитуда подскока провода (физическая модель)
- Оценка риска короткого замыкания или аварии

Используется для научных исследований и анализа устойчивости воздушных линий электропередачи к воздействию гололёда и изморози.

---

## 🧠 Цель проекта

> Предсказание вероятности сброса льда с провода ВЛ и связанного с этим явления подскока провода, которое может привести к аварии. Модель использует доступные климатические и конструктивные параметры вместо недоступных исторических данных.

---

## 📊 Поддерживаемые входные параметры

| Параметр | Описание |
|----------|----------|
| Температура | Температура воздуха (°C) |
| Скорость ветра | м/с |
| Влажность | % |
| Изменение температуры за 6 ч | °C |
| Осадки за час | мм |
| Диаметр провода | мм |
| Длина пролёта | м |
| Месяц года | 1–12 |

---

## 🚀 Как запустить приложение

### 1. Установите зависимости:

```bash
pip install -r requirements.txt