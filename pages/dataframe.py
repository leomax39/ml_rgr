import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Датасет", layout="wide")

st.title("Описание и предобработка датасета")
st.markdown("---")

@st.cache_data
def load_raw():
    return pd.read_csv("data/moldova_cars_task.csv")
@st.cache_data
def load_processed():
    return pd.read_csv("data/moldova_cars_task_filtered2.csv")

raw_df = load_raw()
processed_df = load_processed()
processed_df.drop(columns=['Unnamed: 0'], inplace=True)

st.header("1. Исходный датасет")

st.markdown("""
Исходный датасет содержит информацию о подержанных автомобилях:
- Марка (Make)
- Модель (Model)
- Год выпуска (Year)
- Тип кузова (Style)
- Пробег (Distance)
- Объём двигателя (Engine_capacity)
- Тип топлива (Fuel_type)
- Тип трансмиссии (Transmission)
- Цена (Price)
""")
st.dataframe(raw_df.head(10), use_container_width=True)

st.markdown("---")

st.header("2. Очистка данных")
st.markdown("""
В процессе предобработки были выполнены следующие шаги:
""")
st.markdown("""
### 1. Удаление аномальных значений
Были установлены допустимые диапазоны:
""")
st.code("""
data.loc[(data['Distance'] < 1000) | (data['Distance'] > 500000), 'Distance'] = np.nan
data.loc[(data['Engine_capacity(cm3)'] < 500) | (data['Engine_capacity(cm3)'] > 6000), 'Engine_capacity(cm3)'] = np.nan
data.loc[(data['Year'] < 1980) | (data['Year'] > 2024), 'Year'] = np.nan
data.loc[(data['Price(euro)'] < 1000) | (data['Price(euro)'] > 1000000), 'Price(euro)'] = np.nan
""")
st.markdown("""
### 2. Заполнение пропусков
Пропущенные значения заполнялись с использованием группировки:
""")
st.code("""
group_make_model = data.groupby('Make')['Model'].transform(lambda x: x.mode().iat[0])
data['Model'] = data['Model'].fillna(group_make_model)
group_year_dist = data.groupby('Year')['Distance'].transform('median')
data['Distance'] = data['Distance'].fillna(group_year_dist)
""")
st.markdown("""
Использовались:
- мода для категориальных признаков
- медиана для числовых признаков
""")
st.markdown("---")

st.header("3. Feature Engineering (Создание новых признаков)")
st.markdown("""
Были добавлены производные признаки:
""")
feature_engineering = pd.DataFrame([
    ("Vehicle_age", "Возраст автомобиля"),
    ("Distance_per_year", "Пробег / Возраст"),
    ("Vehicle_age_sq", "Возраст²"),
    ("Log_Distance", "Логарифм пробега"),
    ("Log_Engine", "Логарифм объёма двигателя"),
    ("Age_x_Distance", "Возраст × Пробег"),
    ("Distance_x_Engine", "Пробег × Объём двигателя"),
    ("Distance_per_engine", "Пробег / Объём двигателя"),
], columns=["Признак", "Описание"])
st.dataframe(feature_engineering, use_container_width=True, hide_index=True)
st.markdown("""
Добавление новых признаков позволило:
- улучшить качество моделей
- выявить нелинейные зависимости
- повысить значение R²
""")
st.markdown("---")

st.header("4. Кодирование категориальных признаков")
st.markdown("""
В датасете присутствуют категориальные признаки:
- **Style** (тип кузова)
- **Fuel_type** (тип топлива)
- **Transmission** (тип коробки передач)
- **Make** (марка автомобиля)
- **Model** (модель автомобиля)
Для корректной работы моделей машинного обучения
категориальные признаки были преобразованы в числовой формат.
""")
st.markdown("### 1. One-Hot Encoding")
st.markdown("""
Для признаков:
- Style  
- Fuel_type  
- Transmission  
использовалось **One-Hot кодирование**.
""")
st.code("""
data_filtered = pd.get_dummies(
    data_filtered,
    columns=['Style', 'Fuel_type', 'Transmission'],
    dtype=int
)
""")
st.markdown("""
В результате каждая категория преобразована в отдельный бинарный столбец.
""")
st.markdown("---")
st.markdown("### 2. Frequency Encoding")
st.markdown("""
Для признаков **Make** и **Model** использовалось
**Frequency Encoding** (кодирование по частоте).
""")
st.code("""
for col in ['Make', 'Model']:
    freq = data_filtered[col].value_counts() / len(data_filtered)
    data_filtered[col] = data_filtered[col].map(freq)
""")
st.markdown("""
Таким образом:
- Каждой марке и модели присваивается
  её относительная частота появления в датасете.
- Более популярные модели получают большее значение.
- Менее популярные — меньшее.
""")
st.markdown("---")
st.header("5. Обработанный датасет")
st.dataframe(processed_df.head(10), use_container_width=True)
st.markdown("""
### Итог:
После очистки данных:
- удалены выбросы
- заполнены пропуски
- добавлены новые признаки
- подготовлены данные для обучения моделей ML
""")
