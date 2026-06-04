import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from catboost import CatBoostRegressor

st.set_page_config(
    page_title="Предсказание цены авто",
    layout="wide"
)
st.title("Предсказание цены автомобиля")
st.markdown("""
Данная страница позволяет получить **предсказание стоимости автомобиля** 
с помощью обученных моделей машинного обучения.

Вы можете:
- Ввести данные **вручную** через форму
- Загрузить **CSV файл** с данными об автомобилях
""")
st.markdown("---")
@st.cache_resource
def load_models():
    models = {}
    models["ML1 Ridge"] = pickle.load(open("models/model1_ridge.pkl", "rb"))
    models["ML2 GB"] = pickle.load(open("models/model2_gb.pkl", "rb"))
    models["ML4 RF"] = pickle.load(open("models/model4_rf.pkl", "rb"))
    models["ML5 Stack"] = pickle.load(open("models/model5_stacking.pkl", "rb"))
    cat_model = CatBoostRegressor()
    cat_model.load_model("models/model3_catboost.cbm")
    models["ML3 CatBoost"] = cat_model
    models["ML6 NeuralNet"] = tf.keras.models.load_model("models/model6_nn.keras")
    models["preprocessor"]  = pickle.load(open("models/preprocessor.pkl", "rb"))
    return models
models = load_models()

@st.cache_data
def load_data():
    data = pd.read_csv("data/moldova_cars_task_filtered.csv")
    data.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return data
df = load_data()

TARGET = "Price(euro)"
FEATURE_COLUMNS = df.drop(columns=[TARGET]).columns.tolist()
STYLE_OPTIONS = sorted([
    "Cabriolet", "Combi", "Coupe", "Crossover",
    "Hatchback", "Microvan", "Minivan", "Pickup",
    "Roadster", "SUV", "Sedan", "Universal"
])
FUEL_OPTIONS = sorted([
    "Diesel", "Hybrid", "Metan/Propan",
    "Petrol", "Plug-in Hybrid"
])
TRANSMISSION_OPTIONS = ["Automatic", "Manual"]
def compute_features(
    make, model_val, year, distance,
    engine_capacity, style, fuel_type, transmission
):
    current_year = 2024
    vehicle_age = max(current_year - year, 1)
    distance_per_age = distance / vehicle_age
    distance_per_year = distance / vehicle_age
    vehicle_age_sq = vehicle_age ** 2
    log_distance = np.log(distance + 1)
    log_vehicle_age = np.log(vehicle_age + 1)
    engine_sq = engine_capacity ** 2
    log_engine = np.log(engine_capacity + 1)
    age_x_distance = vehicle_age * distance
    age_x_engine = vehicle_age * engine_capacity
    distance_x_engine = distance * engine_capacity
    distance_per_engine = distance / engine_capacity if engine_capacity > 0 else 0
    engine_per_age = engine_capacity / vehicle_age
    style_encoded = {
        f"Style_{s}": 1 if s == style else 0
        for s in STYLE_OPTIONS
    }
    fuel_encoded = {
        f"Fuel_type_{f}": 1 if f == fuel_type else 0
        for f in FUEL_OPTIONS
    }
    trans_encoded = {
        f"Transmission_{t}": 1 if t == transmission else 0
        for t in TRANSMISSION_OPTIONS
    }
    row = {
        "Make": make,
        "Model": model_val,
        "Year": year,
        "Distance": distance,
        "Engine_capacity(cm3)": engine_capacity,
        "Vehicle_age": vehicle_age,
        "Distance_per_age": distance_per_age,
        **style_encoded,
        **fuel_encoded,
        **trans_encoded,
        "Distance_per_year": distance_per_year,
        "Vehicle_age_sq": vehicle_age_sq,
        "Log_Distance": log_distance,
        "Log_Vehicle_age": log_vehicle_age,
        "Engine_sq": engine_sq,
        "Log_Engine": log_engine,
        "Age_x_Distance": age_x_distance,
        "Age_x_Engine": age_x_engine,
        "Distance_x_Engine": distance_x_engine,
        "Distance_per_engine": distance_per_engine,
        "Engine_per_age": engine_per_age,
    }
    return row

def predict(input_df, model_choice):
    if model_choice == "ML3 CatBoost":
        preds = models["ML3 CatBoost"].predict(input_df)
    elif model_choice == "ML6 NeuralNet":
        X_proc = models["preprocessor"].transform(input_df)
        preds  = models["ML6 NeuralNet"].predict(X_proc, verbose=0).flatten()
    else:
        preds = models[model_choice].predict(input_df)
    return np.maximum(preds, 0)

def show_result(prediction, make, model_val, year,
                style, engine_capacity, fuel_type,
                transmission, distance):
    st.markdown("### Оценочная стоимость")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        label="Евро (EUR)",
        value=f"€ {prediction:,.0f}"
    )
    c2.metric(
        label="Доллары (USD)",
        value=f"$ {prediction * 1.16:,.0f}"
    )
    c3.metric(
        label="Рубли (RUB)",
        value=f"₽ {prediction * 85:,.0f}"
    )

    st.markdown("### Описание автомобиля")
    st.info(f"""
    **Марка / Модель:** {make} {model_val} ({year} г.)  
    **Тип кузова:** {style}  
    **Объём двигателя:** {engine_capacity:,} cm³  
    **Тип топлива:** {fuel_type}  
    **Коробка передач:** {transmission}  
    **Пробег:** {distance:,} км  
    Модель **{model_choice}** оценивает рыночную стоимость  
    данного автомобиля в **€ {prediction:,.0f}** (евро).
    """)

    if prediction > 80_000:
        st.warning("Цена очень высокая. Проверьте введённые данные.")
    elif prediction < 500:
        st.warning("Цена очень низкая. Возможно, данные некорректны.")
    else:
        st.success("Результат в пределах нормального диапазона.")

st.markdown("## Выбор модели")
col_model, col_r2 = st.columns([3, 1])
with col_model:
    model_choice = st.selectbox(
        "Выберите модель машинного обучения:",
        [
            "ML1 Ridge",
            "ML2 GradientBoosting",
            "ML3 CatBoost",
            "ML4 RandomForest",
            "ML5 Stacking",
            "ML6 NeuralNet"
        ],
        help=(
            "ML1 — Ridge Regression\n"
            "ML2 — Gradient Boosting\n"
            "ML3 — CatBoost\n"
            "ML4 — Random Forest\n"
            "ML5 — Stacking\n"
            "ML6 — Neural Network"
        )
    )
with col_r2:
    try:
        results = pickle.load(open("models/results.pkl", "rb"))
        r2_val  = results.get(model_choice)
        if r2_val:
            st.metric("R² модели", f"{r2_val:.2f}")
    except:
        pass

st.markdown("---")

tab1, tab2 = st.tabs(["Ручной ввод", "Загрузка CSV"])
with tab1:
    st.markdown("### Основные характеристики")
    st.caption("Введите характеристики автомобиля. Поля с * обязательны.")
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input(
            "Год выпуска *",
            min_value=1990,
            max_value=2024,
            value=2015,
            step=1,
            help="Год производства автомобиля"
        )
        distance = st.number_input(
            "Пробег * (км)",
            min_value=0,
            max_value=500_000,
            value=100_000,
            step=1000,
            help="Общий пробег автомобиля в километрах"
        )
    with col2:
        engine_capacity = st.number_input(
            "Объём двигателя * (cm³)",
            min_value=500,
            max_value=7000,
            value=1600,
            step=100,
            help="Рабочий объём двигателя в кубических сантиметрах"
        )
        make = st.selectbox(
            "Марка автомобиля *",
            options=sorted(df["Make"].unique()),
            help="Производитель автомобиля"
        )
    with col3:
        model_val = st.selectbox(
            "Модель автомобиля *",
            options=sorted(
                df[df["Make"] == make]["Model"].unique()
            ),
            help="Конкретная модель выбранной марки"
        )
    st.markdown("---")

    st.markdown("### Дополнительные характеристики")
    col4, col5, col6 = st.columns(3)
    with col4:
        style = st.selectbox(
            "Тип кузова *",
            options=STYLE_OPTIONS,
            index=STYLE_OPTIONS.index("Sedan"),
            help="Конфигурация кузова автомобиля"
        )
    with col5:
        fuel_type = st.selectbox(
            "Тип топлива *",
            options=FUEL_OPTIONS,
            index=FUEL_OPTIONS.index("Petrol"),
            help="Вид топлива или энергоносителя"
        )
    with col6:
        transmission = st.selectbox(
            "Коробка передач *",
            options=TRANSMISSION_OPTIONS,
            index=TRANSMISSION_OPTIONS.index("Manual"),
            help="Тип трансмиссии"
        )
    current_year = 2024
    vehicle_age = max(current_year - year, 1)
    with st.expander("Автоматически рассчитанные признаки"):
        derived_df = pd.DataFrame([
    ("Возраст (лет)", vehicle_age),
    ("Пробег / Возраст", distance / vehicle_age),
    ("Лог. пробег", np.log(distance + 1)),
    ("Лог. объём двигателя", np.log(engine_capacity + 1)),
    ("Возраст²", vehicle_age ** 2),
    ("Объём²", engine_capacity ** 2),
    ("Возраст × Пробег", vehicle_age * distance),
    ("Возраст × Объём двигателя", vehicle_age * engine_capacity),
    ("Пробег × Объём двигателя", distance * engine_capacity),
    ("Пробег / Объём двигателя", distance / engine_capacity),
], columns=["Признак", "Значение"])
        st.dataframe(
            derived_df,
            width='stretch',
            hide_index=True
        )
    st.markdown("---")

    errors = []
    if year < 1990 or year > 2024:
        errors.append("Год выпуска должен быть от 1990 до 2024")
    if distance < 0:
        errors.append("Пробег не может быть отрицательным")
    if engine_capacity < 500:
        errors.append("Объём двигателя слишком мал (мин. 500 cm³)")
    if vehicle_age > 50:
        errors.append("Возраст автомобиля слишком большой")
    if errors:
        for err in errors:
            st.error(err)

    if st.button(
        "Рассчитать стоимость",
        type="primary",
        width='stretch',
        disabled=len(errors) > 0
    ):
        row = compute_features(
            make, model_val, year, distance,
            engine_capacity, style, fuel_type, transmission
        )
        input_df = pd.DataFrame([row])[FEATURE_COLUMNS]
        try:
            preds = predict(input_df, model_choice)
            prediction = float(preds[0])
            show_result(
                prediction, make, model_val, year,
                style, engine_capacity, fuel_type,
                transmission, distance
            )
        except Exception as e:
            st.error("Ошибка при предсказании")
            st.exception(e)

with tab2:
    st.markdown("### Загрузка CSV файла")
    st.markdown("""
    Загрузите файл в формате **CSV**, содержащий данные об автомобилях.  
    Файл должен содержать следующие **обязательные** столбцы:
    """)
    required_info = pd.DataFrame([
        ("Make", "Марка автомобиля", "Текст",  "BMW"),
        ("Model", "Модель автомобиля", "Текст",  "320d"),
        ("Year", "Год выпуска", "Число",  "2015"),
        ("Distance", "Пробег", "км", "100000"),
        ("Engine_capacity(cm3)", "Объём двигателя", "cm³", "1600"),
        ("Style", "Тип кузова", "Текст", "Sedan"),
        ("Fuel_type", "Тип топлива", "Текст", "Diesel"),
        ("Transmission", "Коробка передач", "Текст", "Manual"),
    ], columns=["Столбец", "Описание", "Единица измерения", "Пример"])
    st.dataframe(required_info, width='stretch', hide_index=True)
    template_df = pd.DataFrame([
        {
            "Make": "BMW", "Model": "320d",
            "Year": 2015, "Distance": 100000,
            "Engine_capacity(cm3)": 1995,
            "Style": "Sedan",
            "Fuel_type": "Diesel",
            "Transmission": "Manual"
        },
        {
            "Make": "Toyota", "Model": "Corolla",
            "Year": 2018, "Distance": 70000,
            "Engine_capacity(cm3)": 1600,
            "Style": "Sedan",
            "Fuel_type": "Petrol",
            "Transmission": "Automatic"
        },
        {
            "Make": "Audi", "Model": "A4",
            "Year": 2012, "Distance": 200000,
            "Engine_capacity(cm3)": 2000,
            "Style": "Combi",
            "Fuel_type": "Diesel",
            "Transmission": "Manual"
        },
    ])
    st.download_button(
        label="Скачать шаблон CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="car_template.csv",
        mime="text/csv",
        help="Скачайте шаблон, заполните данными и загрузите обратно"
    )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Загрузите CSV файл",
        type=["csv"],
        help="Файл должен соответствовать шаблону выше"
    )
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown("### Предпросмотр загруженных данных")
            st.dataframe(df_upload, width='stretch')
            required_cols = [
                "Make", "Model", "Year", "Distance",
                "Engine_capacity(cm3)", "Style",
                "Fuel_type", "Transmission"
            ]
            missing_cols = [
                c for c in required_cols if c not in df_upload.columns
            ]
            if missing_cols:
                st.error(
                    f"В файле отсутствуют столбцы: "
                    f"**{', '.join(missing_cols)}**"
                )
                st.stop()
            val_errors = []
            invalid_makes = df_upload[
                ~df_upload["Make"].isin(df["Make"].unique())
            ]["Make"].unique()
            if len(invalid_makes) > 0:
                val_errors.append(
                    f"Неизвестные марки: {list(invalid_makes)}"
                )
            invalid_styles = df_upload[
                ~df_upload["Style"].isin(STYLE_OPTIONS)
            ]["Style"].unique()
            if len(invalid_styles) > 0:
                val_errors.append(
                    f"Неизвестный тип кузова: {list(invalid_styles)}"
                )
            invalid_fuels = df_upload[
                ~df_upload["Fuel_type"].isin(FUEL_OPTIONS)
            ]["Fuel_type"].unique()
            if len(invalid_fuels) > 0:
                val_errors.append(
                    f"Неизвестный тип топлива: {list(invalid_fuels)}"
                )
            if df_upload["Year"].min() < 1990:
                val_errors.append("Год выпуска слишком маленький (< 1990)")
            if df_upload["Distance"].min() < 0:
                val_errors.append("Пробег не может быть отрицательным")
            if df_upload["Engine_capacity(cm3)"].min() < 500:
                val_errors.append("Объём двигателя слишком мал (< 500 cm³)")
            if val_errors:
                for err in val_errors:
                    st.warning(err)

            if st.button(
                "Получить предсказание для всех строк",
                type="primary",
                width='stretch'
            ):
                rows = []
                for _, row_data in df_upload.iterrows():
                    r = compute_features(
                        make = row_data["Make"],
                        model_val = row_data["Model"],
                        year = int(row_data["Year"]),
                        distance = int(row_data["Distance"]),
                        engine_capacity = int(row_data["Engine_capacity(cm3)"]),
                        style = row_data["Style"],
                        fuel_type = row_data["Fuel_type"],
                        transmission = row_data["Transmission"]
                    )
                    rows.append(r)
                batch_df = pd.DataFrame(rows)[FEATURE_COLUMNS]
                preds = predict(batch_df, model_choice)
                df_result = df_upload.copy()
                df_result["Цена (EUR)"] = preds.round(0).astype(int)

                st.markdown("### Результаты")
                st.dataframe(df_result, width='stretch')
                st.download_button(
                    label="Скачать результаты CSV",
                    data=df_result.to_csv(index=False).encode("utf-8"),
                    file_name="car_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error("Ошибка при обработке файла")
            st.exception(e)
