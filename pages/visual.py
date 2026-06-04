import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="Визуализации",
    layout="wide"
)
st.title("Визуализация данных об автомобилях")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv('data/moldova_cars_task_filtered.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

df = load_data()

st.header("1. Распределение цен на автомобилях")
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
axes1[0].hist(df['Price(euro)'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes1[0].set_xlabel('Цена (€)')
axes1[0].set_ylabel('Количество автомобилей')
axes1[0].set_title('Распределение цен')
axes1[0].axvline(
    df['Price(euro)'].mean(), color='red',
    linestyle='--',
    label=f"Среднее: €{df['Price(euro)'].mean():,.0f}"
)
axes1[0].legend()
axes1[1].hist(np.log(df['Price(euro)']), bins=50, color='coral', edgecolor='black', alpha=0.7)
axes1[1].set_xlabel('log(Цена)')
axes1[1].set_ylabel('Количество')
axes1[1].set_title('Логарифм цен (нормализованное)')
plt.tight_layout()
st.pyplot(fig1)
plt.close()

st.header("2. Зависимость цены от пробега")
fig2, ax2 = plt.subplots(figsize=(12, 5))
scatter = ax2.scatter(
    df['Distance'], df['Price(euro)'],
    c=df['Vehicle_age'], cmap='plasma',
    alpha=0.5, s=20
)
plt.colorbar(scatter, ax=ax2, label='Возраст авто (лет)')
ax2.set_xlabel('Пробег (км)')
ax2.set_ylabel('Цена (€)')
ax2.set_title('Цена vs Пробег (цвет = возраст автомобиля)')
st.pyplot(fig2)
plt.close()

st.header("3. Средняя цена по году выпуска")
year_price = df.groupby('Year')['Price(euro)'].mean().reset_index()
fig3, ax3 = plt.subplots(figsize=(14, 5))
ax3.plot(year_price['Year'], year_price['Price(euro)'],
         'o-', color='green', linewidth=2, markersize=6)
ax3.fill_between(year_price['Year'], year_price['Price(euro)'],
                 alpha=0.2, color='green')
ax3.set_xlabel('Год выпуска')
ax3.set_ylabel('Средняя цена (€)')
ax3.set_title('Динамика средней цены по годам выпуска')
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)
plt.close()

st.header("4. Корреляционная матрица")
available_features = df.columns.tolist()
desired_features = [
    'Price(euro)', 'Distance', 'Engine_capacity(cm3)',
    'Vehicle_age', 'Year', 'Distance_per_year',
    'Log_Distance', 'Log_Engine', 'Vehicle_age_sq'
]
key_features = [f for f in desired_features if f in available_features]
corr = df[key_features].corr()
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr, annot=True, fmt='.2f', cmap='coolwarm',
    center=0, ax=ax4, square=True,
    linewidths=0.5, annot_kws={"size": 9}
)
ax4.set_title('Корреляция между ключевыми признаками')
plt.tight_layout()
st.pyplot(fig4)
plt.close()

st.header("5. Распределение цен по типу кузова")
style_columns = [col for col in df.columns if col.startswith('Style_')]
@st.cache_data
def get_style_col(dataframe):
    def get_style(row):
        for col in style_columns:
            if row[col] == 1:
                return col.replace('Style_', '')
        return 'Unknown'
    dataframe = dataframe.copy()
    dataframe['Style'] = dataframe.apply(get_style, axis=1)
    return dataframe

df = get_style_col(df)
fig5, ax5 = plt.subplots(figsize=(14, 6))
df.boxplot(column='Price(euro)', by='Style', ax=ax5)
ax5.set_xlabel('Тип кузова')
ax5.set_ylabel('Цена (€)')
ax5.set_title('Распределение цен по типу кузова')
plt.suptitle('')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig5)
plt.close()

st.header("6. Цена по типу топлива")
fuel_columns = [col for col in df.columns if col.startswith('Fuel_type_')]
@st.cache_data
def get_fuel_col(dataframe):
    def get_fuel(row):
        for col in fuel_columns:
            if row[col] == 1:
                return col.replace('Fuel_type_', '')
        return 'Unknown'
    dataframe = dataframe.copy()
    dataframe['Fuel'] = dataframe.apply(get_fuel, axis=1)
    return dataframe

df = get_fuel_col(df)
fig6, ax6 = plt.subplots(figsize=(12, 6))
sns.violinplot(
    data=df,
    x='Fuel',
    y='Price(euro)',
    hue='Fuel',
    palette='Set2',
    legend=False,
    ax=ax6
)
ax6.set_xlabel('Тип топлива')
ax6.set_ylabel('Цена (€)')
ax6.set_title('Violin Plot: Цена по типу топлива')
plt.tight_layout()
st.pyplot(fig6)
plt.close()

st.header("7. Цена vs Объём двигателя")
fig7, ax7 = plt.subplots(figsize=(12, 5))
for fuel in df['Fuel'].unique():
    mask = df['Fuel'] == fuel
    ax7.scatter(
        df.loc[mask, 'Engine_capacity(cm3)'],
        df.loc[mask, 'Price(euro)'],
        alpha=0.4, s=15, label=fuel
    )
ax7.set_xlabel('Объём двигателя (cm³)')
ax7.set_ylabel('Цена (€)')
ax7.set_title('Зависимость цены от объёма двигателя')
ax7.legend(title='Тип топлива', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
st.pyplot(fig7)
plt.close()
