import streamlit as st
st.set_page_config(page_title="Разработчик", layout="wide")
st.title("Информация о разработчике")
st.markdown("---")

col1, col2 = st.columns([1, 2])
with col1:
        st.image(
            "assets/photo.jpg",
            width=250,
            caption="Фото разработчика"
        )
with col2:
    st.markdown(
        """
        <div style="margin-top: 30px;"></div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### Персональные данные")
    st.markdown("""
        <div style="
            background-color: rgba(255,255,255,0.05);
            padding: 15px 20px;
            border-radius: 10px;
            line-height: 1.8;
        ">
        <b>ФИО:</b> Леонкин Максим Александрович<br>
        <b>Группа:</b> ФИТ-241<br>
        <b>Дисциплина:</b> Машинное обучение и большие данные
        </div>
""", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("### Тема РГР")
    st.success("""
    **«Разработка Web-приложения (дашборда) для инференса (вывода)
    моделей ML и анализа данных»**
    """)
    st.markdown("")
st.markdown("---")
