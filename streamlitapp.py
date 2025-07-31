import streamlit as st
import requests
import json

import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="XGBoost Конверсия", page_icon="🎯", layout="centered")
st.title("🎯 XGBoost Предсказания")

# --- Ввод данных ---
st.header("🔹 Введите данные для предсказания")
age = st.number_input("Возраст", min_value=18, max_value=100, value=25)
ad_spend = st.number_input("Реклама (AdSpend)", min_value=0, value=100, step=1, format="%d")
ctr = st.slider("CTR (Click-Through Rate) — это доля кликов по рекламе или предложению от общего числа её показов. Например, если показали 100 раз и кликнули 10 раз, CTR = 10%", min_value=0.0, max_value=1.0, value=0.05)
visits = st.number_input("Посещения сайта", min_value=0, value=10)
time_on_site = st.number_input("Время на сайте (сек)", min_value=0.0, value=30.0)

gender_male = st.selectbox("Пол", ["Мужской", "Женский"])
gender_male = 1 if gender_male == "Мужской" else 0

channel = st.selectbox("Канал кампании", ["PPC", "Referral", "SEO", "Social Media"])
channels = {
    "CampaignChannel_PPC": 0,
    "CampaignChannel_Referral": 0,
    "CampaignChannel_SEO": 0,
    "CampaignChannel_Social Media": 0
}
channels[f"CampaignChannel_{channel}"] = 1

# --- Кнопка ---
if st.button("Сделать предсказание"):
    payload = {
        "Age": age,
        "AdSpend": ad_spend,
        "ClickThroughRate": ctr,
        "WebsiteVisits": visits,
        "TimeOnSite": time_on_site,
        "Gender_Male": gender_male,
        **channels
    }

    try:
        r = requests.post(f"{API_URL}/predict", json=payload)
        if r.status_code == 200:
            result = r.json()
            st.success(f"✅ Предсказание: {result['prediction']} (вероятность: {result['probability']})")
        else:
            st.error(f"Ошибка: {r.status_code} - {r.text}")
    except Exception as e:
        st.error(f"Ошибка соединения: {e}")

# --- История ---
st.header("📜 История предсказаний")
if st.button("Загрузить историю"):
    try:
        r = requests.get(f"{API_URL}/predictions")
        if r.status_code == 200:
            history = r.json()
            if len(history) == 0:
                st.info("История пуста.")
            else:
                df = pd.DataFrame(history)
                st.dataframe(df)
        else:
            st.error(f"Ошибка загрузки истории: {r.status_code}")
    except Exception as e:
        st.error(f"Ошибка соединения: {e}")


# --- Скачать JSON ---
st.header("📥 Скачать историю")
r = requests.get(f"{API_URL}/predictions")
if r.status_code == 200:
    history = r.json()
    if len(history) > 0:
        json_data = json.dumps(history, indent=4, ensure_ascii=False)
        st.download_button(
            label="📥 Скачать predictions.json",
            data=json_data,
            file_name="predictions.json",
            mime="application/json"
        )
    else:
        st.info("История пуста, нечего скачивать.")
else:
    st.error("Не удалось загрузить историю.")

