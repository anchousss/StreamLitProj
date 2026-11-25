import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Классификатор одежды Fashion-MNIST",
    layout="centered",
)

st.title("Классификатор одежды Fashion-MNIST")
st.markdown(
    "Загрузите изображение одежды, обуви или аксессуара — "
    "модель (CNN, обученная на Fashion-MNIST) определит категорию."
)

with st.sidebar:
    st.header("ℹ️ О сервисе")
    st.write("**Модель:** CNN, Fashion-MNIST")
    st.write("**Прошло эпох:** 20")
    st.write("**Точность (test):** ~92%")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        st.write("**API:** ✅ доступен" if r.status_code == 200 else "**API:** ❌ ошибка")
    except Exception:
        st.write("**API:** ❌ недоступен")

uploaded_file = st.file_uploader(
    "Выберите изображение...", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Загруженное изображение", width="stretch")

    if st.button("🔍 Классифицировать", width="stretch"):
        with st.spinner("Анализируем..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    best_ru = result.get("predicted_class_ru", result["predicted_class"])
                    conf = result["confidence"]

                    with col2:
                        st.success(f"**{best_ru}**")
                        st.caption(f"({result['predicted_class']})")
                        st.metric("Уверенность", f"{conf:.2%}")

                    if conf < 0.5:
                        st.warning(
                            f"⚠️ Низкая уверенность ({conf:.0%}). Модель обучена на "
                            "Fashion-MNIST (28×28, чёрный фон) и может ошибаться на "
                            "реальных фото с фоном. Для лучшего результата загрузите "
                            "фото с однотонным фоном, где вещь занимает большую часть кадра."
                        )
                    elif conf < 0.75:
                        st.info(f"Уверенность средняя ({conf:.0%}).")

                    st.subheader("Возможные варианты")
                    for pred in result["top_predictions"]:
                        name_ru = pred.get("class_name_ru", pred["class_name"])
                        st.progress(
                            pred["confidence"],
                            text=f"{name_ru} — {pred['confidence']:.2%}",
                        )

                    if "preprocessed_image_base64" in result:
                        st.subheader("Что «видит» модель (28×28)")
                        processed_img = Image.open(
                            BytesIO(base64.b64decode(result["preprocessed_image_base64"]))
                        )
                        st.image(
                            processed_img,
                            width=150,
                            caption="Вход модели после препроцессинга",
                        )
                else:
                    st.error(f"Ошибка API {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Не удалось соединиться с API: {e}")
                st.info("Проверьте, что API запущен.")
