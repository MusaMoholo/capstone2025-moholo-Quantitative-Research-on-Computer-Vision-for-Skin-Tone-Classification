# app.py
import streamlit as st
from scripts.app.utils import fake_model_prediction
import time

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "results" not in st.session_state:
    st.session_state.results = None

# Upload Page
if st.session_state.page == "upload":
    st.title("📸 Skin Tone Classification")
    st.write("Upload or take a photo of your skin to get started.")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        if st.button("Analyze Skin Tone"):
            st.session_state.page = "loading"
            st.session_state.image = uploaded_image

# Loading Page
elif st.session_state.page == "loading":
    st.title("⏳ Processing...")
    with st.spinner("Analyzing your skin tone..."):
        time.sleep(2)  # simulate latency
        st.session_state.results = fake_model_prediction()
    st.session_state.page = "results"
    st.rerun()


# Results Page
elif st.session_state.page == "results":
    st.title("🧾 Results")
    st.write("Here are your skin tone classification probabilities:")

    st.table(st.session_state.results)

    if st.button("Try another image"):
        st.session_state.page = "upload"
        st.session_state.results = None