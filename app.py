import streamlit as st
from transformers import pipeline
from PIL import Image

# 页面配置
st.set_page_config(page_title="Gender Classifier", page_icon="👤")

@st.cache_resource
def load_gender_model():
    # 明确指定使用 timm 引擎
    return pipeline(
        "image-classification", 
        model="rizwandari/gender-classification-vit"
    )

st.title("Gender Classification")

try:
    with st.spinner("Downloading/Loading model... This may take a minute on first run."):
        gender_pipe = load_gender_model()
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, width=300)
        results = gender_pipe(img)
        st.write(f"Result: **{results[0]['label']}** (Confidence: {results[0]['score']:.2%})")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Tip: Ensure 'timm' is added to your requirements.txt and you have a stable internet connection.")
