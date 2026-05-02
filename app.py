import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. Set up the app title and layout
st.set_page_config(page_title="Gender Classifier", page_icon="👤")
st.title("👤 Gender Classification using ViT")
st.write("Upload an image to predict the gender (Male/Female).")

# 2. Cache the model (Updated to rizwandari/gender-classification-vit)
@st.cache_resource
def load_gender_classifier():
    # This model requires 'timm' library installed via requirements.txt
    return pipeline("image-classification", model="rizwandari/gender-classification-vit")

with st.spinner("Loading AI Model..."):
    gender_classifier = load_gender_classifier()

# 3. File uploader for user images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Using columns for a cleaner UI
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Analyzing gender..."):
            # Classify gender
            results = gender_classifier(image)
            
            # Sort predictions by score (highest first)
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Display the top result
            top_prediction = results[0]
            label = top_prediction['label'].upper()
            score = top_prediction['score']
            
            st.subheader("Result")
            if "FEMALE" in label:
                st.success(f"**Predicted Gender: {label}**")
            else:
                st.info(f"**Predicted Gender: {label}**")
                
            st.metric(label="Confidence", value=f"{score:.2%}")
            
            # Show all probabilities in a small chart
            with st.expander("See detailed probabilities"):
                labels = [p['label'] for p in results]
                scores = [p['score'] for p in results]
                st.bar_chart(data=dict(zip(labels, scores)))
else:
    st.info("Please upload an image to start.")
