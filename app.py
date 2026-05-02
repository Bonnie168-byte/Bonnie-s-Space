import streamlit as st
from transformers import pipeline
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Gender Classifier", page_icon=":material/transgender:")

# Title and Description
st.title(":material/face: Gender Classification")
st.write("Upload a face image to classify the gender using a Vision Transformer (ViT) model.")

# Load Model with Caching to prevent reloading on every interaction
@st.cache_resource
def load_gender_model():
    # Model: rizwandari/gender-classification-vit
    return pipeline("image-classification", model="rizwandari/gender-classification-vit")

with st.spinner("Loading AI Model..."):
    gender_pipe = load_gender_model()

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process Image
    image = Image.open(uploaded_file).convert("RGB")
    
    # UI Layout: Two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            # Run Inference
            results = gender_pipe(image)
            
            # Get Top Result
            # The model usually returns a list of labels/scores sorted by probability
            top_result = results[0]
            label = top_result['label']
            score = top_result['score']
            
            # Display Result
            st.subheader("Result")
            if label.lower() == "male":
                st.info(f"Gender: **{label}**")
            else:
                st.success(f"Gender: **{label}**")
                
            st.metric(label="Confidence Score", value=f"{score:.2%}")
            
            # Details Expander
            with st.expander("Show detailed probabilities"):
                for res in results:
                    st.write(f"{res['label']}: {res['score']:.4f}")
else:
    st.info("Please upload an image file to start classification.")
