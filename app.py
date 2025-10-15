# app.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import joblib
import logging

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    # Load the full model using joblib if necessary or just load the state_dict into a new model structure
    model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=4)
    model.load_state_dict(torch.load('saved_weights.pt'))
    model.eval()
    logging.info("Model loaded successfully.")
    return model

model = load_model()

# Function to make predictions
def predict(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    predictions = torch.argmax(output.logits, dim=-1).squeeze().tolist()
    # Transform predictions to labels if necessary
    return predictions

#### 2. Streamlit Interface for User Inputs

st.title("NER Prediction App")
st.write("Upload a text file or enter text directly to get NER predictions.")

input_text = st.text_area("Enter your text here:", "")

if st.button("Predict"):
    if input_text:
        labels = predict(input_text, model, tokenizer)
        st.write("Predicted Labels:")
        st.write(labels)
        logging.info("Predictions made for user input.")

#### 3. Handling File Uploads

uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    lines = content.split('\n')
    if st.button("Predict File Content"):
        results = {line: predict(line, model, tokenizer) for line in lines}
        st.write("Batch Predictions:")
        for line, result in zip(lines, results):
            st.write(f"Text: {line}")
            st.write(f"Prediction: {result}")
        logging.info("Predictions made for uploaded file.")

#### 4. Additional Features and Error Handling
# You can add more features such as visualizations, model evaluations, etc.

# Run the app with:
# streamlit run app.py

