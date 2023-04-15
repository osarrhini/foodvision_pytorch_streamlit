### 1. Imports and class names setup ### 
#import gradio as gr
#import os
import torch
import streamlit as st
from PIL import Image
import pandas as pd

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["pizza", "steak", "sushi"]

### 2. Model and transforms preparation ###
@st.cache_resource()
def load_model():
    # Create EffNetB2 model
    effnetb2, effnetb2_transforms = create_effnetb2_model(
        num_classes=3, # len(class_names) would also work
    ) 
    # Load saved weights
    effnetb2.load_state_dict(
        torch.load(
            f="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth",
            map_location=torch.device("cpu"),  # load to CPU
        )
    )
    return effnetb2, effnetb2_transforms

### Load model ###
with st.spinner("Loading model into memory..."):
    effnetb2, effnetb2_transforms = load_model()

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0).cpu()
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Streamlit app ###
title = "FoodVision Mini 🍕🥩🍣"
st.header(title)

sub_title = "This application tries to predict if the displayed image corresponds to a **pizza**, a **steak** or a **sushi**."
st.markdown(sub_title)

file = st.file_uploader(
    label="Please upload an image",
    type=["jpg", "png"]
)

if file is None:
    st.text("Please import a valid png or jpg image")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    btn_predict = st.button(
        label="Predict",
        help="Click to make inference on the image"
    )
    if btn_predict:
        pred_labels_and_probs_dict, prediction_time = predict(image)
        st.text(f"Elapsed time: {prediction_time} seconds")

        # Write Results as a DataFrame object        
        st.dataframe(
            data = pd.DataFrame(pred_labels_and_probs_dict, index=['Probability'])
        )
    