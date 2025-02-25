import numpy as np
from PIL import Image
import PIL.Image as Image
import csv
from streamlit_echarts import st_echarts
from st_on_hover_tabs import on_hover_tabs
import streamlit as st
st.set_page_config(layout="wide")
from streamlit_drawable_canvas import st_canvas
from transformers import AutoFeatureExtractor, SwinForImageClassification
import warnings
from torchvision import transforms
from datasets import load_dataset
import cv2
import torch
from torch import nn
from typing import List, Callable, Optional
import os
import pandas as pd
import pydicom
from IPython.display import Image, display
import responses
from PIL import Image
import requests
from io import BytesIO
import io
import tensorflow
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from datetime import datetime
import streamlit as st

labels = ["Normal"]
model_name_or_path = "Santipab/Esan-code-Maimeetrang-model-action-recognition"

@st.cache_resource(show_spinner=False,ttl=1800,max_entries=2)
def FeatureExtractor(model_name_or_path):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    return feature_extractor


@st.cache_resource(show_spinner=False,ttl=1800,max_entries=2)
def LoadModel(model_name_or_path):
    model = SwinForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={int(i): c for i, c in enumerate(labels)},
        label2id={c: int(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True)
    return model


# Model wrapper to return a tensor
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

# """ Translate the category name to the category index.
#     Some models aren't trained on Imagenet but on even larger "data"sets,
#     so we can't just assume that 761 will always be remote-control.

# """
def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]
    
# """ Helper function to run GradCAM on an image and create a visualization.
#     (note to myself: this is probably useful enough to move into the package)
#     If several targets are passed in targets_for_gradcam,
#     e.g different categories,
#     a visualization for each of them will be created.
    
# """
    
def print_top_categories(model, img_tensor, top_k=5):
    feature_extractor = FeatureExtractor(model_name_or_path)
    inputs = feature_extractor(images=img_tensor, return_tensors="pt")
    outputs = model(**inputs)
    logits  = outputs.logits
    logits  = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k :][::-1]
    probabilities = nn.functional.softmax(logits, dim=-1)
    topK = dict()
    for i in indices:
        topK[model.config.id2label[i]] = probabilities[0][i].item()*100
    return topK

def swinT_reshape_transform_huggingface(tensor, width, height):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result



st.markdown('''
<style>
    section[data-testid='stSidebar'] {
        background-color: #111;
        min-width: unset !important;
        width: unset !important;
        flex-shrink: unset !important;
    }

    button[kind="header"] {
        background-color: transparent;
        color: rgb(180, 167, 141);
    }

    @media (hover) {
        /* header element to be removed */
        header["data"-testid="stHeader"] {
            display: none;
        }

        /* The navigation menu specs and size */
        section[data-testid='stSidebar'] > div {
            height: 100%;
            width: 95px;
            position: relative;
            z-index: 1;
            top: 0;
            left: 0;
            background-color: #111;
            overflow-x: hidden;
            transition: 0.5s ease;
            padding-top: 60px;
            white-space: nowrap;
        }

        /* The navigation menu open and close on hover and size */
        /* section[data-testid='stSidebar'] > div {
        height: 100%;
        width: 75px; /* Put some width to hover on. */
        /* } 

        /* ON HOVER */
        section[data-testid='stSidebar'] > div:hover{
        width: 300px;
        }

        /* The button on the streamlit navigation menu - hidden */
        button[kind="header"] {
            display: none;
        }
    }

    @media (max-width: 272px) {
        section["data"-testid='stSidebar'] > div {
            width: 15rem;
        }/.
    }
</style>
''', unsafe_allow_html=True)

# Define CSS styling for centering
centered_style = """
        display: flex;
        justify-content: center;
"""

st.markdown(
    """
<div style='border: 2px solid #00CCCC; border-radius: 5px; padding: 10px; background-color: rgba(255, 255, 255, 0.5);'>
    <h1 style='text-align: center; color: black;'>
    ❤️‍🩹 PARKINJAI 🪫
    </h1>
</div>
    """, unsafe_allow_html=True)

with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
# end def

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home','Drawing','Action','History',], 
    iconName=['🏠','📝','🚶‍♂️','📃'], 
    styles={'navtab': {'background-color': '#111', 'color': '#818181', 'font-size': '18px', 
                    'transition': '.3s', 'white-space': 'nowrap', 'text-transform': 'uppercase'}, 
                    'tabOptionsStyle': 
                    {':hover :hover': {'color': 'red', 'cursor': 'pointer'}}, 'iconStyle': 
                    {'position': 'fixed', 'left': '7.5px', 'text-align': 'left'}, 'tabStyle': 
                    {'list-style-type': 'none', 'margin-bottom': '30px', 'padding-left': '30px'}}, 
                    key="1",default_choice=0)
    st.markdown(
    """
        <div style='border: 2px solid green; padding: 10px; white; margin-top: 5px; margin-buttom: 5px; margin-right: 20px; bottom: 50;'>
            <h1 style='text-align: center; color: #3399FF; font-size: 100%'> E-SAN Thailand Coding & AI Academy  </h1>
            <h1 style='text-align: center; color: while; font-size: 100%'> 2023 </h1>
            <h1 style='text-align: center; color: #3399FF; font-size: 100%'> 🌎 Personal AI 💻 </h1>
        </div>
    """, unsafe_allow_html=True)

data_base = []
if tabs == 'Home':
    st.image('home.png',use_column_width=True)

if tabs == 'Drawing':
    image_filenames = ["d_1.png", "d_2.png", "d_3.png", "d_4.png", "d_5.png"]
    left_column, center_column, right_column , right_next_column , right_next_2_column = st.columns(5)

    for filename, column in zip(image_filenames, [left_column, center_column, right_column, right_next_column, right_next_2_column]):
        with column:
            st.image(filename, use_column_width=True)
    
    
    st.markdown(
        """
    <div style='border: 2px solid #CC99FF; border-radius: 5px; padding: 5px; background-color: white;'>
        <h3 style='text-align: center; color: orange; font-size: 180%'> 📝 Draw to Check Your Chances of Having Parkinson's.  👋 </h3>
    </div>
        """, unsafe_allow_html=True) 
    

    uploaded_files = st.file_uploader(" ", 
        type=["jpg", "jpeg", "png", "dcm"], accept_multiple_files=True)    
    if uploaded_files is not None:
        processor = AutoFeatureExtractor.from_pretrained('Santipab/Esan-code-Maimeetrang-model-drawing')
        model = SwinForImageClassification.from_pretrained('Santipab/Esan-code-Maimeetrang-model-drawing')
        answer = [] 
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(file_bytes))
            img_out = img.resize((224,224))
            img_out = np.array(img_out)
            image = img.convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            # model predicts one of the 1000 ImageNet classes
            predicted_class_idx = logits.argmax(-1).item()
            print("Predicted class:", model.config.id2label[predicted_class_idx])
            answer.append(model.config.id2label[predicted_class_idx])
    print(answer)
    if len(answer) != 0:
        if answer[0] == "healthy":
            st.markdown(
                    f"""
                    <div style='border: 2px solid green; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: green; font-size: 180%'> 💖 {answer[0].upper()} 💚 </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )   
        else:
            st.markdown(
                    f"""
                    <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: red; font-size: 180%'> ⚠️ {answer[0].upper()} 🧑‍⚕️ </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )               

if tabs == "Action":
    st.markdown(
        """
    <div style='border: 2px solid #CC99FF; border-radius: 5px; padding: 5px; background-color: white;'>
        <h3 style='text-align: center; color: #0066CC; font-size: 180%'> 🤳 Take a Video of Your Side Trip 📸 </h3>
    </div>
        """, unsafe_allow_html=True) 
    def split_video_to_images(video_path, output_folder, interval=0.5):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        frame_count = 0

        while True:
            success, frame = video_capture.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                image_path = os.path.join(output_folder, f"frame_{frame_count // frame_interval}.png")
                cv2.imwrite(image_path, frame)

            frame_count += 1

        video_capture.release()
        st.markdown(
            """
        <div style='border: 2px solid #CC99FF; border-radius: 5px; padding: 5px; background-color: white;'>
            <h3 style='text-align: center; color: #0066CC; font-size: 180%'> 🔮 Prediction the Video 📄 </h3>
        </div>
            """, unsafe_allow_html=True) 
    def main():
        uploaded_file = st.file_uploader("", type=["mp4","mov"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())

            st.video(temp_file.name)

            output_folder = "imv"
            split_video_to_images(temp_file.name, output_folder, interval=0.5)

    if __name__ == "__main__":
        main()

    processor = AutoFeatureExtractor.from_pretrained('Santipab/Esan-code-Maimeetrang-model-action-recognition') 
    model = SwinForImageClassification.from_pretrained('Santipab/Esan-code-Maimeetrang-model-action-recognition') 
    answer = [] 
    path = r"C:\Users\santi\Desktop\PMU-Esan\imv" 
    if os.path.exists(path): 
        files = os.listdir(path) 
        count_files = len(files) 

    answer = [] 
    folder_path = 'imv' # Replace with the actual folder path
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = Image.open(file_path)
        img_out = img
        img_out = np.array(img_out)
        image = img.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])
        answer.append(model.config.id2label[predicted_class_idx])
        print(answer)
    for i in range(len(answer)):
        if answer[i] == "normal":
            st.markdown(" ")
            st.markdown(
                    f"""
                    <div style='border: 2px solid green; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: green; font-size: 180%'> {answer[i]} </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )        
        else:
            st.markdown(" ")
            st.markdown(
                    f"""
                    <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: red; font-size: 180%'> {answer[i]} </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    if len(answer) != 0:                        
        csv_file_path = "database.csv"
        df = pd.read_csv(csv_file_path, parse_dates=["date"])
        current_date = datetime.now().strftime('%Y-%m-%d')
        if answer[0] == "normal":
            df = df.append({"date": current_date, "score": 0, "detail": "normal"}, ignore_index=True)
        elif answer[0] == "stage1":
            df = df.append({"date": current_date, "score": 0.33, "detail": "stage1"}, ignore_index=True)
        elif answer[0] == "stage2":
            df = df.append({"date": current_date, "score": 0.66, "detail": "stage2"}, ignore_index=True)
        elif answer[0] == "stage3":
            df = df.append({"date": current_date, "score": 0.99, "detail": "stage3"}, ignore_index=True)
        df.to_csv(csv_file_path, index=False)
if tabs == 'History':

    def plot_graph(data):
        plt.figure(figsize=(21, 9))
        plt.scatter(data['date'], data['score'], label='Points')
        plt.plot(data['date'], data['score'], linestyle='-', marker='o', color='r', label='Lines')
        plt.title('Graph Showing Past Inspection Results With the Website.')
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.xticks(rotation=-270)
        plt.legend()
        st.pyplot()
    def main():
        try:
            data = pd.read_csv('database.csv')
        except FileNotFoundError:
            st.warning("File 'database.csv' not found. Please make sure the file exists.")
            return
        except pd.errors.EmptyDataError:
            st.warning("File 'database.csv' is empty.")
            return
        if 'date' not in data.columns or 'score' not in data.columns:
            st.warning("Columns 'date' and 'score' not found in 'database.csv'. Please check your file.")
            return
        if data.empty:
            st.warning("No data found in 'database.csv'.")
        else:
            st.success("Data found in 'database.csv'. Plotting the point and line graph.")
            plot_graph(data)

    if __name__ == "__main__":
        main()
