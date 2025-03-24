import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
import CNN
import torch
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import time
from PIL import Image
import os
import torchvision.transforms.functional as TF
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')
import gdown
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import CNN  # Ensure CNN.py defines the correct model architecture
from PIL import Image
import requests
YOUTUBE_API_KEY = "AIzaSyDYEeSTrT7pPpVzpmaJ491gxogVxfWwpvM"
file_id = "1pxdrNiOivql6s0ArWE18AQ5HiXiSUrwj"
output = "plant_disease_model_1_latest.pt"

url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, output, quiet=False)

def fetch_agriculture_news():
    API_KEY = "933be40f4f7f42c5bdc3d127e5c2cc58"  # Replace with your NewsAPI key
    url = "https://newsapi.org/v2/everything"
    
    params = {
        "q": "farming",
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": API_KEY
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        return []

def fetch_youtube_videos(query, max_results=6):
    url = f"https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'key': YOUTUBE_API_KEY,
        'maxResults': max_results
    }
    response = requests.get(url, params=params)
    videos = []
    if response.status_code == 200:
        data = response.json()
        for item in data['items']:
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            videos.append({'video_id': video_id, 'title': video_title})
    return videos


# Step 1: Download the model from Google Drive
output = "plant_disease_model_1_latest.pt"


# Step 2: Load the model
model = CNN.CNN(39)  # Ensure CNN.CNN is defined in your CNN.py file
model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))
model.eval()


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index
def info_box(price_per_kg):
    def format_number_indian(num):
        num_str = f"{num:,}"
        parts = num_str.split(",")
        if len(parts) > 2:
            return parts[0] + "," + ",".join(parts[1:]).replace(",", "_", 1).replace("_", ",")
        return num_str
    
    return f"""
        <div style="
            background-color: rgba(123, 216, 237, 0.6);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            color: black;
            font-weight: bold;
            width: 50%;
            margin: 0 auto;
        ">
            <span style='color: red;'>The present price:</span> <span style='color: black;'>{format_number_indian(price_per_kg)} INR per kg</span><br>
        </div>
    """
def english_page():
    # Paths for model, scaler, and assets
    loaded_model = joblib.load('gradient_boosting_model.pkl')
    # Prediction Function
    def predict_weather(input_array):
        real_time_input = [[input_array[0][0], input_array[0][1], input_array[0][2], input_array[0][3]]]
        prediction_proba = loaded_model.predict_proba(real_time_input)
        weather_labels = ["drizzle", "fog", "rain", "snow", "sun"]
        output = dict(zip(weather_labels, prediction_proba[0]))
        output = {k: v for k, v in output.items()}
        output["result"] = np.argmax(prediction_proba)
        return output
    user_data = st.session_state.get('user', None)
    lang=user_data[5]
    name=user_data[1]

    st.markdown(
    f"""
    <h1 style="text-align: center; color: black;">
        Welcome {name}  👋✨
    </h1>
    """,
    unsafe_allow_html=True
    )
    def user_profile():
        st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url("https://media.istockphoto.com/id/1384550157/photo/rising-wheat-prices-in-europe-due-to-the-conflict-between-russia-and-ukraine-flour-and-wheat.jpg?s=612x612&w=0&k=20&c=gobiFbQgqkA4FLK3BeSKfaMkangmtjIANitMFp0yxvg=");  
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 255, 255, 0.5); /* Add a semi-transparent overlay */
            background-blend-mode: overlay; /* Blend the image with the overlay */
        }
        </style>
        """,
        unsafe_allow_html=True
        )
        # Input Section

        with st.form(key='my_form'):
            col1,col2,col3=st.columns([2,3,2])
            crop_name=col2.selectbox("Select Crop",('Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Barley', 'Bean', 'Black pepper', 'Blackgram', 'Bottle Gourd', 'Brinjal', 'Cabbage', 'Cardamom', 'Carrot', 'Cashewnut', 'Castor seed', 'Cauliflower', 'Citrus Fruit', 'Coconut', 'Coffee', 'Coriander', 'Cotton', 'Cowpea', 'Drum Stick', 'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 'Gram', 'Grapes', 'Groundnut', 'Guar seed', 'Horse-gram', 'Jack Fruit', 'Jowar', 'Jute', 'Khesari', 'Korra', 'Lemon', 'Lentil', 'Linseed', 'Maize', 'Mango', 'Masoor', 'Mesta', 'Moong(Green Gram)', 'Moth', 'Niger seed', 'Oilseeds total', 'Onion', 'Orange', 'Other  Rabi pulses', 'Other Cereals & Millets', 'Other Citrus Fruit', 'Other Dry Fruit', 'Other Fresh Fruits', 'Other Kharif pulses', 'Other Vegetables', 'Papaya', 'Peach', 'Pear', 'Peas & beans (Pulses)', 'Pineapple', 'Plums', 'Pome Fruit', 'Pome Granet', 'Potato', 'Pulses total', 'Pump Kin', 'Ragi', 'Rajmash Kholar', 'Rapeseed &Mustard', 'Redish', 'Ribed Guard', 'Rice', 'Rubber', 'Safflower', 'Samai', 'Sannhamp', 'Sapota', 'Sesamum', 'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato', 'Tapioca', 'Tea', 'Tobacco', 'Tomato', 'Total foodgrain', 'Turmeric', 'Turnip', 'Urad', 'Varagu', 'Water Melon', 'Wheat', 'Yam', 'other oilseeds', 'other misc. pulses', 'other fibres', 'other cereals', 'other vegetables', 'Total Pulses', 'Total foodgrain', 'Total fruits', 'Total vegetables'))

            # Make prediction
            col1,col2,col3=st.columns([2.3,2,1])
            if col2.form_submit_button('Predict',type='primary') and crop_name:
                pri=pd.read_csv('crop_prices.csv')
                price=pri[pri['Crop Name']==crop_name]['Price 2025 (INR per kg)'].values[0]
                st.markdown(info_box(price), unsafe_allow_html=True)
                #show previous price vs predicted price
                pri=pd.read_csv('crop_prices.csv')
                st.write(" ")
                st.write(" ")
                # Get price
                #show plot for next 5 years
                price_2020 = pri[pri['Crop Name'] == crop_name]['Price 2020'].values[0]
                price_2025 = pri[pri['Crop Name'] == crop_name]['Price 2025 (INR per kg)'].values[0]

                # Generate price trend from 2020 to 2025
                years_past = np.arange(2020, 2026)
                prices_past = np.   linspace(price_2020, price_2025, num=len(years_past)) + np.random.uniform(-1, 1, size=len(years_past))

                # Forecast next 5 years (2026-2030) using a simple linear growth model
                slope = (price_2025 - price_2020) / (2025 - 2020)
                years_future = np.arange(2026, 2031)
                prices_future = [price_2025 + slope * (year - 2025) + np.random.uniform(-1, 1) for year in years_future]

                # Combine past and forecasted data
                all_years = np.concatenate((years_past, years_future))
                all_prices = np.concatenate((prices_past, prices_future))

                # Plot the price trend with forecast
                fig, ax = plt.subplots()
                ax.plot(all_years, all_prices, marker='o', linestyle='-', color='blue', label=f'Price Trend for {crop_name}')
                ax.axvline(x=2025, color='red', linestyle='--', label='Forecast Start')
                ax.set_title(f'Price Forecast for {crop_name} (2020-2030)')
                ax.set_xlabel('Year')
                ax.set_ylabel('Price (INR per kg)')
                ax.legend()
                col1,col2,col3=st.columns([1,3,1])
                col2.pyplot(fig)

    def diseases():
        st.markdown(
            """
            <style>
            /* Apply background image to the main content area */
            .main {
                background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20230610/pngtree-close-up-shot-of-a-plant-with-some-brown-spots-on-image_2957649.jpg");  
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-color: rgba(255, 255, 255, 0.5); /* Add a semi-transparent overlay */
                background-blend-mode: overlay; /* Blend the image with the overlay */
            }
            </style>
            """,
            unsafe_allow_html=True
            )
        # File uploader
        col1, col2, col3 = st.columns([1, 3, 1])
        image = col2.file_uploader("Upload an image of the plant", type=['jpg', 'jpeg', 'png'])
        if image:
            col1, col2, col3 = st.columns([5, 6, 1])
            col2.image(image, caption='Uploaded Image',width=250)
            try:
                # Perform prediction
                pred = prediction(image)

                # Fetch details based on prediction
                title = disease_info['disease_name'][pred]
                description = disease_info['description'][pred]
                prevent = disease_info['Possible Steps'][pred]
                image_url = disease_info['image_url'][pred]
                supplement_name = supplement_info['supplement name'][pred]
                supplement_image_url = supplement_info['supplement image'][pred]
                supplement_buy_link = supplement_info['buy link'][pred]

                # Display results
                col1, col2, col3 = st.columns([4, 6, 1])
                col2.markdown(f"<h2 style='color:red;'>{title}</h2>", unsafe_allow_html=True)
                col1,col2=st.columns([5,5])
                col1.markdown(
                    f"""
                    <div style="text-align: justify; padding: 10px; background-color: #d3e876; border-radius: 20px; border: 1.5px solid black; margin-bottom: 20px;">
                        <h2 style="color: #111df7; font-size: 20px;"><b>Disease Description:</b></h2>
                        <p style="color: black; font-size: 15px;"><b>{description}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                col2.markdown(
                    f"""
                    <div style="text-align: justify; padding: 10px; background-color: #ffa1ef; border-radius: 20px; border: 1.5px solid black; margin-bottom: 20px;">
                        <h2 style="color: #111df7; font-size: 20px;"><b>Prevntion Steps:</b></h2>
                        <p style="color: black; font-size: 15px;"><b>{prevent}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(f"")
                col1, col2, col3 = st.columns([3, 4, 3])
                col2.markdown(
                    f"""
                    <div style="text-align: center; padding: 8px; background-color: #ffd5a1; border-radius: 30px; border: 1.5px solid black; margin-bottom: 10px;">
                        <h2 style="color: #111df7; font-size: 20px;"><b>Recommended Supplement:</b> {supplement_name}</h2>
                        <div style="text-align: center; margin-top: 10px;">
                            <img src="{supplement_image_url}" alt="Supplement Image" style="width: 300px; height: auto; border-radius: 15px; border: 1px solid black;">
                        </div>
                        <div style="margin-top: 15px;">
                            <a href="{supplement_buy_link}" target="_blank" style="text-decoration: none;">
                                <button style="background-color: red; color: white; font-size: 16px; padding: 10px 20px; border: none; border-radius: 10px; cursor: pointer;">
                                    Buy Supplement Here
                                </button>
                            </a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except:
                st.error('Invalid Image')      
    # Navigation menu for user dashboard
    selected_tab = option_menu(
        menu_title=None,
        options=["Price",'Diseases','Pests','Weeds','News','Logout'],
        icons=['cash-coin','prescription2','bug-fill','flower1','newspaper','unlock-fill'], menu_icon="cast", default_index=0,
        orientation="horizontal",
    styles={
    "nav-link-selected": {"background-color": "#62f088", "color": "black", "border-radius": "5px"},
    }
    )
    if selected_tab == "Price":
        user_profile()
    elif selected_tab=='Diseases':
        diseases()
    elif selected_tab=='Logout':
        # Logout functionality
        st.cache()
        st.session_state.clear()
        st.experimental_rerun()
    elif selected_tab=='Pests':
        st.markdown(
            """
            <style>
            /* Apply background image to the main content area */
            .main {
                background-image: url("https://bloximages.chicago2.vip.townnews.com/mankatofreepress.com/content/tncms/assets/v3/editorial/c/fe/cfe13d3c-282f-11ef-a581-3bd03e631fc8/628d9c250e467.image.jpg?resize=1024%2C770");  
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-color: rgba(255, 255, 255, 0.5); /* Add a semi-transparent overlay */
                background-blend-mode: overlay; /* Blend the image with the overlay */
            }
            </style>
            """,
            unsafe_allow_html=True
            )
        # File uploader
        col1, col2, col3 = st.columns([1, 3, 1])
        model = YOLO('best.pt')
        # File uploader
        uploaded_file = col2.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Convert uploaded file to an OpenCV format
            image = Image.open(uploaded_file)
            image = np.array(image)
            
            # Run YOLOv8 inference on the image
            results = model(image)
            annotated_image = results[0].plot()
            
            # Display original and annotated images
            # Extract the class with the highest detection confidence
            max_detection = max(results[0].boxes.data, key=lambda x: x[4].item(), default=None)
            def disease_info_box(disease_name):
                return f"""
                    <div style="
                        background-color: rgba(255, 255, 255, 0.6);
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        font-size: 30px;
                        font-weight: bold;
                        color: red;">
                        {disease_name}
                    </div>
                """
            if max_detection is not None:
                col1,col2,col3=st.columns([1,3,1])
                with col2:
                    col3,col4=st.columns([6,4])
                    col3.image(uploaded_file, caption='Processed Image')
                    col4.markdown(disease_info_box(model.names[int(max_detection[5].item())].upper()), unsafe_allow_html=True)
                class_id = int(max_detection[5].item())  # Extract class ID
                class_name = model.names[class_id]  # Get class name
                query=class_name+' pest control in english'
                col1,col2,col3=st.columns([1,3,1])
                videos = fetch_youtube_videos(query)
                for i in range(0, len(videos), 2):
                    cols = col2.columns(2)  # Create 3 columns
                    for j, video in enumerate(videos[i:i+2]):  # Iterate over videos for the current row
                        with cols[j]:
                            st.video(f"https://www.youtube.com/watch?v={video['video_id']}")
            else:
                st.markdown(
                    """
                    <style>
                    /* Apply background image to the main content area */
                    .main {
                        background-image: url("");  
                        background-size: cover;
                        background-position: center;
                        background-repeat: no-repeat;
                        background-color: rgba(255, 255, 255, 0.5); /* Add a semi-transparent overlay */
                        background-blend-mode: overlay; /* Blend the image with the overlay */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                    )
                col1,col2,col3=st.columns([1,3,1])
                col2.image('https://img.freepik.com/free-vector/hand-drawn-no-data-concept_52683-127823.jpg', caption='No Pests Detected',use_column_width=True)
    elif selected_tab=='Weeds':
        st.markdown(
            """
            <style>
            /* Apply background image to the main content area */
            .main {
                background-image: url("https://img.freepik.com/free-vector/botanical-cannabis-leaf-background_23-2148778834.jpg");  
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-color: rgba(255, 255, 255, 0.5); /* Add a semi-transparent overlay */
                background-blend-mode: overlay; /* Blend the image with the overlay */
            }
            </style>
            """,
            unsafe_allow_html=True
            )
        # File uploader
        col1, col2, col3 = st.columns([1, 3, 1])
        labels_path = 'obj.names'
        LABELS = open(labels_path).read().strip().split("\n")

        # Load YOLO model
        weights_path = 'crop_weed_detection.weights'
        config_path = 'crop_weed.cfg'

        # File uploader
        uploaded_file = col2.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            # Convert to OpenCV format
            image = Image.open(uploaded_file)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            (H, W) = image.shape[:2]

            # Load YOLO model
            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

            # Parameters
            confi = 0.5
            thresh = 0.5

            # Get output layer names
            ln = net.getLayerNames()
            ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

            # Prepare image for YOLO
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
            net.setInput(blob)
            layer_outputs = net.forward(ln)
            # Process detections
            boxes = []
            confidences = []
            classIDs = []

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > confi:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # Apply Non-Maximum Suppression
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)

            if len(idxs) > 0:
                idxs = idxs.flatten()

                # Get max confidence detection
                max_index = idxs[np.argmax([confidences[i] for i in idxs])]

                # Extract the bounding box coordinates
                (x, y) = (boxes[max_index][0], boxes[max_index][1])
                (w, h) = (boxes[max_index][2], boxes[max_index][3])

                # Draw bounding box
                color = (255, 255, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = f"{LABELS[classIDs[max_index]]} : {confidences[max_index]:.4f}"
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Display output
                col1,col2,col3=st.columns([1,3,1])
                col2.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=LABELS[classIDs[max_index]],use_column_width=True)
                if LABELS[classIDs[max_index]]=='weed':
                    query=LABELS[classIDs[max_index]]+' weed control in english'
                    col1,col2,col3=st.columns([1,3,1])
                    videos = fetch_youtube_videos(query)
                    for i in range(0, len(videos), 2):
                        cols = col2.columns(2)
                        for j, video in enumerate(videos[i:i+2]):
                            with cols[j]:
                                st.video(f"https://www.youtube.com/watch?v={video['video_id']}")
            else:
                st.markdown(
                    """
                    <style>
                    /* Apply background image to the main content area */
                    .main {
                        background-image: url("");  
                        background-size: cover;
                        background-position: center;
                        background-repeat: no-repeat;
                        background-color: rgba(255, 255, 255, 0.5); /* Add a semi-transparent overlay */
                        background-blend-mode: overlay; /* Blend the image with the overlay */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                    )
                col1,col2,col3=st.columns([1,3,1])
                col2.image('https://img.freepik.com/free-vector/hand-drawn-no-data-concept_52683-127823.jpg', caption='No Weed Detected',use_column_width=True)

    elif selected_tab=='News':
        st.write(" ")
        st.write(" ")
        st.write(" ")
        try:
            articles = fetch_agriculture_news()
            col1,col2,col3=st.columns([1,3,1])
            #place select box to slecte for news or youtube videos
            option=col3.selectbox('Select Category',['News','Videos'])
            if option=='News':  
                if articles:
                    cols = st.columns(2)  # Create two columns
                    for i, article in enumerate(articles[:6]):  # Display top 6 articles (2 per row)
                        with cols[i % 2]:
                            with st.container():
                                st.markdown(f"### {article['title']}")
                                st.write(f"📰 **Source:** {article['source']['name']}")
                                st.write(f"📅 **Published At:** {article['publishedAt']}")
                                if article['urlToImage']:
                                    st.image(article['urlToImage'], use_column_width=True)
                                st.markdown(f"[🔗 Read More]({article['url']})")
                                st.markdown("---")
                                #fetch latest youtube videos ofnews
                else:
                    st.error("❌ Failed to fetch news. Please check and try again later.")
            else:
                query='agriculture news in english latest'
                videos = fetch_youtube_videos(query, max_results=10)
                for i in range(0, len(videos), 2):
                    cols = st.columns(2)
                    for j, video in enumerate(videos[i:i+2]):
                        with cols[j]:
                            st.video(f"https://www.youtube.com/watch?v={video['video_id']}")

            
        except:
            st.error("❌ Failed to fetch news. Please check and try again later.")
