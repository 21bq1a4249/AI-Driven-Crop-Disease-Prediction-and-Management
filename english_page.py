import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
import CNN
import torch
from PIL import Image
import os
import torchvision.transforms.functional as TF
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
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
            background-image: url("https://wmo.int/sites/default/files/styles/featured_image_x1_768x512/public/2023-12/thumbnails_5.jpg?h=d1cb525d&itok=aZ4qUGTc");  
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
        with st.form(key="weather_form"):
            st.subheader("Enter Weather Parameters")
            col1, col2 = st.columns(2)
            precipitation = col1.number_input("Precipitation (mm)", min_value=0.0, max_value=500.0, value=10.0, step=0.1)
            temp_max = col2.number_input("Max Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0, step=0.1)
            temp_min = col1.number_input("Min Temperature (°C)", min_value=-50.0, max_value=50.0, value=10.0, step=0.1)
            wind = col2.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=5.0, step=0.1)
            col1,col2,col3=st.columns([12,5,10])
            submitted = col2.form_submit_button("Predict Weather")
        # Predict Button
        if submitted:
            # Prepare input array
            input_array = np.array([[precipitation, temp_max, temp_min, wind]])                
            try:
                prediction = predict_weather(input_array)
                col1,col2=st.columns([10,5])
                weather_dict = {0: "Drizzle", 1: "Rain", 2: "Sun", 3: "Snow", 4: "Fog"}
                predicted_weather = weather_dict[prediction["result"]]
                col1.markdown(
                    f"""
                    <div style="text-align: center; padding: 5px; background-color: #42f55d; border-radius: 1px; border: 2px solid black; margin-bottom: 2px;">
                        <p style="color: black; font-size: 20px;"><b>Predicted Weather: {predicted_weather}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # Extract probabilities
                categories = ["Drizzle", 'Rain', 'Sun', 'Snow', 'Fog']
                values = [
                    prediction["drizzle"],
                    prediction["fog"],
                    prediction["rain"],
                    prediction["snow"],
                    prediction["sun"]
                ]

                # Create a dataframe
                df = pd.DataFrame({
                    'Category': categories,
                    'Probability': values
                })

                colors = ['#FF5733', '#33FF57', '#3357FF', '#F4D03F', '#8E44AD']
                # Create a Plotly bar chart
                fig = px.bar(
                    df,
                    x='Category',
                    y='Probability',
                    color='Category',
                    color_discrete_sequence=colors,
                    title="Weather Probabilities",
                    labels={'Category': 'Weather', 'Probability': 'Probability'}
                )
                # Display the bar chart
                col1.plotly_chart(fig,use_container_width=True)

                # Radar Chart Visualization
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Probabilities'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                )

                # Display the radar chart
                col2.plotly_chart(fig_radar,use_container_width=True)

            except Exception as e:
                st.write(e)  
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
        options=["Rainfall Prediction",'Crop Diseases','Logout'],
        icons=['cloud-drizzle-fill','prescription2','unlock-fill'], menu_icon="cast", default_index=0,
        orientation="horizontal",
    styles={
    "nav-link-selected": {"background-color": "#62f088", "color": "black", "border-radius": "5px"},
    }
    )
    if selected_tab == "Rainfall Prediction":
        user_profile()
    elif selected_tab=='Crop Diseases':
        diseases()
    elif selected_tab=='Logout':
        # Logout functionality
        st.cache()
        st.session_state.clear()
        st.experimental_rerun()