import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Paths for model, scaler, and assets
MODEL_PATH = 'weather_model.pkl'
SCALER_PATH = 'scaler.pkl'

# Load Model and Scaler
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pkl(MODEL_PATH)
scaler = load_pkl(SCALER_PATH)

# Prediction Function
def predict_weather(input_array):
    input_array_scaled = scaler.transform(input_array)
    result = model.predict(input_array_scaled)
    prob_drizzle = round(model.predict_proba(input_array_scaled)[0][0], 2)
    prob_rain = round(model.predict_proba(input_array_scaled)[0][1], 2)
    prob_sun = round(model.predict_proba(input_array_scaled)[0][2], 2)
    prob_snow = round(model.predict_proba(input_array_scaled)[0][3], 2)
    prob_fog = round(model.predict_proba(input_array_scaled)[0][4], 2)

    results = {
        "result": int(result[0]),
        "probabilities": {
            "Drizzle": prob_drizzle,
            "Rain": prob_rain,
            "Sun": prob_sun,
            "Snow": prob_snow,
            "Fog": prob_fog
        }
    }

    return results
user_data = st.session_state.get('user', None)
lang=user_data[5]
name=user_data[1]
if lang=='telugu':
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url("https://media.istockphoto.com/id/690022274/photo/defocused-lights-background.jpg?s=612x612&w=0&k=20&c=Hq8w3rrMmXfWJNhKC9zFQr_owsk9D-cIWwLB4pJbsmk=");  
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    f"""
    <h1 style="text-align: center; color: black;">
        స్వాగతం {name} గారు 👋✨
    </h1>
    """,
    unsafe_allow_html=True
    )
    def user_profile():
        # Input Section
        with st.form(key="weather_form"):
            st.subheader("వాతావరణ పారామితులను నమోదు చేయండి")
            col1, col2 = st.columns(2)
            precipitation = col1.number_input("అవపాతం (mm)", min_value=0.0, max_value=500.0, value=10.0, step=0.1)
            temp_max = col2.number_input("గరిష్ట ఉష్ణోగ్రత (°C)", min_value=-50.0, max_value=50.0, value=20.0, step=0.1)
            temp_min = col1.number_input("కనిష్ట ఉష్ణోగ్రత (°C)", min_value=-50.0, max_value=50.0, value=10.0, step=0.1)
            wind = col2.number_input("గాలి వేగం (km/h)", min_value=0.0, max_value=150.0, value=5.0, step=0.1)
            col1,col2,col3=st.columns([12,10,10])
            submitted = col2.form_submit_button("వాతావరణాన్ని అంచనా వేయండి")
        # Predict Button
        if submitted:
            # Prepare input array
            input_array = np.array([[precipitation, temp_max, temp_min, wind]])
            try:
                prediction = predict_weather(input_array)

                # Displaying Result
                col1,col2=st.columns(2)

                weather_dict = {0: "చినుకులు", 1: "వర్షం", 2: "యెండ", 3: "మంచు", 4: "పొగమంచు"}
                predicted_weather = weather_dict[prediction["result"]]
                probabilities = prediction["probabilities"]
                categories =  ["చినుకులు","వర్షం","యెండ", "మంచు", "పొగమంచు"]
                values = list(probabilities.values())
                col1.success(f"ఊహించిన వాతావరణం: {predicted_weather}")
                #display the probabilities in bar graph
                #create a dataframe
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
                    title="వాతావరణ పరిస్థితుల భావితవ్యాలు (Probabilities)",
                    labels={'Category': 'వాతావరణం', 'Probability': 'ప్రామాణికత'}
                )

                # Display the bar chart
                col1.plotly_chart(fig)


                # Radar Chart Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Probabilities'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False
                )
                col2.plotly_chart(fig,use_container_width=True)

            except Exception as e:
                pass
    def user_home_page():
        # Navigation menu for user dashboard
        selected_tab = option_menu(
            menu_title=None,
            options=["వర్షపాతం అంచనా",'లాగ్అవుట్'],
            icons=['cloud-drizzle-fill','unlock-fill'], menu_icon="cast", default_index=0,
            orientation="horizontal",
        styles={
        "nav-link-selected": {"background-color": "#62f088", "color": "black", "border-radius": "5px"},
        }
        )
        if selected_tab == "వర్షపాతం అంచనా":
            user_profile()
        elif selected_tab=='లాగ్అవుట్':
            # Logout functionality
            st.cache()
            st.session_state.clear()  # Clear session state to "log out"
            st.experimental_rerun()
else:
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url("https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDI0LTExL3Jhd3BpeGVsb2ZmaWNlNF93aGl0ZV9hbmRfc2lsdmVyX3NpbXBsZV9wbGFpbl9ncmFkaWVudF9iYWNrZ3JvdV8xMTgwZTY5Yy0yNjczLTQ2MTItYmFhNC1jMGFiMDFiODRmYzIuanBn.jpg");  
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    f"""
    <h1 style="text-align: center; color: black;">
        Welcome {name}  👋✨
    </h1>
    """,
    unsafe_allow_html=True
    )
    def user_profile():
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

                # Displaying Result
                col1,col2=st.columns(2)

                weather_dict = {0: "Drizzle", 1: "Rain", 2: "Sun", 3: "Snow", 4: "Fog"}
                predicted_weather = weather_dict[prediction["result"]]
                probabilities = prediction["probabilities"]
                categories =  ["Drizzle","Rain","Sun", "Snow", "Fog"]
                values = list(probabilities.values())
                col1.success(f"Predicted Weather: {predicted_weather}")
                #create a dataframe
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
                col1.plotly_chart(fig)
                # Radar Chart Visualization
                categories = list(probabilities.keys())
                values = list(probabilities.values())
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Probabilities'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False
                )
                col2.plotly_chart(fig,use_container_width=True)

            except Exception as e:
                pass                    
    def user_home_page():
        # Navigation menu for user dashboard
        selected_tab = option_menu(
            menu_title=None,
            options=["Rainfall Prediction",'Logout'],
            icons=['cloud-drizzle-fill','unlock-fill'], menu_icon="cast", default_index=0,
            orientation="horizontal",
        styles={
        "nav-link-selected": {"background-color": "#62f088", "color": "black", "border-radius": "5px"},
        }
        )
        if selected_tab == "Rainfall Prediction":
            user_profile()
        elif selected_tab=='Logout':
            # Logout functionality
            st.cache()
            st.session_state.clear()
            st.experimental_rerun()