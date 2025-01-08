import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

def english_page():
    # Paths for model, scaler, and assets
    MODEL_PATH = 'weather_model.pkl'
    SCALER_PATH = 'scaler.pkl'
    model = joblib.load('weather_model.pkl')
    # Load the scaler
    scaler = joblib.load('scaler.pkl')


    # Prediction Function
    def predict_weather(input_array):
        prediction_proba = model.predict_proba(input_array)
        weather_labels = scaler.inverse_transform(model.classes_)
        output = dict(zip(weather_labels, prediction_proba[0]))
        output = {k: v for k, v in output.items()}
        output["result"] = np.argmax(prediction_proba)
        return output
    user_data = st.session_state.get('user', None)
    lang=user_data[5]
    name=user_data[1]

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
                categories = ["Drizzle", "Fog", "Rain", "Snow", "Sun"]
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
                pass                    
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