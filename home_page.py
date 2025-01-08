import streamlit as st

def home_page():
    
    # Center the image
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url('https://images.smiletemplates.com/uploads/screenshots/480/0000480955/powerpoint-template-450w.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 255, 255, 0.3); /* Add a semi-transparent overlay */
            background-blend-mode: overlay; /* Blend the image with the overlay */

        }
        </style>
        """,
        unsafe_allow_html=True
    )    
