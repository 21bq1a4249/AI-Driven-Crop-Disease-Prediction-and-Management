import streamlit as st
import re
from db_manager import register_user

def register_page():
    # Center the registration form container using Streamlit form layout
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url('https://static.vecteezy.com/system/resources/thumbnails/007/566/388/small_2x/doodle-line-art-rose-flower-bouquet-on-paper-background-free-vector.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;  /* Ensure the background covers the whole screen */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    col1,col2,col3=st.columns([1,20,1])
    # Center the login form using Streamlit form layout
    with col2.form(key="register_form"):
        # Title
        st.title("Registration Form (దయచేసి నమోదు చేయండి)")
        # Form Fields
        col1, col2 = st.columns(2)
        name = col1.text_input("Name (పేరు)")
        email = col2.text_input("Email (ఇమెయిల్)")
        number = col1.text_input("Phone Number (ఫోన్ నంబర్)")
        language = col2.selectbox("Preferred Language (మీ ఇష్టమైన భాష)", ["English (ఇంగ్లీష్)", "Telugu (తెలుగు)"])
        if language == "English (ఇంగ్లీష్)":
            lang = "english"
        else:
            lang = "telugu"
        col1, col2 = st.columns(2)
        password = col1.text_input("Password (పాస్వర్డ్)", type="password")
        retype_password = col2.text_input("Retype Password (పాస్‌వర్డ్‌ని మళ్లీ టైప్ చేయండి)", type="password")

        # Submit Button inside the form
        register_button = st.form_submit_button("Register (నమోదు చేసుకోండి)")

        # Handling form submission
        if register_button:
            # Validate email using regex
            email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
            if not re.match(email_regex, email):
                st.error("Invalid Email (చెల్లని ఇమెయిల్)!")
                
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long (పాస్‌వర్డ్ తప్పనిసరిగా కనీసం 6 అక్షరాల పొడవు ఉండాలి)!")
            elif password != retype_password:
                st.error("Passwords do not match (పాస్‌వర్డ్‌లు సరిపోలడం లేదు)!")
            else:
                if register_user(name, email, number, lang, password):
                    st.success("Registration Successful (నమోదు విజయవంతమైంది)!")
                else:
                    st.error("Email already exists (మీరు ఇప్పటికే లాగిన్ అయ్యారు)!")  