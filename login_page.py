import streamlit as st
from db_manager import validate_user

def login_page():
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url('https://img.freepik.com/free-vector/watercolor-green-leaves-border-with-dots_65186-3397.jpg?semt=ais_hybrid');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;  /* Ensure the background covers the whole screen */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Center the login form using Streamlit form layout
    col1,col2,col3=st.columns([2,6,2])
    with col2.form(key="login_form"):
        # Title
        st.title("Login Page (లాగిన్ చెయ్యండి)")

        # Email and Password inputs
        email = st.text_input("Email (ఇమెయిల్ ఇవ్వండి)")
        password = st.text_input("Password (పాస్వర్డ్)", type="password")

        # Submit button inside the form
        login_button = st.form_submit_button("Login (లాగిన్ చేయండి)")

        # Handling form submission
        if login_button:
            user = validate_user(email, password)
            if user:
                # Set session state to user_home and store user details
                st.session_state["page"] = "user_home"
                st.session_state["user"] = user  # Store user info (e.g., name, email)
                st.session_state["user_tab"] = "Loan Page"  # Default tab after login
                st.experimental_rerun()
            else:
                st.error("Invalid Email or Password (చెల్లని ఇమెయిల్ లేదా పాస్‌వర్డ్)!")

