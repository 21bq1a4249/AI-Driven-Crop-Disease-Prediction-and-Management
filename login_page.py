import streamlit as st
from db_manager import validate_user,update_otp,fetch_otp
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random
def send_alert_email(to_email, subject, message, from_email, from_password):
    # Set up the SMTP server
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    
    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        # Connect to the server and send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        pass

def login_page():
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url('https://www.vaniperen.com/wp-content/uploads/2021/12/Plants-for-Plants-header-young-plant-fruit-growing-1440x710.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;  /* Ensure the background covers the whole screen */
            background-color: rgba(255, 255, 255, 0.6); /* Add a semi-transparent overlay */
            background-blend-mode: overlay; /* Blend the image with the overlay */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Center the login form using Streamlit form layout
    col, col2, col3 = st.columns([2, 6, 2])
    with col2:
        if "otp_sent" not in st.session_state:
            st.session_state["otp_sent"] = False
        if "email" not in st.session_state:
            st.session_state["email"] = None
        if "otp_verified" not in st.session_state:
            st.session_state["otp_verified"] = False

        if not st.session_state["otp_sent"]:  # Login Form
            with st.form(key="login_form"):
                st.title("Login Page (లాగిన్ చెయ్యండి)")
                email = st.text_input("Email (ఇమెయిల్ ఇవ్వండి)")
                password = st.text_input("Password (పాస్వర్డ్)", type="password")
                login_button = st.form_submit_button("Login (లాగిన్ చేయండి)", type='primary')
                if login_button:
                    user = validate_user(email, password)
                    if user:
                        otp = random.randint(1000, 9999)
                        to_email=email
                        subject = "OTP for Crop Disease Detection Page"
                        body = f"Hello,\n\nYour OTP is {otp}. Please enter this OTP to login.\n\nRegards,\nTeam Disease Detection Team"
                        from_email = 'cropdisesedetection@gmail.com'
                        from_password = 'tnslborsclbxqgge'  
                        # Send the alert email
                        send_alert_email(to_email, subject, body, from_email, from_password)
                        update_otp(email, otp)
                        st.success("OTP Sent to your email. Please enter the OTP to login (మీ ఇమెయిల్‌కు OTP పంపబడింది. లాగిన్ అవ్వడానికి దయచేసి OTPని నమోదు చేయండి).")
                        st.session_state["otp_sent"] = True
                        st.session_state["email"] = email
                        st.session_state["user"] = user
                        st.experimental_rerun()  # Rerun script to show OTP form
                    else:
                        st.error("Invalid Email or Password!")

        else:  # OTP Verification Form
            with st.form(key="otp_form"): 
                st.title("Enter OTP (OTPని నమోదు చేయండి)")
                otp_input = st.text_input("Enter OTP (ఇక్కడ నమోదు చేయండి)")
                submit_button = st.form_submit_button("Submit (సమర్పించండి)", type='primary')
                stored_otp = fetch_otp(st.session_state["email"])[0]
                if submit_button and otp_input:
                    if int(otp_input) == int(stored_otp):
                        st.success("Login Successful! (లాగిన్ విజయవంతమైంది)")
                        st.session_state["otp_verified"] = True
                        st.session_state["page"] = "user_home"
                        st.experimental_rerun()
                    else:
                        st.error("Invalid OTP! Try again. (చెల్లని OTP! మళ్ళీ ప్రయత్నించండి)")

        # Redirect after OTP verification
        if st.session_state.get("otp_verified"):
            st.experimental_rerun()