import streamlit as st

def about_us_page():
    # About Us page description
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url('https://img.freepik.com/free-vector/hand-painted-watercolor-floral-background_79603-1868.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;  /* Ensure the background covers the whole screen */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.form('About Us'):
        # Contact Us Form
        st.subheader("Contact Us (మమ్మల్ని సంప్రదించండి)")
        col1,col2,col3=st.columns(3)
        # Create form fields
        name = col1.text_input("Your Name (పేరు)")
        email = col2.text_input("Your Email (ఇమెయిల్)")
        phone = col3.text_input("Your Phone Number (ఫోన్ నంబర్)")
        issue = st.text_area("Describe your issue or query (సమస్య లేదా ప్రశ్నను వివరించండి)")

        # Submit button
        if st.form_submit_button("Submit (సమర్పించండి)"):
            if name and email and phone and issue:
                # Process the form data (you can save it or send an email here)
                st.success("Thank you for reaching out! We'll get back to you soon. (చేరుకున్నందుకు ధన్యవాదాలు! మేము త్వరలో మిమ్మల్ని సంప్రదిస్తాము.)")
            else:
                st.error("Please fill in all fields before submitting. (దయచేసి సమర్పించే ముందు అన్ని ఫీల్డ్‌లను పూరించండి)")