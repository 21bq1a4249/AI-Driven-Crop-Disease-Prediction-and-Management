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
    col1,col2,col3=st.columns([1,2,1])
    col2.image("https://cdni.iconscout.com/illustration/premium/thumb/farmer-planting-seed-with-wheelbarrow-illustration-download-in-svg-png-gif-file-formats--agriculture-farm-farming-man-farmers-pack-illustrations-3491186.png?f=webp",use_column_width=True)
    