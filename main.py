import streamlit as st
from login_page import login_page
from register_page import register_page
from db_manager import init_db
from streamlit_option_menu import option_menu
from home_page import home_page
from about_page import about_us_page
# Initialize the database
init_db()

# Streamlit Page Config
st.set_page_config(page_title = 'Smart Irrigation', layout='wide', page_icon="🌱")


# Session State Initialization
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if st.session_state["page"] == "Home":
    # Horizontal navigation for non-logged-in users
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    <h1 style="text-align: center; color: black;">
        AI-Driven Crop Disease Prediction and Management
        (ఆధారిత పంట వ్యాధి అంచనా మరియు నిర్వహణ)
    </h1>
    """,
    unsafe_allow_html=True
)

    selected_page = option_menu(
        menu_title=None,
        options=["Home (హోమ్)", "Login (లాగిన్ అవ్వండి)", "Register (నమోదు)",'Contact (సంప్రదించండి)'],
        icons=["house", "box-arrow-in-right", "person-plus",'file-earmark-fill'],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "nav-link-selected": {
                "background-color": "green",  # Background color of the selected item
                "color": "white",
            },
            "nav-link": {
                "background-color": "#cff2ae",  # Background color of unselected items
                "color": "black",  # Text color of unselected items
            },
        },
    )

    # Render the selected page
    if selected_page == "Home (హోమ్)":
        home_page()
    elif selected_page == "Login (లాగిన్ అవ్వండి)":
        login_page()
    elif selected_page == "Register (నమోదు)":
        register_page()
    elif selected_page == 'Contact (సంప్రదించండి)':
        about_us_page()

elif st.session_state["page"] == "user_home":
    # Redirect to the user dashboard after login
    from user_home import user_home_page
    user_home_page()
