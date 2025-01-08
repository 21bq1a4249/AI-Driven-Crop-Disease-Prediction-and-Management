import streamlit as st
from telugu_page import telugu_page
from english_page import english_page
def user_home_page():
    user_data = st.session_state.get('user', None)
    lang=user_data[4]
    if lang=='telugu':
        telugu_page()
    if lang=='english':
        english_page()