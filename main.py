import streamlit as st
st.set_page_config(page_title="INFOX")
from streamlit_option_menu import option_menu
import home, about, register

# Define a function to handle the page selection
def display_page(page_name):
    if page_name == "Home":
        home.app()
    elif page_name == "About":
        about.app()
    elif page_name == "Account":
        register.app()
    elif page_name == "Chatbot":
        st.write("Chatbot page")  # Placeholder for the chatbot page function

# Sidebar for navigation
with st.sidebar:
    if st.session_state.get("logged_in", False):
        options = ["Home", "About", "Chatbot", "Account", "Logout"]
    else:
        options = ["Home", "About", "Account"]

    selected = option_menu(
        menu_title=None,
        options=options,
        icons=["house-fill", "info-circle-fill", "chat-dots-fill", "person-circle", "box-arrow-right"] if "Logout" in options else ["house-fill", "info-circle-fill", "person-circle"],
        menu_icon="menu",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding-top": "0.5rem", "padding-right": "0.5rem", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {
                "font-size": "18px",
                "text-align": "center",
                "color": "white",
                "padding": "10px",
                "margin": "0px",
                "--hover-color": "#87CEFA",
            },
            "nav-link:hover": {
                "background-color": "#87CEFA",
            },
            "nav-link-selected": {"background-color": "#FFA500"},
        },
    )

    if selected == "Logout":
        st.session_state.logged_in = False
        st.experimental_rerun()  # Rerun to reflect changes

# Display the selected page
display_page(selected)
