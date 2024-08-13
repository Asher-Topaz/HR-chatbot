import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import firebase_admin
from firebase_admin import credentials, firestore, auth
import uuid
import os

# Initialize Firebase (if not already initialized)
if not firebase_admin._apps:
    cred = credentials.Certificate('sample-authentication-697a4-firebase-adminsdk-nsvjq-5eef284dc1.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

def register_user(username, name, email, industry, occupation, password):
    try:
        user = auth.create_user(email=email, password=password, display_name=username)
        user_ref = db.collection('users').document(user.uid)
        user_ref.set({
            'username': username,
            'name': name,
            'email': email,
            'industry': industry,
            'occupation': occupation,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        return True, "Registration successful!"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def login_user(username_or_email, password):
    try:
        if '@' in username_or_email:
            email = username_or_email
            user = auth.get_user_by_email(email)
        else:
            users_ref = db.collection('users')
            query = users_ref.where('username', '==', username_or_email).limit(1)
            user_docs = query.get()
            if not user_docs:
                return False, "User not found"
            user_data = user_docs[0].to_dict()
            email = user_data['email']
            user = auth.get_user_by_email(email)

        return True, "Login successful!"
    except Exception as e:
        return False, f"Login failed: {str(e)}"

def app():
    st.markdown(
        """
        <style>
        .stButton>button { background-color: #FFA500; color: white; }
        .stTextInput>div>div>input, .stSelectbox>div>div>select { border-color: #FFA500; }
        h1, h2, h3 { color: #FFA500; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Welcome to INFOX')
    
    action = st.selectbox("Choose an action", ["Sign Up", "Log In"])

    if action == "Sign Up":
        st.markdown("### Sign Up")
        with st.form("signup_form"):
            username = st.text_input('Username')
            name = st.text_input('Full Name')
            email = st.text_input('Email Address')
            industry = st.selectbox('Industry', ['Technology', 'Healthcare', 'Finance', 'Education', 'Other'])
            occupation = st.text_input('Occupation')
            password = st.text_input('Password', type='password')
            confirm_password = st.text_input('Confirm Password', type='password')
            
            submit_button = st.form_submit_button('Sign Up')

            if submit_button:
                if username and name and email and industry and occupation and password and confirm_password:
                    if password == confirm_password:
                        success, message = register_user(username, name, email, industry, occupation, password)
                        if success:
                            st.success(message)
                            st.session_state.logged_in = True
                            st.session_state.page = "Chatbot"  # Set the page state for redirection
                            switch_page("Chatbot")  # Redirect to the Chatbot page
                        else:
                            st.error(message)
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning('Please fill in all fields')

    elif action == "Log In":
        st.markdown("### Log In")
        username_or_email = st.text_input('Username or Email')
        password = st.text_input('Password', type='password')
        
        if st.button('Log In', type="primary"):
            if username_or_email and password:
                success, message = login_user(username_or_email, password)
                if success:
                    st.success(message)
                    st.session_state.logged_in = True
                    st.session_state.page = "Chatbot"  # Set the page state for redirection
                    switch_page("Chatbot")  # Redirect to the Chatbot page
                else:
                    st.error(message)
            else:
                st.warning('Please fill in all fields')
