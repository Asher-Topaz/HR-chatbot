import streamlit as st
import firebase_admin

from firebase_admin import credentials
from firebase_admin import auth

import os
import sys

cred = credentials.Certificate('sample-authentication-697a4-firebase-adminsdk-nsvjq-5eef284dc1.json')
firebase_admin.initialize_app(cred)





def app():

    st.title('Welcome to :violet[INFOX] ')
    #choice = st.selectbox('Login/SignUp',['Login','Sign Up'])

    if 'username' not in st.session_state:
        st.session_state.username = ''

    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''
    

    def f():
        try:
            user = auth.get_user_by_email(email)

            st.write('Login successful')

            st.session_state.username = user.uid
            st.session_state.useremail = user.email


        except:
            st.warning('Login Failed')

    if 'signedout' not in st.session_state:
        st.session_state.signedout = False
    if 'signout' not in st.session_state:
        st.session_state.signout = False

        

    if not st.session_state['signedout']:
        choice = st.selectbox('Login/SignUp',['Login', 'SignUp'])

        if choice == 'Login':
            email= st.text_input('Email Adress')
            password=st.text_input('Password',type = 'password')

            st.button('Login',on_click=f)

        else:
            email= st.text_input('Email Adress')
            password=st.text_input('Password',type = 'password')
            username = st.text_input("Enter your unique username")

            if st.button('Create my account'):
                user = auth.create_user(email = email, password=password, uid = username)
                st.success('Account created successfully')
                st.markdown('Please Log in using your email and password')
                st.balloons()

        if 'username' in st.session_state and 'useremail' in st.session_state:
            st.write('Redirecting to the chatbot page...')
            os.system(f'streamlit run test.py')
            sys.exit()






    

    
                          
                          
