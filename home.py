import streamlit as st

def app():
    st.markdown(
        """
        <style>
        .main-content {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: center;
            height: 100vh;
            text-align: center;
            color: white;
            margin: 0;
            padding: 0;
        }
        .main-content h1 {
            font-size: 5rem;
            margin: 0;
            white-space: nowrap;
            padding-top: 1rem;
        }
        .main-content h2 {
            font-size: 2.5rem;
            margin: 0.5rem 0;
            font-weight: bold;
        }
        .main-content p {
            font-size: 1.5rem;
            margin: 1rem 0;
            max-width: 800px;
            line-height: 1.5;
        }
        </style>
        <div class="main-content">
            <div>
                <h1>Welcome to <span style="color: #FFA500;">INFOX</span></h1>
                <h2>Your HR wing's New MVP</h2>
                <p>Our chatbot is designed to quickly answer your HR related questions, saving you time.
                By using this tool, you can instantly access the information and answers you need 
                without the hassle of extensive research or processes.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )