import streamlit as st

def app():
    st.markdown(
        """
        <style>
        .about-container {
            display: flex;
            justify-content: space-around;
            align-items: flex-start;
            flex-wrap: wrap;
            padding: 20px;
        }
        .about-section {
            width: 30%;
            min-width: 250px;
            text-align: center;
            margin-bottom: 20px;
        }
        .about-section h2 {
            color: #FFA500;
            margin-bottom: 10px;
        }
        .about-section p {
            font-size: 14px;
        }
        </style>
        
        <div class="about-container">
            <div class="about-section">
                <h2>Our Mission</h2>
                <p>To simplify and streamline HR processes through innovative AI solutions, empowering organizations to focus on their most valuable asset - their people.</p>
            </div>
            <div class="about-section">
                <h2>Our Vision</h2>
                <p>To revolutionize the HR landscape by providing cutting-edge AI tools that enhance efficiency, promote fairness, and foster a more engaged and productive workforce.</p>
            </div>
            <div class="about-section">
                <h2>Our Values</h2>
                <p>Innovation, Integrity, Empathy, Excellence, Collaboration</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style="text-align: center; color: #FFA500;">About INFOX</h2>
        <p style="text-align: center;">
        INFOX is an AI-powered HR assistant designed to revolutionize how organizations manage their human resources. 
        By leveraging advanced natural language processing and machine learning algorithms, INFOX provides instant, 
        accurate responses to a wide range of HR-related queries, streamlining processes and enhancing decision-making.
        </p>
        """,
        unsafe_allow_html=True
    )