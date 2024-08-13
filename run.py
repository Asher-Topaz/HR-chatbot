import os
import streamlit as st
import requests
import time
from functools import wraps
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.prompts.chat import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from openai import OpenAIError 

# Load environment variables
load_dotenv()

# Getting API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Rate limiter decorator
def rate_limit(max_per_minute):
    min_interval = 60.0 / max_per_minute
    def decorator(func):
        last_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Document preprocessing function
def doc_preprocessing():
    json_loader = DirectoryLoader(
        'data/',
        glob='**/*.json',
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": ".content", "text_content": False},
        show_progress=True
    )
    json_docs = json_loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(json_docs)
    return docs_split

# Embedding database function
@st.cache_resource
def embedding_db():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'hr-chatbot-001'
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    embeddings = OpenAIEmbeddings()
    docs_split = doc_preprocessing()

    doc_db = LangchainPinecone.from_documents(docs_split, embeddings, index_name=index_name)
    return doc_db

# Query answer retrieval function
@st.cache_resource
def initialize_conversation():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.45)  # Changed to gpt-3.5-turbo
    doc_db = embedding_db()
    memory = ConversationBufferMemory()

    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    return conversation, doc_db

conversation, doc_db = initialize_conversation()

# Define the prompt template for the HR chatbot
system_prompt = (
    "You are an HR chatbot providing location-specific business information. "
    "Cite sources with URLs. Keep answers professional and concise. "
    "Use the following context to answer the question: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(conversation.llm, prompt)

# Cache for 10 minutes
@rate_limit(max_per_minute=10)
@st.cache_data(ttl=600)
def retrieval_answer(query):
    max_retries = 5
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            rag_chain = create_retrieval_chain(doc_db.as_retriever(), question_answer_chain)
            result = rag_chain.invoke({"input": query})
            answer = result["answer"]
            conversation.memory.save_context({"input": query}, {"output": answer})
            return answer
        except OpenAIError as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(retry_delay)
            retry_delay *= 2

# Function to fetch countries from API
@st.cache_data
def fetch_countries():
    url = "https://countriesnow.space/api/v0.1/countries/states"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return [country['name'] for country in data['data']]
    return []

# Function to fetch states/provinces for a selected country
@st.cache_data
def fetch_states(country):
    url = "https://countriesnow.space/api/v0.1/countries/states"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for country_data in data['data']:
            if country_data['name'] == country:
                return [state['name'] for state in country_data['states']]
    return []

def app():
    if not st.session_state.get("logged_in", False):
        st.warning("Please log in to access the chatbot.")
        st.stop()

    st.title("INFOXBOT")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "country" not in st.session_state:
        st.session_state.country = None
    if "state" not in st.session_state:
        st.session_state.state = None
    if "use_case" not in st.session_state:
        st.session_state.use_case = None

    # Display historical conversation
    if st.session_state.history:
        for entry in st.session_state.history:
            if entry.get("role") == "bot":
                st.markdown(
                    f"""
                    <div style="background-color: #FFA500; border-radius: 10px; padding: 10px; margin-bottom: 5px; color: white;">
                        <img src="https://via.placeholder.com/40?text=B" style="width: 40px; float: left; margin-right: 10px;">
                        <strong>Chatbot:</strong>
                        <p>{entry["input"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
            elif entry.get("role") == "user":
                st.markdown(
                    f"""
                    <div style="background-color: #4B0082; border-radius: 10px; padding: 10px; margin-bottom: 5px; color: white;">
                        <img src="https://via.placeholder.com/40?text=U" style="width: 40px; float: left; margin-right: 10px;">
                        <strong>User:</strong>
                        <p>{entry["input"]}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Dropdown for country selection
    countries = fetch_countries()
    selected_country = st.selectbox("Select a country", [""] + countries, index=0 if st.session_state.country is None else countries.index(st.session_state.country) + 1)
    
    if selected_country:
        st.session_state.country = selected_country
        
        # Dropdown for state/province selection
        states = fetch_states(selected_country)
        selected_state = st.selectbox("Select a state/province", [""] + states, index=0 if st.session_state.state is None else states.index(st.session_state.state) + 1)
        
        if selected_state:
            st.session_state.state = selected_state
            
            # Dropdown for use case (only Grants available)
            use_case = st.selectbox("Select use case", ["", "Grants"], index=0 if st.session_state.use_case is None else ["", "Grants"].index(st.session_state.use_case))
            
            if use_case:
                st.session_state.use_case = use_case
                
                # Allow user to ask queries
                text_input = st.text_input("Ask your query...")
                if st.button("Ask Query"):
                    if len(text_input) > 0:
                        context = f"Country: {st.session_state.country}, State/Province: {st.session_state.state}, Use Case: {st.session_state.use_case}"
                        combined_query = f"{context} {text_input}"
                        st.session_state.history.append({"input": combined_query, "role": "user"})
                        st.info("Your Query: " + combined_query)
                        try:
                            answer = retrieval_answer(combined_query)
                            st.session_state.history.append({"input": answer, "role": "bot"})
                            st.success(answer)
                        except OpenAIError:
                            st.error("We're experiencing high demand. Please try again in a few moments.")

# If this file is run directly, execute the app function
if __name__ == "__main__":
    app()
