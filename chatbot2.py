import os
import streamlit as st
import requests
import time
import firebase_admin
from firebase_admin import credentials, firestore
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
from langchain.chains.summarize import load_summarize_chain
from openai import OpenAIError
from datetime import datetime

# Load environment variables
load_dotenv()

# Getting API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Firebase
cred = credentials.Certificate("sample-authentication-697a4-firebase-adminsdk-nsvjq-5eef284dc1.json")
#firebase_admin.initialize_app(cred)
db = firestore.client()

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

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Reduced chunk size
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

# Summarize context if too long
def summarize_context(conversation):
    if len(conversation.memory.chat_memory.messages) > 3:  # Adjust as needed
        summarizer = load_summarize_chain(conversation.llm)
        context_summary = summarizer.invoke(conversation.memory.chat_memory.messages)
        conversation.memory.chat_memory.clear()
        conversation.memory.save_context({"input": "Context summary"}, {"output": context_summary['output_text']})

@st.cache_resource
def initialize_conversation():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.45)  # Use the 16k model
    doc_db = embedding_db()
    
    # Limit memory to the last 2 interactions
    memory = ConversationBufferMemory(memory_key="history", max_length=2)
    
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
def retrieval_answer(query, country,state,use_case):
    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            # Construct a more detailed query
            detailed_query = f"For a {use_case} in {state}, {country}: {query}"
            # Trim user query to fit within a reasonable length
            trimmed_query = query[:500]  # Reduce from 1000 to 500

            # Summarize context if needed
            summarize_context(conversation)

            # Retrieve relevant conversation history (limit to last N messages)
            relevant_history = conversation.memory.chat_memory.messages[-2:]  # Get last 2 messages

            # Create a prompt with trimmed context
            context = " ".join([msg.content for msg in relevant_history])
            if len(context) > 1000:  # Reduce max context length
                context = context[:1000]  # Truncate context to a shorter length

            # Create the retrieval chain
            retriever = doc_db.as_retriever(search_kwargs={"k": 2})  # Limit to 2 most relevant documents
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Get result from the chain
            result = rag_chain.invoke({"input": trimmed_query, "context": context})

            print("Result:", result)
            answer = result["answer"]

            # Truncate the answer if it's too long
            if len(answer) > 1000:
                answer = answer[:1000] + "... (truncated)"

            # Save context and answer
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

    # Title with orange color
    st.markdown(
        "<h1 style='color: orange;'>INFOX</h1>",
        unsafe_allow_html=True,
    )

    # Initialize session state if not already done
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "country" not in st.session_state:
        st.session_state.country = None
    if "state" not in st.session_state:
        st.session_state.state = None
    if "use_case" not in st.session_state:
        st.session_state.use_case = None

    # Sidebar for session management
    st.sidebar.header("Chat Sessions")
    
    # Button to start a new chat session
    if st.sidebar.button("New Chat Session"):
        st.session_state.chat_id = f"session_{datetime.utcnow().isoformat()}"
        st.session_state.history = []
        st.sidebar.success("Started a new chat session.")
        st.experimental_rerun()  # Force a rerun to refresh the page and clear history

    # Load existing sessions
    if st.session_state.get("chat_id"):
        chat_id = st.session_state["chat_id"]
        doc_ref = db.collection("chats").document(chat_id)
        doc = doc_ref.get()
        if doc.exists:
            st.session_state.history = doc.to_dict().get("history", [])

    # Display chat session list
    sessions = db.collection("chats").stream()
    session_list = [(doc.id, doc.to_dict().get("created_at", "")) for doc in sessions]
    if session_list:
        selected_session_id = st.sidebar.selectbox(
            "Select a session",
            [f"{session[1]} - {session[0]}" for session in session_list] + ["Start New Session"],
            format_func=lambda x: x if x != "Start New Session" else "Start New Session"
        )
        if selected_session_id == "Start New Session":
            st.session_state.chat_id = f"session_{datetime.utcnow().isoformat()}"
            st.session_state.history = []
            st.sidebar.success("Started a new chat session.")
            st.experimental_rerun()  # Force a rerun to refresh the page and clear history
        else:
            selected_session_id = selected_session_id.split(" - ")[1]
            st.session_state.chat_id = selected_session_id
            doc_ref = db.collection("chats").document(st.session_state.chat_id)
            doc = doc_ref.get()
            if doc.exists:
                st.session_state.history = doc.to_dict().get("history", [])

    # Sidebar for dropdowns and input fields
    with st.sidebar:
        # Dropdown for country selection
        countries = fetch_countries()
        selected_country = st.selectbox(
            "What country would you like to inquire about?",
            countries,
            key="country"
        )

        # Dropdown for state/province selection
        if selected_country:
            states = fetch_states(selected_country)
            selected_state = st.selectbox(
                "Perfect, which specific city do you need help with?",
                states,
                key="state"
            )

        # Business size and sector input
        if selected_country and selected_state:
            use_case = st.text_input("What size is your business and what sector?")
            if use_case:
                st.session_state.use_case = use_case

    # Main section for displaying chat history and input query
    if st.session_state.history:
        for entry in st.session_state.history:
            if entry.get("role") == "bot":
                st.markdown(
                    f"""
                    <div style="background-color: transparent; padding: 10px;">
                        <p style="color: darkyellow; font-weight: bold;">Chatbot:</p>
                        <p style="color: white;">{entry.get("message")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: grey; padding: 10px; border-radius: 5px;">
                        <p style="color: darkpurple; font-weight: bold;">User:</p>
                        <p style="color: white;">{entry.get("message")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Question search bar
    query = st.text_input("Ask a question:")
    if query:
        answer = retrieval_answer(query)
        st.session_state.history.append({"role": "user", "message": query})
        st.session_state.history.append({"role": "bot", "message": answer})
        st.markdown(
            f"""
            <div style="background-color: transparent; padding: 10px;">
                <p style="color: darkyellow; font-weight: bold;">Chatbot:</p>
                <p style="color: white;">{answer}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Save conversation to Firebase
    if st.session_state.get("chat_id"):
        db.collection("chats").document(st.session_state.chat_id).set(
            {"history": st.session_state.history, "created_at": datetime.utcnow().isoformat()}
        )

if __name__ == "__main__":
    app()
