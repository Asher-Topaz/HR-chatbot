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
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from openai import OpenAIError

# Load environment variables
load_dotenv()

# Getting API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

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

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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

def summarize_context(conversation):
    if len(conversation.memory.chat_memory.messages) > 3:
        summarizer = load_summarize_chain(conversation.llm)
        
        docs = [
            Document(page_content=msg.content)
            for msg in conversation.memory.chat_memory.messages
        ]
        
        context_summary = summarizer.invoke(docs)
        conversation.memory.chat_memory.clear()
        conversation.memory.save_context({"input": "Context summary"}, {"output": context_summary['output_text']})

@st.cache_resource
def initialize_conversation():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.45)
    doc_db = embedding_db()
    
    memory = ConversationBufferMemory(memory_key="history", max_length=2)
    
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    return conversation, doc_db

conversation, doc_db = initialize_conversation()

# Define the prompt template for the HR chatbot
system_prompt = (
    "You are an HR chatbot providing location-specific business information."
    "Cite your sources with clickable URLs. Keep answers professional and concise. "
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
def retrieval_answer(query, country, state, use_case):
    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            detailed_query = f"For a {use_case} in {state}, {country}: {query}"
            trimmed_query = detailed_query[:500]

            summarize_context(conversation)

            relevant_history = conversation.memory.chat_memory.messages[-2:]

            context = " ".join([msg.content for msg in relevant_history])
            if len(context) > 1000:
                context = context[:1000]

            retriever = doc_db.as_retriever(search_kwargs={"k": 2})
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            result = rag_chain.invoke({"input": trimmed_query, "context": context})

            print("Result:", result)
            answer = result["answer"]

            if len(answer) > 2000:
                answer = answer[:2000] + "... (truncated)"

            conversation.memory.save_context({"input": detailed_query}, {"output": answer})

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
    if "history" not in st.session_state:
        st.session_state.history = []
    if "country" not in st.session_state:
        st.session_state.country = None
    if "state" not in st.session_state:
        st.session_state.state = None
    if "use_case" not in st.session_state:
        st.session_state.use_case = ""
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""

    # Move country, city, and use case inputs to the sidebar
    st.sidebar.header("Query Parameters")
    st.session_state.country = st.sidebar.selectbox(
        "Country",
        fetch_countries(),
        key="country_select",
        index=fetch_countries().index(st.session_state.country) if st.session_state.country else 0
    )

    if st.session_state.country:
        states = fetch_states(st.session_state.country)
        st.session_state.state = st.sidebar.selectbox(
            "City",
            states,
            key="state_select",
            index=states.index(st.session_state.state) if st.session_state.state in states else 0
        )

    st.session_state.use_case = st.sidebar.text_input("Business size and sector", value=st.session_state.use_case)

    # Chat window display
    for entry in st.session_state.history:
        if 'user' in entry and 'bot' in entry:
            if entry['user'] and entry['bot']:
                st.markdown(
                    f"<div style='border-radius: 8px; background-color: #333; color: white; padding: 10px; margin-bottom: 5px; float: right; width: fit-content; max-width: 70%;'>"
                    f"<b>User:</b> {entry['user']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='border-radius: 8px; background-color: black; color: white; padding: 10px; margin-bottom: 5px; width: fit-content; max-width: 70%;'>"
                    f"<b>Bot:</b> {entry['bot']}</div>",
                    unsafe_allow_html=True,
                )

    # Query input and submission
    st.session_state.user_query = st.text_input("Enter your query here...", value=st.session_state.user_query)

    if st.button("Submit"):
        if st.session_state.user_query and st.session_state.country and st.session_state.state and st.session_state.use_case:
            with st.spinner("Processing..."):
                response = retrieval_answer(st.session_state.user_query, st.session_state.country, st.session_state.state, st.session_state.use_case)
            st.session_state.history.append({"user": st.session_state.user_query, "bot": response})
            st.session_state.user_query = ""  # Clear the input after submission
            st.experimental_rerun()  # Force a rerun to update the display
        else:
            st.warning("Please fill in all required fields (query, country, city, and business details).")

# Execute the app function to start the chatbot
if __name__ == "__main__":
    app()