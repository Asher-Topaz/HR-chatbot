from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import ConversationChain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.chains.conversation.memory import ConversationBufferMemory
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Getting API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Document preprocessing function
def doc_preprocessing():
    # Load PDF files
    pdf_loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',
        show_progress=True
    )
    pdf_docs = pdf_loader.load()

    # Load JSON files
    json_loader = DirectoryLoader(
        'data/',
        glob='**/*.json',
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": ".content", "text_content": False},
        show_progress=True
    )
    json_docs = json_loader.load()

    # Combine PDF and JSON documents
    docs = pdf_docs + json_docs

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
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
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    embeddings = OpenAIEmbeddings()
    docs_split = doc_preprocessing()

    doc_db = LangchainPinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name=index_name
    )
    return doc_db

# Query answer retrieval function
@st.cache_resource
def initialize_conversation():
    llm = ChatOpenAI()
    doc_db = embedding_db()
    memory = ConversationBufferMemory()

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    return conversation, doc_db

conversation, doc_db = initialize_conversation()

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
        llm=conversation.llm, 
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    result = qa.run(query)
    # Store the conversation in memory
    conversation.memory.save_context({"input": query}, {"output": result})
    return result

# Main function for Streamlit app
def main():
    st.title("INFOXBOT")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "responses" not in st.session_state:
        st.session_state.responses = []

    predefined_questions = [
        "What country would you like to inquire about?",
        "Perfect, which specific city do you need help with?",
        "What is your use case?"
    ]

    # Display historical conversation above input form
    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            st.write("**User:** " + entry["input"])
            st.write("**Chatbot:** " + entry["output"])
           
            
    # Handle predefined questions
    if st.session_state.step < len(predefined_questions):
        question = predefined_questions[st.session_state.step]
        st.write("**Chatbot:** " + question)
        user_input = st.text_input("Your response", key="user_input")

        if st.button("Next"):
            if user_input:
                st.session_state.history.append({"input": user_input, "output": question})
                st.session_state.responses.append({"question": question, "answer": user_input})
                st.session_state.step += 1
                st.experimental_rerun()
    else:
        # After predefined questions, allow free-form queries
        text_input = st.text_input("Ask your query...")
        if st.button("Ask Query"):
            if len(text_input) > 0:
                # Combine predefined questions and responses with the user's query
                context = " ".join([f"{resp['question']} {resp['answer']}" for resp in st.session_state.responses])
                combined_query = f"{context} {text_input}"
                st.session_state.history.append({"input": combined_query, "output": "Thinking..."})
                st.info("Your Query: " + combined_query)
                answer = retrieval_answer(combined_query)
                st.session_state.history[-1]["output"] = answer
                st.success(answer)

if __name__ == "__main__":
    main()
