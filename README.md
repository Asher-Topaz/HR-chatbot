# Chatbot using openai,streamlit,pinecone and langchain
Overview

This code is for an HR chatbot application built using Streamlit, LangChain, OpenAI, and Pinecone.
The chatbot processes user queries related to HR information specific to a selected country and state/province, 
retrieves relevant information from pre-processed JSON files, and returns concise, professional responses. 
The chatbot also cites sources for its answers.

## Dependencies ##
The following Python packages are required to run the code:

> os
> Streamlit
> OpenAI
> Langchain---> Langchain community
> dotenv
> Pinecone
> requests
> Firbebase

## Setup ##

Environment Variables

This code requires API keys for pinecone and OpenAI which are loaded from environment variables
using the dotenv library

> PINECONE_API_KEY
> PINECONE_ENV
> OPENAI_API_KEY 

## Key Functions ##

1. doc_preprocessing
> Purpose: Pre-processes the JSON files by loading them and splitting their content into chunks.
Details:
Uses DirectoryLoader to load all JSON files from the data/ directory.
Splits the documents into chunks of 1000 characters using CharacterTextSplitter.
Returns the processed documents.

2. embedding.db()
> Purpose: Creates and/or connects to a Pinecone vector database to store document embeddings
Details:
Checks if an index named hr-chatbot-001 exists; if not, it creates one.
Embeds the pre-processed documents using OpenAI embeddings.
Stores the embeddings in the Pinecone database.
Caching: Uses @st.cache_resource to cache the database connection for efficiency.

3. initialize_conversation()
> Purpose: Initializes the conversation chain and memory buffer for the chatbot.
Details:
Creates an instance of ChatOpenAI with a set temperature for generating creative responses.
Initializes a ConversationBufferMemory to keep track of the conversation context.
Returns the conversation chain and the document database.
Caching: Uses @st.cache_resource to cache the conversation setup.

4. fetch_countries() and fetch_states(country)
> Purpose: Fetches a list of countries and states/provinces using an external API.
Details:
fetch_countries() retrieves and returns a list of country names.
fetch_states(country) retrieves and returns a list of states/provinces for a selected country.
Caching: Uses @st.cache_data to cache API responses.

5. retrieval_answer(query)
> Purpose: Retrieves answers to user queries using the document database and the conversation chain.
Details:
Combines the conversation context with the user query.
Uses the LangChain create_retrieval_chain function to find the most relevant documents and generate a response.
Stores the conversation in memory for context tracking.
Returns the chatbot's answer.

## Running the application ##
Streamlit run "filename.py"






