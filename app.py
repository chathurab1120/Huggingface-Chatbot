import streamlit as st
import requests
import json
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="AI Chatbot with HuggingFace API",
    page_icon="🤖",
    layout="centered"
)

# Get API token from Streamlit secrets
API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
if not API_TOKEN:
    st.error("API token not found. Please set HUGGINGFACEHUB_API_TOKEN in your Streamlit secrets.")
    st.stop()

# Set model parameters
MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to query the model
def query_model(payload):
    try:
        logger.info(f"Querying model with payload: {payload}")
        response = requests.post(MODEL_URL, headers=HEADERS, json=payload)
        
        # Check if the model is still loading
        if response.status_code == 503:
            logger.warning("Model is loading...")
            return {"error": "Model is loading, please try again in a few seconds."}
        
        # Check for other errors
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        logger.info(f"Response from API: {result}")
        return result
    except Exception as e:
        logger.error(f"Error querying model: {str(e)}")
        return {"error": f"Error: {str(e)}"}

# Function to get a response from the model
def get_ai_response(question):
    # Prepare payload
    payload = {
        "inputs": question,
        "parameters": {
            "max_length": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    # Query model
    result = query_model(payload)
    
    # Check for errors
    if "error" in result:
        return f"Sorry, I encountered an error: {result['error']}"
    
    # Extract response
    if isinstance(result, list) and len(result) > 0:
        # The API returns a list of generated texts
        return result[0].get("generated_text", "Sorry, I couldn't generate a response")
    else:
        return "Sorry, I received an unexpected response format"

# Add a title and description
st.title("🤖 GenAI Chatbot")
st.write("""
This chatbot is powered by a Hugging Face language model (flan-t5-base). 
It demonstrates integration with state-of-the-art AI technologies.
""")

# Initialize session state for conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask me anything...")

# Process user input
if prompt:
    # Add user message to chat
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_ai_response(prompt)
            st.markdown(response)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add information about the technology used
with st.expander("About this GenAI Demo"):
    st.write("""
    ### Technology Stack
    
    This project demonstrates the integration of modern AI technologies:
    
    - **Language Model**: Google's Flan-T5 (via Hugging Face API)
    - **Framework**: Streamlit for the interactive web interface
    - **Deployment**: Streamlit Cloud
    
    ### How It Works
    
    1. User inputs are sent to the Hugging Face API
    2. The model processes the input and generates a response
    3. The interface displays the response in a conversational format
    
    This architecture allows for showcasing GenAI capabilities without the overhead of running models locally.
    """)

# Add custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.stTextInput > div > div > input {
    background-color: #f0f0f0;
}
.css-1ec096l {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True) 