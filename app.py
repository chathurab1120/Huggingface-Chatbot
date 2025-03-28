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

# Dark mode UI
st.markdown("""
<style>
    body {
        color: #FFFFFF;
        background-color: #121212;
    }
    .main {
        background-color: #121212;
    }
    .stTextInput > div > div > input {
        background-color: #333333;
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #4F4F4F;
        color: #FFFFFF;
    }
    .stButton > button:hover {
        background-color: #666666;
    }
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .css-18e3th9 {
        background-color: #121212;
    }
    .css-1d391kg {
        background-color: #121212;
    }
    .st-bb {
        background-color: #333333;
    }
    .st-at {
        background-color: #333333;
    }
    .st-af {
        background-color: #121212;
    }
    p, h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    .streamlit-expanderHeader {
        color: #FFFFFF;
        background-color: #333333;
    }
    .streamlit-expanderContent {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .message-container {
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #2C2C2C;
        border-left: 5px solid #4CAF50;
    }
    .ai-message {
        background-color: #1E1E1E;
        border-left: 5px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Add a title and description
st.title("🤖 GenAI Chatbot")
st.write("""
This chatbot is powered by a Hugging Face language model (flan-t5-base). 
It demonstrates integration with state-of-the-art AI technologies.
""")

# Initialize session state for conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history with better styling
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="message-container user-message">
            <strong>You:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-container ai-message">
            <strong>AI:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)

# User input with form - note about Enter key
user_input = st.text_input("Type your message:", key="user_message", 
                          help="Press Enter to submit your message")

# Process user input - support for both Enter key and button
send_button = st.button("Send")
if (user_input and send_button) or (user_input and st.session_state.get('user_message') != user_input):
    # Store current input to check against next time
    current_input = user_input
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get AI response
    with st.spinner("AI is thinking..."):
        ai_response = get_ai_response(user_input)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    # Clear the input box by modifying session state
    st.session_state.user_message = ""
    
    # Use experimental_rerun for compatibility with older Streamlit versions
    try:
        st.experimental_rerun()
    except:
        # If experimental_rerun is not available, don't do anything
        # The page will still update with the new content
        pass

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