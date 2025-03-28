import streamlit as st
import requests
import json
import time
import logging
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="AI Chatbot with HuggingFace API",
    page_icon="🤖",
    layout="centered"
)

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
    /* Styles to improve button appearance */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 2.4rem;
        margin-top: 0.8rem;
    }
    /* Hide the footer made by Streamlit */
    footer {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Add a title and description
st.title("🤖 GenAI Chatbot")
st.write("""
This chatbot is powered by a Hugging Face language model (flan-t5-base). 
It demonstrates integration with state-of-the-art AI technologies.
""")

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to handle basic math operations
def calculate_expression(expression):
    # Simple math operations regex
    addition = re.match(r'^\s*(\d+)\s*\+\s*(\d+)\s*$', expression)
    subtraction = re.match(r'^\s*(\d+)\s*\-\s*(\d+)\s*$', expression)
    multiplication = re.match(r'^\s*(\d+)\s*\*\s*(\d+)\s*$', expression)
    division = re.match(r'^\s*(\d+)\s*\/\s*(\d+)\s*$', expression)
    
    if addition:
        a, b = int(addition.group(1)), int(addition.group(2))
        return f"{a} + {b} = {a + b}"
    elif subtraction:
        a, b = int(subtraction.group(1)), int(subtraction.group(2))
        return f"{a} - {b} = {a - b}"
    elif multiplication:
        a, b = int(multiplication.group(1)), int(multiplication.group(2))
        return f"{a} * {b} = {a * b}"
    elif division:
        a, b = int(division.group(1)), int(division.group(2))
        if b == 0:
            return "Division by zero is not allowed"
        return f"{a} / {b} = {a / b}"
    
    return None  # Not a simple math expression

# Function to get a response from the model
def get_ai_response(question):
    # Check for API token
    API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
    if not API_TOKEN:
        return "Error: API token not found in secrets. Please add your HUGGINGFACEHUB_API_TOKEN."
    
    # Check for math expression first
    math_result = calculate_expression(question)
    if math_result:
        return math_result
    
    # Set model parameters
    MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
    
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
    
    try:
        # Make the API request
        response = requests.post(MODEL_URL, headers=HEADERS, json=payload)
        
        # Check if model is loading
        if response.status_code == 503:
            return "The model is still loading. Please try again in a few seconds."
        
        # Check for other errors
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Extract text from response
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "Sorry, I couldn't generate a response")
        else:
            return "Sorry, I received an unexpected response format"
    
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except json.JSONDecodeError:
        return "Failed to parse API response"
    except Exception as e:
        return f"Error: {str(e)}"

# Display chat history
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

# Create the input form - simplifying the structure
user_input = st.text_input("Type your message:", key="user_message")
send_button = st.button("Send")

# Process the message when Send is clicked or Enter is pressed
if send_button and user_input:
    # Add user message to history first
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show a temporary message that the AI is thinking
    ai_thinking_placeholder = st.empty()
    ai_thinking_placeholder.markdown("""
    <div class="message-container ai-message">
        <strong>AI:</strong> <em>Thinking...</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Get AI response
    ai_response = get_ai_response(user_input)
    
    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    # Replace thinking message with the actual response
    ai_thinking_placeholder.empty()
    
    # Force a page rerun without trying to reset the text input
    st.experimental_rerun()

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

# Add information about Flan-T5-base
with st.expander("About Flan-T5-base Model"):
    st.write("""
    ### Google's Flan-T5-base
    
    Flan-T5-base is a powerful language model developed by Google. It is part of the T5 (Text-to-Text Transfer Transformer) family of models that have been fine-tuned on a mixture of tasks.
    
    **Key characteristics:**
    
    - **Size**: A medium-sized model with approximately 250 million parameters
    - **Training**: Fine-tuned with instruction-based prompts across hundreds of tasks
    - **Capabilities**: Can perform various natural language tasks like question answering, summarization, translation, and more
    - **Advantages**: Good balance between performance and computational requirements
    
    The "Flan" models represent a significant improvement over the original T5 models through enhanced instruction-tuning techniques, making them better at following instructions and generating more helpful, accurate responses.
    """) 