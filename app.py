import os
import streamlit as st
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import base64

# Get API key from Streamlit secrets and set it in environment
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Initialize the Hugging Face model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature": 0.5, "max_length": 128}
)

# Create the prompt template and chain
prompt = PromptTemplate(
    input_variables=['question'],
    template='Question: {question}'
)
chain = LLMChain(llm=llm, prompt=prompt)

# Function to display SVG
def render_svg():
    svg_code = '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <rect width="100" height="100" rx="20" fill="#333333"/>
      <circle cx="30" cy="35" r="8" fill="#7792E3"/>
      <circle cx="70" cy="35" r="8" fill="#7792E3"/>
      <path d="M25,60 Q50,80 75,60" stroke="#7792E3" stroke-width="4" fill="transparent"/>
    </svg>
    '''
    return svg_code

# Configure the Streamlit page with dark theme
st.set_page_config(page_title="Hugging Face Chatbot", page_icon=":robot_face:", layout="centered")

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    body {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #333333;
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #555555;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin-left: 0.5rem;
    }
    .stButton > button:hover {
        background-color: #777777;
    }
    .stChatMessage {
        background-color: #2C2C2C;
        color: #FFFFFF;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-width: 100%;
        overflow-wrap: break-word;
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }
    .container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    .input-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .user-message {
        background-color: #2C2C2C;
        margin-bottom: 8px;
    }
    .bot-message {
        background-color: #383838;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and introduction
st.title("Hugging Face Chatbot")
st.markdown("Ask me anything!")

# Display SVG icon
st.markdown(f'<div class="centered">{render_svg()}</div>', unsafe_allow_html=True)

# Add welcoming text
st.markdown("<h3 style='color:#AAAAAA;'>How can I help you today?</h3>", unsafe_allow_html=True)

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Display conversation history
for i, message in enumerate(st.session_state.conversation):
    if i % 2 == 0:  # User message
        st.markdown(f"<div class='stChatMessage user-message'>{message['content']}</div>", unsafe_allow_html=True)
    else:  # Bot message
        st.markdown(f"<div class='stChatMessage bot-message'>{message['content']}</div>", unsafe_allow_html=True)

# Create a form for input
with st.form(key='my_form', clear_on_submit=True):
    # Create columns for input and button
    col1, col2 = st.columns([6, 1])
    
    # User input and button
    with col1:
        user_question = st.text_input("Your Question:", key="input_field")
    with col2:
        submit_button = st.form_submit_button("Ask")

    # Handle input submission
    if submit_button and user_question:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "You", "content": f"**You:** {user_question}"})
        
        # Generate response
        with st.spinner("Thinking..."):
            response = chain.run(question=user_question)
        
        # Add bot response to conversation
        st.session_state.conversation.append({"role": "Chatbot", "content": f"**Chatbot:** {response}"})
        
        # Rerun to update the conversation
        st.rerun() 