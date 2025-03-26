# Hugging Face Chatbot

A Streamlit-based chatbot powered by Hugging Face's language models. The chatbot features a modern dark theme interface and can answer questions on various topics.

## Features
- Clean, dark-themed UI
- Real-time responses using Hugging Face's language models
- Conversation history tracking
- Mobile-responsive design

## Live Demo
[Access the live demo here](https://your-app-name.streamlit.app) (Link will be available after deployment)

## Local Development Setup

1. Clone this repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face API token:
   - Get your token from [Hugging Face](https://huggingface.co/settings/tokens)
   - Create `.streamlit/secrets.toml` file
   - Add your token: `HUGGINGFACEHUB_API_TOKEN = "your-token-here"`

4. Run the application:
```bash
streamlit run app.py
```

## Deployment
This app is deployed on Streamlit Cloud. To deploy your own version:

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy your app
4. Add your Hugging Face API token in Streamlit Cloud secrets

## Technologies Used
- Streamlit
- Langchain
- Hugging Face
- Python

## License
MIT License 