# Hugging Face Chatbot

A modern chatbot application powered by Hugging Face's language models. This project demonstrates how to integrate AI language models into a web application using Streamlit.

## Demo

Visit the live demo at: [Hugging Face Chatbot](https://huggingface-chatbot-bw5j9edduvy5ory7q7gjzj.streamlit.app/)

## Features

- Interactive chatbot with natural language understanding
- Powered by state-of-the-art language models
- User-friendly interface with conversation history
- Responsive design for all devices

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/huggingface-chatbot.git
cd huggingface-chatbot
pip install -r requirements.txt
```

## Configuration

1. Get your Hugging Face API token from [Hugging Face](https://huggingface.co/settings/tokens)
2. Create a file at `.streamlit/secrets.toml` with your token:

```toml
HUGGINGFACEHUB_API_TOKEN = "your_token_here"
```

## Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Navigate to http://localhost:8501 in your browser to interact with the chatbot.

## Implementation Approaches

This project explored two different methods for implementing an AI chatbot:

### 1. API-based Approach (Current Implementation)

The current implementation uses Hugging Face's Inference API to send requests to a hosted model.

**Advantages:**
- Minimal dependencies and simple setup
- Works reliably on most hosting platforms
- No need to manage model weights locally
- Can access larger, more powerful models
- Easier deployment and maintenance

**Disadvantages:**
- Requires internet connection
- Subject to API rate limits
- Sends data to external service
- May have higher latency due to network requests

### 2. Local Model Approach (Alternative Implementation)

We also explored running the model locally within the application.

**Advantages:**
- Complete control over model parameters
- No dependency on external services
- Better data privacy (data stays in the app)
- Can work offline once loaded
- Potentially lower latency for responses

**Disadvantages:**
- Much heavier dependencies (PyTorch, transformers, etc.)
- Requires significant computational resources
- More complex setup and deployment
- Limited to smaller models due to memory constraints
- Prone to dependency conflicts

## Technical Architecture

### API-based Implementation

1. User submits a question through the Streamlit interface
2. Application sends a request to Hugging Face's Inference API
3. The hosted model (flan-t5-base) processes the request
4. The API returns the generated response
5. Application displays the response to the user

### Local Model Implementation

1. Application loads the model during startup
2. User submits a question through the interface
3. Input is tokenized and processed locally
4. Model generates a response within the application
5. Response is displayed to the user

## Deployment

The application is deployed on Streamlit Cloud. When deploying:

1. Make sure to add your Hugging Face API token in the Streamlit Cloud secrets management
2. Set the Python version to 3.9 for best compatibility
3. Await the initial building process (may take a few minutes)

## Dependencies

- streamlit: Web application framework
- requests: For API communication
- python-dotenv: For environment variable management
- markdown: For text formatting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the language models
- [Streamlit](https://streamlit.io/) for the web application framework 