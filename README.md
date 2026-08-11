<p align="center">
  <img src="ai-agent-banner.png" alt="Python AI Agent Banner" width="100%">
</p>

# Python AI Agent

A Python-based AI assistant with persistent memory, web tools, file operations, safe calculations, and OpenAI integration.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenAI-API-black?logo=openai&logoColor=white" alt="OpenAI API">
  <img src="https://img.shields.io/badge/HTTPX-Web%20Requests-blue" alt="HTTPX">
  <img src="https://img.shields.io/badge/BeautifulSoup-HTML%20Parsing-green" alt="BeautifulSoup">
  <img src="https://img.shields.io/badge/Status-Portfolio%20Project-success" alt="Status">
</p>

## Features

- Persistent conversation memory
- OpenAI-powered responses
- Web page fetching and summarization
- Basic web search
- Local file read/write tools
- Safe mathematical expression evaluation
- URL content extraction
- Rate limiting and robots.txt awareness
- Voice interface with microphone input and spoken responses

## Technologies

- Python
- OpenAI API
- HTTPX
- BeautifulSoup
- Readability
- DuckDuckGo Search
- PDFMiner
- Gradio

## How It Works

The agent evaluates the user's request and decides whether to use memory, web tools, file operations, calculations, or the OpenAI model before generating a response.

## Setup

1. Clone the repository.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key as an environment variable:

```bash
OPENAI_API_KEY
```

4. Run the command-line AI agent:

```bash
python ai_agent.py
```

### Voice Interface

To launch the web-based voice interface:

```bash
python voice_chat.py
```

The interface supports microphone input, speech-to-text, text responses, and optional text-to-speech playback.

## Security

- API keys are loaded through environment variables and are not stored in the source code.
- Local memory and workspace files are excluded through `.gitignore`.
- Web requests include basic rate limiting and robots.txt awareness.
- File operations are restricted to the application's local workspace.

## Project Structure

```text
python-ai-agent/
├── ai_agent.py
├── voice_chat.py
├── requirements.txt
├── .gitignore
├── ai-agent-banner.png
└── README.md
```

## Author

**Angel Abrigo**

Information Science Student — University of Maryland  
A.S. in Computer Science — Montgomery College
