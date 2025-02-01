# Context-Aware AI Chatbot

A versatile conversational AI assistant built with Streamlit and LangChain that provides intelligent responses based on uploaded documentation. The chatbot uses Retrieval-Augmented Generation (RAG) to ensure accurate, context-specific interactions.

## Features

- 🤖 Interactive chat interface built with Streamlit
- 📚 RAG capabilities for context-aware responses
- 📝 Support for PDF and TXT document uploads
- 💾 Persistent vector storage using Chroma
- 🧠 Configurable conversation memory
- 🎛️ Adjustable AI temperature settings
- 🔄 Real-time response generation
- 📊 User-friendly interface with sidebar controls

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/procodz/context-aware-chatbot.git
cd context-aware-chatbot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

## Required Dependencies

```
streamlit
langchain
langchain-openai
langchain-community
PyPDF2
python-dotenv
chromadb
```

## Usage

1. Start the application:
```bash
streamlit run chat_bot.py
```

2. Access the chatbot through your web browser (typically at `http://localhost:8501`)

3. Configure the chatbot using the sidebar:
   - Adjust the temperature (0.0-1.0) for response variation
   - Set the memory length (1-10 messages)

4. Upload relevant documents (PDF/TXT) to enhance the chatbot's knowledge

5. Start chatting with the assistant

## Features in Detail

### Document Processing
- Supports PDF and TXT file formats
- Automatically extracts and processes text content
- Splits documents into manageable chunks
- Creates and updates vector embeddings for contextual search

### Conversation Management
- Maintains conversation history
- Configurable memory window
- Preserves context across multiple interactions

### Vector Store
- Uses Chroma for efficient vector storage
- Persistent storage in local directory
- Dynamic updating with new document uploads

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| Temperature | Controls response randomness | 0.3 |
| Memory Length | Number of previous messages to retain | 5 |
| Model | OpenAI model used | GPT-4 |

## Project Structure

```
.
├── chat_bot.py          # Main application file
├── .env                # Environment variables
├── requirements.txt    # Project dependencies
├── chroma_db/         # Vector store directory
└── README.md          # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request


## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://www.langchain.com/)
- Uses [OpenAI](https://openai.com/) GPT-4 model
