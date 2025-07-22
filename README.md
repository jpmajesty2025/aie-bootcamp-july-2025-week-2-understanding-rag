# Embedding sparse data from Githubb into the Milvus Vector database 

## Features

- 🤖 **ChatGPT Integration**: Real-time chat with OpenAI's GPT models
- 🔍 **RAG System**: Retrieval-Augmented Generation using Milvus vector database
- 🎨 **Modern UI**: Beautiful, responsive design with smooth animations
- 📚 **Document Management**: Add documents to the knowledge base
- 🔐 **API Key Management**: Secure storage of OpenAI API keys
- 📊 **Real-time Status**: Monitor API and Milvus connection status
- 📱 **Responsive Design**: Works on desktop and mobile devices


## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables 
(Note: you'll need a Zillz account to give you access to managed Milvus)


Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
MILVUS_URI=uri_for_your_milvus_cluster
MILVUS_TOKEN=token_for_your_milvus_cluster
```

### 3. Start the Application

```bash
python load_data.py
```


## Usage

### Chat Interface

1. **Start a Conversation**: Type your message in the input field and press Enter or click the send button
2. **View Sources**: RAG sources are displayed in the sidebar when available
3. **Clear Chat**: Use the "Clear Chat" button to start a new conversation

### Adding Documents to RAG

1. Click the "Add Document" button
2. Enter the document text and optional metadata
3. Click "Add Document" to store it in the vector database
4. The document will be available for context in future conversations

### API Key Management

1. Enter your OpenAI API key in the settings section
2. Click "Save" to store it securely
3. The key will be saved in your browser's local storage

## API Endpoints

### Chat
- `POST /api/chat` - Send a message and get AI response with RAG sources

### Document Management
- `POST /api/add-document` - Add a document to the RAG system

### Health Check
- `GET /api/health` - Check API and Milvus connection status

## Configuration

### Milvus Setup

The application is configured to work with Zilliz Cloud (managed Milvus). The default configuration uses:

- **URI**: https://in03-4efcec782ae2f4c.serverless.gcp-us-west1.cloud.zilliz.com
- **Token**: dca9ee30dd6accca68a63953d96a07cf3295cb68d1df55d93823135499762886d4ea0c5cb68b7307f72afce73a991ebc16447360

### OpenAI Configuration

You'll need an OpenAI API key to use the chat functionality. Get one from [OpenAI's platform](https://platform.openai.com/api-keys).

## Features in Detail

### RAG System

The RAG (Retrieval-Augmented Generation) system works as follows:

1. **Document Ingestion**: Documents are converted to embeddings using OpenAI's text-embedding-ada-002 model
2. **Vector Storage**: Embeddings are stored in Milvus vector database
3. **Query Processing**: User queries are converted to embeddings
4. **Similarity Search**: Milvus finds the most similar documents
5. **Context Enhancement**: Retrieved documents provide context for GPT responses

### Vector Database Schema

```python
Collection Schema:
- id: INT64 (Primary Key, Auto ID)
- text: VARCHAR (Document text, max 65535 chars)
- embedding: FLOAT_VECTOR (1536 dimensions)
- metadata: VARCHAR (Optional metadata, max 65535 chars)
```

### UI Features

- **Real-time Chat**: Instant message sending and receiving
- **Typing Indicators**: Visual feedback during AI processing
- **Source Display**: Shows relevant documents with relevance scores
- **Responsive Design**: Works on all screen sizes
- **Dark/Light Theme**: Modern gradient design
- **Modal Dialogs**: Clean document addition interface

## Development

### Project Structure

```
fastapi-vibe-coding/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── env.example         # Environment variables template
├── templates/
│   └── index.html      # Main chat interface
└── static/
    ├── css/
    │   └── style.css   # Styling
    └── js/
        └── chat.js     # Frontend logic
```

### Adding New Features

1. **Backend**: Add new endpoints in `main.py`
2. **Frontend**: Update `static/js/chat.js` for new functionality
3. **Styling**: Modify `static/css/style.css` for UI changes

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Ensure your API key is valid and has sufficient credits
2. **Milvus Connection**: Check if Milvus is running and accessible
3. **CORS Issues**: The application serves static files directly, so CORS shouldn't be an issue

### Health Check

The application includes a health check endpoint that reports:
- API server status
- Milvus connection status

Access it at `/api/health` or view it in the sidebar.

## Security Notes

- API keys are stored in browser local storage (client-side)
- Consider implementing server-side API key management for production
- Milvus credentials are stored in environment variables
- The application doesn't persist chat history on the server

## Performance

- Uses async/await for non-blocking operations
- Implements connection pooling for Milvus
- Caches embeddings for better performance
- Limits conversation history to prevent memory issues

## License

This project is open source and available under the MIT License.
