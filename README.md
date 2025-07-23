# RAG System with FastAPI and Milvus

## Getting Started

### 1. Install Dependencies

Make sure you have all required packages installed:

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root with your Milvus and OpenAI credentials:

```
OPENAI_API_KEY=your_openai_key
MILVUS_URI=your_milvus_uri
MILVUS_TOKEN=your_milvus_token
```

### 3. Start the FastAPI Server

Use **uvicorn** to run the server (do not use `python main.py`):

```bash
uvicorn main:app --reload
```

- The server will start at [http://localhost:8000](http://localhost:8000)
- Interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Example Usage

You can test the `/search` endpoint using the Swagger UI or with `curl`/Postman:

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "your search term", "limit": 5}'
```

### 5. Stopping the Server

Press `Ctrl+C` in the terminal to stop the server.

---

For more details, see the plan.md and code comments.
