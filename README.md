# LLM-Powered Intelligent Query–Retrieval System

This project is an intelligent query-retrieval system designed to process large documents and provide contextual answers to natural language questions. It is built with FastAPI and leverages Hugging Face models for embeddings and language understanding.

## System Architecture

The system follows these steps:

1.  **Input Document:** The system accepts a URL to a PDF document.
2.  **Document Loading and Chunking:** The PDF is downloaded, loaded, and split into smaller, manageable chunks of text using the `RecursiveCharacterTextSplitter` from LangChain. This splitter is designed to work well with long texts, recursively splitting them by a set of characters.
3.  **Embedding Generation:** Each text chunk is converted into a numerical representation (embedding) using the `all-MiniLM-L6-v2` sentence-transformer model. This is a high-performance model that is well-suited for semantic search.
4.  **Vector Store:** The embeddings are stored in a FAISS (Facebook AI Similarity Search) vector store, which allows for efficient similarity searches on large datasets of vectors.
5.  **Query Processing:** For each question, the system performs a similarity search on the FAISS vector store to find the most relevant text chunks. This process is also known as Retrieval-Augmented Generation (RAG).
6.  **LLM-Powered Answering:** The retrieved context and the question are passed to the `mistralai/Mistral-7B-Instruct-v0.2` large language model to generate a human-like answer. This model is a powerful, instruction-following LLM that is well-suited for this task.
7.  **JSON Output:** The final answers are returned in a structured JSON format.

## API Documentation

**Base URL:** `http://localhost:8000/api/v1`

### Run Submission

*   **Endpoint:** `/hackrx/run`
*   **Method:** `POST`
*   **Description:** Submits a document and a list of questions for processing.
*   **Request Body:**

    ```json
    {
        "documents": "<URL_TO_YOUR_PDF>",
        "questions": [
            "<Your Question 1>",
            "<Your Question 2>"
        ]
    }
    ```

*   **Success Response (200 OK):**

    ```json
    {
        "answers": [
            "<Answer to Question 1>",
            "<Answer to Question 2>"
        ],
        "time_taken": <Time in seconds>,
        "document_url": "<URL_TO_YOUR_PDF>",
        "questions": [
            "<Your Question 1>",
            "<Your Question 2>"
        ]
    }
    ```

## How to Run the System

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Your Hugging Face API Key:**

    Make sure to replace `"YOUR_HUGGINGFACE_API_KEY"` in `app/main.py` with your actual Hugging Face API key.

3.  **Run the FastAPI Server:**

    ```bash
    uvicorn app.main:app --reload
    ```

    The server will be available at `http://localhost:8000`.

4.  **Making a Public URL with ngrok (Optional):**

    If you want to create a public URL for your local server, you can use `ngrok`.

    a.  **Install ngrok:** Follow the instructions on the [ngrok website](https://ngrok.com/download).

    b.  **Run ngrok:**

        ```bash
        ngrok http 8000
        ```

    ngrok will provide you with a public URL (e.g., `https://<random-string>.ngrok.io`) that you can use as a webhook.

## Evaluation Criteria

This system is designed with the following evaluation criteria in mind:

*   **Accuracy:** The use of semantic search and a powerful LLM ensures a high degree of accuracy in understanding the query and matching it with the relevant clauses in the document.
*   **Token Efficiency:** The retrieval-augmented generation (RAG) approach is token-efficient. Instead of sending the entire document to the LLM, we only send the most relevant parts, which significantly reduces the number of tokens used.
*   **Latency:** The system is optimized for speed. The use of a FAISS vector store for searching and a fast LLM endpoint helps to keep the response time low.
*   **Reusability:** The code is modular and can be easily extended to support other document types, embedding models, or LLMs.
*   **Explainability:** The RAG approach provides a degree of explainability. Since the answer is generated from specific, retrieved text chunks, you can trace the answer back to the source document.
