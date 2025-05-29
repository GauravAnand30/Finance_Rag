# 💰 Financial Intelligence RAG System

## Overview

The Financial Intelligence RAG (Retrieval Augmented Generation) System is an AI-powered application designed to provide comprehensive financial insights by combining real-time market data with a knowledge base built from financial documents. It leverages Large Language Models (LLMs) to answer complex financial queries, offer real-time stock and cryptocurrency data, and enable users to expand its knowledge by uploading relevant financial documents.

This project aims to demonstrate an end-to-end RAG system for the financial domain, showcasing capabilities in data retrieval, processing, intelligent querying, and dynamic UI presentation.

## Features

* **Intelligent Querying:** Ask natural language financial questions and get insightful answers augmented by both real-time data and stored documents.
* **Real-Time Market Data:** Access up-to-the-minute stock prices and cryptocurrency data with interactive charts and technical indicators (Moving Average, RSI).
* **Document Knowledge Base:** Upload earnings reports, SEC filings, research papers, or any text-based financial documents to enrich the system's understanding.
* **Technical Indicator Calculation:** Automatically computes common technical indicators (e.g., 20-period Moving Average, 14-period Relative Strength Index) for market data.
* **Scalable Architecture:** Built with FastAPI for the backend and Streamlit for the frontend, ensuring a modular and scalable design.
* **Vector Database Integration:** Utilizes ChromaDB (or FAISS) for efficient storage and retrieval of document embeddings.

## Technical Stack

* **Backend:** FastAPI (Python)
    * **LLM Integration:** Groq API (llama3-8b-8192 model)
    * **Data Fetching:** `yfinance` (for stocks), CoinGecko API (for cryptocurrencies), Alpha Vantage (fallback)
    * **Technical Analysis:** `TA-Lib`
    * **Vector Database:** `ChromaDB` (default) or `FAISS`
    * **Embeddings:** `GPT4AllEmbeddings` (local embedding model from Langchain)
    * **Text Processing:** `langchain_text_splitters`
* **Frontend:** Streamlit (Python)
    * **Interactive Charts:** `Plotly`
* **Containerization (Optional but Recommended):** Docker (for easier deployment)

## System Architecture

The system follows a client-server architecture:

1.  **Streamlit Frontend:** Provides the user interface for interacting with the system, submitting queries, uploading documents, and viewing data.
2.  **FastAPI Backend:**
    * Receives requests from the frontend.
    * **`DataFetcher`:** Handles fetching real-time financial data from external APIs (yfinance, CoinGecko, Alpha Vantage).
    * **`DataProcessor`:** Calculates technical indicators on fetched financial data.
    * **`VectorDBManager`:** Manages the document knowledge base (embedding, storage, and retrieval) using ChromaDB/FAISS and GPT4AllEmbeddings.
    * **`LLMService`:** Interacts with the Groq API to generate responses based on user queries, augmented with financial and document context.
    * **`intelligent_query` endpoint:** The core RAG orchestration, combining real-time data lookups, document retrieval, and LLM prompting.


## Setup Documentation

Follow these steps to set up and run the Financial Intelligence RAG System locally.

### Prerequisites

* Python 3.9+
* `pip` (Python package installer)
* Git
* API Keys:
    * **Groq API Key:** Required for LLM inference. Get one from [Groq Console](https://console.groq.com/keys).
    * **Alpha Vantage API Key (Optional, but recommended for fallback):** Get one from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/financial-rag-system.git](https://github.com/your-username/financial-rag-system.git)
    cd financial-rag-system
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv myenv
    # On Windows
    .\myenv\Scripts\activate
    # On macOS/Linux
    source myenv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If `TA-Lib` installation fails, refer to [TA-Lib's installation guide](https://mrjbq7.github.io/ta-lib/install.html). You might need to install `ta-lib` system-wide first.)*

4.  **Configure API Keys:**
    Create a `.env` file in the root directory of the project and add your API keys:
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ALPHA_VANTAGE_API_KEY="your_alpha_vantage_api_key_here"
    # Optional: Choose your vector database (default is chroma)
    # VECTOR_DB_TYPE="faiss"
    ```

### Running the Application

1.  **Start the Backend Server:**
    Open a new terminal, activate your virtual environment, navigate to the project root, and run:
    ```bash
    uvicorn backend:app --reload --host 0.0.0.0 --port 8000
    ```
    The backend should now be running on `http://localhost:8000`.

2.  **Start the Frontend Application:**
    Open another new terminal, activate your virtual environment, navigate to the project root, and run:
    ```bash
    streamlit run frontend.py
    ```
    This will open the Streamlit application in your web browser.

## API Documentation

The backend API is built with FastAPI and automatically generates interactive documentation using Swagger UI.

* **Swagger UI:** Access the API documentation at `http://localhost:8000/docs` once the backend is running.
* **Redoc:** Access the API documentation at `http://localhost:8000/redoc` once the backend is running.

### Key Endpoints:

* **`POST /api/query/`**: Intelligent query endpoint.
    * **Request Body:**
        ```json
        {
          "query": "string",
          "conversation_history": [
            {
              "role": "user",
              "content": "string"
            },
            {
              "role": "assistant",
              "content": "string"
            }
          ]
        }
        ```
    * **Response:** `QueryResponse` containing `response`, `sources`, `metadata`, and `performance_metrics`.
* **`GET /api/finance/stocks/{symbol}`**: Get real-time and historical stock data.
    * **Parameters:** `symbol` (e.g., `AAPL`, `NVDA`)
    * **Response:** List of `StockData` objects.
* **`GET /api/finance/crypto/{symbol}`**: Get real-time and historical cryptocurrency data.
    * **Parameters:** `symbol` (e.g., `BTCUSD`, `ETHUSD`)
    * **Response:** List of `StockData` objects.
* **`POST /api/documents/upload`**: Upload a financial document for RAG.
    * **Request Body:** `file` (multipart/form-data)
    * **Response:** `DocumentUploadResponse` indicating success/failure.
* **`GET /api/finance/health`**: Health check for financial data services.
* **`GET /api/documents/health`**: Health check for document RAG services.

## Demo Instructions

Follow these steps to test the key features of the system.

1.  **Ensure both Backend and Frontend are running.** (Refer to "Running the Application" section).

2.  **Test "Intelligent Query":**
    * Navigate to the "Intelligent Query" page in the Streamlit app.
    * Try the following sample queries:
        * "What is the current market price of Nvidia?"
        * "Show me the latest stock data for GOOGL including its RSI."
        * "What are the recent trends for Bitcoin?"
        * "Summarize what you know about Apple's financial performance from the documents." (This assumes you've uploaded relevant documents).
        * "Explain the concept of RSI in stock analysis." (Tests general financial knowledge)

3.  **Test "Real-Time Data" lookup:**
    * Navigate to the "Real-Time Data" page.
    * Enter a symbol like `MSFT` or `ETHUSD` in the input box.
    * Click "Get Data" to view the latest data, candlestick chart, and technical indicators.

4.  **Test "Upload Document":**
    * Navigate to the "Upload Document" page.
    * Prepare a `.txt` file (e.g., `sample_earnings.txt`) with some financial information.
    * Upload the file using the file uploader.
    * Observe the success message.
    * Go back to "Intelligent Query" and ask a question related to the content of your uploaded document (e.g., "What were the key takeaways from `sample_earnings.txt`?").

5.  **Test "System Health":**
    * Navigate to the "System Health" page.
    * Click "Check Health" to see the status of the financial data and document RAG services.

## Technical Report

### Architecture Decisions

* **Separation of Concerns:** The project is divided into a FastAPI backend and a Streamlit frontend, promoting modularity, independent scaling, and clear API boundaries.
* **Retrieval Augmented Generation (RAG):** This approach was chosen to overcome the limitations of pre-trained LLMs, allowing the system to access up-to-date real-time data and proprietary document knowledge.
* **LLM Choice (Groq/Llama3-8b-8192):** Groq was selected for its high inference speed, making the RAG system more responsive and improving user experience. Llama3-8b-8192 provides a good balance of capability and efficiency.
* **Vector Database (ChromaDB/FAISS):** These were chosen for efficient semantic search and retrieval of relevant document chunks, which are crucial for the "Retrieval" part of RAG. `GPT4AllEmbeddings` provides a robust local embedding solution.
* **Financial Data APIs:** `yfinance` offers a convenient and widely used Pythonic interface for stock data, while CoinGecko is a reliable source for cryptocurrency data. Alpha Vantage was included as a robust fallback.
* **Technical Indicators (`TA-Lib`):** Integrating TA-Lib allows for the calculation of common financial indicators directly within the data processing pipeline, providing deeper insights.

### Implementation Challenges and Solutions

1.  **Real-Time Data Integration:**
    * **Challenge:** Ensuring the LLM receives the most current market data and correctly interprets it. Initial attempts might lead to stale data being used or misinterpretations.
    * **Solution:** Implemented a dedicated `DataFetcher` to hit external APIs directly for real-time stock and crypto data (1-minute or 5-minute intervals). The `intelligent_query` function actively fetches this data for identified symbols and injects it into the LLM's prompt context, overriding static knowledge. Dynamic symbol detection using `COMPANY_TICKER_MAP` and regex helps in identifying query-mentioned entities.

2.  **Robust Symbol Identification:**
    * **Challenge:** Accurately identifying stock tickers or crypto pairs from natural language queries, especially for companies like "Nvidia" which might be mentioned by name rather than symbol "NVDA".
    * **Solution:** Developed a `COMPANY_TICKER_MAP` to translate common company names to their symbols. Combined this with basic regex to catch potential ticker symbols directly mentioned. This significantly improves the system's ability to fetch relevant data without explicit ticker input.

3.  **Technical Indicator Calculation Reliability:**
    * **Challenge:** `TA-Lib` functions often require a minimum number of data points to calculate indicators (e.g., 20 for 20-day MA, 14 for 14-day RSI). Fetching only the latest data point would not suffice.
    * **Solution:** Modified `DataFetcher` to fetch sufficient historical data (e.g., 1-day of 1-minute intervals for stocks, 1-day of 5-minute intervals for crypto) to allow `DataProcessor` to correctly calculate indicators before extracting just the latest values for the LLM context. Error handling was added for cases where insufficient data is still returned.

4.  **UI/UX for Data Visualization:**
    * **Challenge:** Presenting complex financial data (candlestick charts, multiple indicators) clearly and interactively within Streamlit.
    * **Solution:** Leveraged `Plotly.graph_objects` to create interactive candlestick charts and line plots for MA and RSI. Created a reusable `display_financial_data_and_charts` helper function to keep the frontend code DRY and consistent across different pages. Added a dedicated "Real-Time Data" page for direct data lookup.

## Demonstration Video (Optional)

You can find a video demonstration of the system's key features and functionalities here:

[![Watch the demo video](https://img.youtube.com/vi/YOUR_VIDEO_ID/hqdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)
*(Replace `YOUR_VIDEO_ID` with your actual YouTube video ID and update the image URL.)*

## Screenshots

Here are a few screenshots showcasing the system in action:

1.  **Intelligent Query Page with Real-time Data Response**
    ![Intelligent Query Demo](docs/intelligent_query_demo.png)
    *(Replace `docs/intelligent_query_demo.png` with your actual image path)*

2.  **Real-Time Data Page with Interactive Chart**
    ![Real-time Data Demo](docs/realtime_data_demo.png)
    *(Replace `docs/realtime_data_demo.png` with your actual image path)*

3.  **Document Upload Feature**
    ![Document Upload Demo](docs/document_upload_demo.png)
    *(Replace `docs/document_upload_demo.png` with your actual image path)*

## Future Enhancements

* **More Advanced NLP:** Implement named entity recognition (NER) specifically for financial terms to more robustly identify symbols and financial concepts in queries.
* **PDF Text Extraction:** Fully implement robust PDF text extraction using libraries like `PyMuPDF` or `pdfminer.six` to handle diverse PDF formats.
* **Time Series Analysis:** Integrate more sophisticated time series forecasting models.
* **Sentiment Analysis:** Add sentiment analysis capabilities for news articles or social media relevant to specific stocks/cryptos.
* **User Authentication:** Implement secure user authentication for production environments.
* **Scalability:** Deploy using Docker Compose or Kubernetes for easier management and scaling.
* **More comprehensive Technical Indicators:** Add support for a wider range of TA-Lib indicators (e.g., Bollinger Bands, MACD).

## Contributing

Feel free to fork this repository, submit pull requests, or open issues. Any contributions are welcome!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
