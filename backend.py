# backend.py
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil
import logging
import time
import pandas as pd
import numpy as np
import talib
import requests
from groq import Groq
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import GPT4AllEmbeddings # A local embedding model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic_settings import BaseSettings, SettingsConfigDict
import re # Import regex module

# --- Configuration (Settings) ---
class Settings(BaseSettings):
    PROJECT_NAME: str = "Financial Intelligence RAG System"
    PROJECT_VERSION: str = "1.0.0"
    PROJECT_DESCRIPTION: str = "AI system for financial market data and document analysis."

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"

    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma") # or "faiss"
    FAISS_INDEX_PATH: str = "embeddings/faiss_index.bin"
    CHROMA_DB_PATH: str = "embeddings/chroma_db"

    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY")
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY") # Not used in current example, but good to have
    FMP_API_KEY: str = os.getenv("FMP_API_KEY") # Not used in current example, but good to have

    STOCK_SYMBOLS: list[str] = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    CRYPTO_PAIRS: list[str] = ["BTCUSD", "ETHUSD"]
    # Add a mapping for common company names to tickers for better detection
    COMPANY_TICKER_MAP: dict[str, str] = {
        "NVIDIA": "NVDA",
        "TESLA": "TSLA",
        "APPLE": "AAPL",
        "GOOGLE": "GOOGL",
        "MICROSOFT": "MSFT",
        "AMAZON": "AMZN",
        "BITCOIN": "BTCUSD",
        "ETHEREUM": "ETHUSD",
    }


    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Schemas (Models) ---
class StockData(BaseModel):
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    ma_20: Optional[float] = None
    rsi_14: Optional[float] = None
    volatility: Optional[float] = None

class DocumentUploadResponse(BaseModel):
    filename: str
    status: str
    message: str

class QueryRequest(BaseModel):
    query: str
    conversation_history: List[Dict[str, str]] = []

class QueryResponse(BaseModel): # Corrected from BaseBaseModel
    response: str
    sources: List[str]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    dependencies: Dict[str, str]

# --- Utility Functions ---
def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.create_documents([text])

# --- Services ---

class DataFetcher:
    def __init__(self):
        self.alpha_vantage_ts = None
        # Initialize Alpha Vantage only if API key is present
        if settings.ALPHA_VANTAGE_API_KEY:
            try:
                from alpha_vantage.timeseries import TimeSeries
                self.alpha_vantage_ts = TimeSeries(key=settings.ALPHA_VANTAGE_API_KEY, output_format='pandas')
            except ImportError:
                logger.warning("alpha_vantage not installed. Some features might be limited.")
            except Exception as e:
                logger.error(f"Error initializing Alpha Vantage: {e}")

        # Add a mapping for crypto symbols to CoinGecko IDs
        self.crypto_id_map = {
            "BTCUSD": "bitcoin",
            "ETHUSD": "ethereum",
            # Add other crypto mappings here if needed
        }

    async def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d", interval="1m") # Fetching 1 day of 1-minute data for near real-time
            if df.empty:
                # If 1-minute data is empty, try daily data for last month
                df = ticker.history(period="1mo", interval="1d")
                if df.empty:
                    logger.warning(f"No data fetched for {symbol} using yfinance.")
                    return pd.DataFrame()

            df = df.reset_index()
            # Handle both 'Date' and 'Datetime' for index
            if 'Date' in df.columns:
                df['timestamp'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif 'Datetime' in df.columns:
                df['timestamp'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                logger.error(f"Unexpected index column in yfinance data for {symbol}.")
                return pd.DataFrame()

            df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            return df
        except ImportError:
            logger.warning("yfinance not installed. Falling back to Alpha Vantage if available.")
            if self.alpha_vantage_ts:
                try:
                    data, meta_data = self.alpha_vantage_ts.get_intraday(symbol=symbol, interval='1min', outputsize='compact') # Try intraday
                    if data.empty:
                        data, meta_data = self.alpha_vantage_ts.get_daily(symbol=symbol, outputsize='compact') # Fallback to daily
                    data = data.rename(columns={
                        '1. open': 'open', '2. high': 'high', '3. low': 'low',
                        '4. close': 'close', '5. volume': 'volume'
                    })
                    data.index = data.index.map(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S'))
                    data = data.reset_index().rename(columns={'index': 'timestamp'})
                    return data
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol} with Alpha Vantage: {e}")
                    return pd.DataFrame()
            else:
                logger.error(f"No financial data API available for {symbol}.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} with yfinance: {e}")
            return pd.DataFrame()

    async def fetch_crypto_data(self, symbol: str) -> pd.DataFrame:
        # Get the CoinGecko ID from the map, default to lowercase symbol if not found
        coingecko_id = self.crypto_id_map.get(symbol.upper(), symbol.lower().replace('usd', ''))
        if not coingecko_id:
            logger.error(f"Invalid crypto symbol or no CoinGecko ID mapping for {symbol}")
            return pd.DataFrame()

        try:
            # Use the mapped CoinGecko ID in the URL
            # Fetch 1 day of 5-minute data for near real-time crypto data
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart?vs_currency=usd&days=1&interval=5m"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()['prices']
            df = pd.DataFrame(data, columns=['timestamp', 'close'])
            df['timestamp'] = df['timestamp'].apply(lambda x: pd.to_datetime(x / 1000, unit='s').strftime('%Y-%m-%d %H:%M:%S'))
            # For crypto, often only close price is readily available via simple APIs,
            # so open/high/low/volume might be placeholders or need more advanced APIs.
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']
            df['volume'] = 0
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()

data_fetcher = DataFetcher()

class DataProcessor:
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'close' not in df.columns:
            return df

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])

        if len(df) >= 20:
            df['ma_20'] = talib.SMA(df['close'], timeperiod=20)
        if len(df) >= 14:
            df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        if len(df) >= 2:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_returns'].rolling(window=20).std() * np.sqrt(252)

        return df.replace({np.nan: None})

    def format_stock_data(self, df: pd.DataFrame) -> List[StockData]:
        if df.empty:
            return []
        return [StockData(**row.dropna().to_dict()) for index, row in df.iterrows()]

data_processor = DataProcessor()

class VectorDBManager:
    def __init__(self):
        self.embedding_function = GPT4AllEmbeddings()
        self.vectorstore = None
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        if settings.VECTOR_DB_TYPE == "chroma":
            persist_directory = settings.CHROMA_DB_PATH
            os.makedirs(persist_directory, exist_ok=True)
            try:
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embedding_function
                )
                logger.info(f"ChromaDB initialized at {persist_directory}")
            except Exception as e:
                logger.error(f"Error initializing ChromaDB: {e}")
                self.vectorstore = None
        elif settings.VECTOR_DB_TYPE == "faiss":
            os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
            if os.path.exists(settings.FAISS_INDEX_PATH):
                try:
                    self.vectorstore = FAISS.load_local(
                        folder_path=os.path.dirname(settings.FAISS_INDEX_PATH),
                        embeddings=self.embedding_function,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"FAISS index loaded from {settings.FAISS_INDEX_PATH}")
                except Exception as e:
                    logger.error(f"Error loading FAISS index: {e}. Reinitializing.")
                    self.vectorstore = FAISS.from_texts([""], self.embedding_function)
            else:
                self.vectorstore = FAISS.from_texts([""], self.embedding_function)
                logger.info(f"FAISS index initialized (new).")
        else:
            raise ValueError(f"Unsupported VECTOR_DB_TYPE: {settings.VECTOR_DB_TYPE}")

    async def add_document(self, text_content: str, filename: str) -> None:
        chunks = split_text_into_chunks(text_content, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not chunks:
            logger.warning(f"No chunks generated for document: {filename}")
            return

        documents_with_metadata = []
        for i, chunk in enumerate(chunks):
            chunk.metadata = {"source": filename, "chunk_id": i}
            documents_with_metadata.append(chunk)

        if self.vectorstore:
            try:
                if settings.VECTOR_DB_TYPE == "chroma":
                    self.vectorstore.add_documents(documents_with_metadata)
                    self.vectorstore.persist()
                elif settings.VECTOR_DB_TYPE == "faiss":
                    texts = [doc.page_content for doc in documents_with_metadata]
                    metadatas = [doc.metadata for doc in documents_with_metadata]

                    if self.vectorstore.index.ntotal == 0:
                        self.vectorstore = FAISS.from_texts(texts, self.embedding_function, metadatas=metadatas)
                    else:
                        self.vectorstore.add_texts(texts, metadatas=metadatas)

                    self.vectorstore.save_local(folder_path=os.path.dirname(settings.FAISS_INDEX_PATH), index_name=os.path.basename(settings.FAISS_INDEX_PATH))
                logger.info(f"Document '{filename}' added to vector DB.")
            except Exception as e:
                logger.error(f"Error adding document to vector DB: {e}")
        else:
            logger.error("Vector database not initialized. Cannot add document.")

    async def retrieve_documents(self, query: str, k: int = settings.TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            logger.error("Vector database not initialized. Cannot retrieve documents.")
            return []
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving documents from vector DB: {e}")
            return []

vector_db_manager = VectorDBManager()

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    async def generate_response(self, prompt: str, conversation_history: List[Dict[str, str]] = []) -> str:
        messages = [{"role": "system", "content": "You are a helpful financial AI assistant. Provide concise and accurate answers based on the provided context and your knowledge. If you don't know the answer, state that."}]
        for msg in conversation_history:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})

        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1024,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."

llm_service = LLMService()

# --- FastAPI Application ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Financial Intelligence RAG System API!"}

# --- API Endpoints ---

@app.get("/api/finance/stocks/{symbol}", response_model=List[StockData], summary="Get real-time and historical stock data with indicators")
async def get_stock_data(symbol: str):
    logger.info(f"Fetching data for stock: {symbol}")
    df = await data_fetcher.fetch_stock_data(symbol.upper())
    if df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Stock data not found for {symbol}")
    df_processed = data_processor.calculate_technical_indicators(df)
    return data_processor.format_stock_data(df_processed)

@app.get("/api/finance/crypto/{symbol}", response_model=List[StockData], summary="Get real-time and historical crypto data with indicators")
async def get_crypto_data(symbol: str):
    logger.info(f"Fetching data for crypto: {symbol}")
    df = await data_fetcher.fetch_crypto_data(symbol.upper())
    if df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Crypto data not found for {symbol}")
    df_processed = data_processor.calculate_technical_indicators(df)
    return data_processor.format_stock_data(df_processed)

@app.get("/api/finance/health", response_model=HealthCheckResponse, summary="Health check endpoint for financial data")
async def financial_health_check():
    status_msg = "Healthy"
    dependencies = {"data_fetcher": "Operational"}
    try:
        test_df = await data_fetcher.fetch_stock_data("AAPL")
        if test_df.empty:
            status_msg = "Degraded"
            dependencies["data_fetcher"] = "Could not fetch sample data (API limit or invalid key)"
    except Exception as e:
        status_msg = "Unhealthy"
        dependencies["data_fetcher"] = f"Error: {e}"
    return HealthCheckResponse(status=status_msg, message="Financial data service health check", dependencies=dependencies)

@app.post("/api/documents/upload", response_model=DocumentUploadResponse, summary="Upload a financial document for RAG")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file name provided.")

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".txt", ".pdf"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type. Only .txt and .pdf are supported.")

    # Define the document storage path relative to the backend.py file
    DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "financial_documents")
    os.makedirs(DOCS_DIR, exist_ok=True)
    file_path = os.path.join(DOCS_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text_content = ""
        if file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        elif file_extension == ".pdf":
            # Placeholder for PDF parsing. Install PyMuPDF (fitz) for actual implementation.
            # pip install PyMuPDF
            # import fitz
            # doc = fitz.open(file_path)
            # for page in doc:
            #     text_content += page.get_text()
            # doc.close()
            raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="PDF parsing is not fully implemented. Please upload .txt files or implement PDF parsing.")

        if not text_content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not extract text from document or document is empty.")

        await vector_db_manager.add_document(text_content, file.filename)
        return DocumentUploadResponse(filename=file.filename, status="success", message="Document uploaded and processed successfully.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process document: {e}")

@app.get("/api/documents/health", response_model=HealthCheckResponse, summary="Health check endpoint for document RAG")
async def document_rag_health_check():
    status_msg = "Healthy"
    dependencies = {"vector_db": "Operational"}
    try:
        # Try a dummy retrieval to check DB
        test_docs = await vector_db_manager.retrieve_documents("test", k=1)
        if vector_db_manager.vectorstore is None:
             status_msg = "Degraded"
             dependencies["vector_db"] = "Vector database not initialized."
    except Exception as e:
        status_msg = "Unhealthy"
        dependencies["vector_db"] = f"Error: {e}"
    return HealthCheckResponse(status=status_msg, message="Document RAG service health check", dependencies=dependencies)

@app.post("/api/query/", response_model=QueryResponse, summary="Intelligent query combining real-time data and document knowledge")
async def intelligent_query(request: QueryRequest):
    start_time = time.time()
    query = request.query
    conversation_history = request.conversation_history
    sources = []
    metadata = {}

    financial_data_context = ""
    # Combine predefined symbols and company names for better detection
    all_known_symbols = list(settings.STOCK_SYMBOLS) + list(settings.CRYPTO_PAIRS)
    potential_symbols_to_fetch = set()

    # Add predefined symbols
    potential_symbols_to_fetch.update(all_known_symbols)

    # Attempt to extract ticker-like strings or map company names from the query
    # Simple regex to find uppercase words that might be tickers (2-5 characters)
    extracted_tickers = re.findall(r'\b[A-Z]{2,5}\b', query.upper())
    potential_symbols_to_fetch.update(extracted_tickers)

    # Map company names to tickers from the COMPANY_TICKER_MAP
    for company_name, ticker in settings.COMPANY_TICKER_MAP.items():
        if company_name.lower() in query.lower():
            potential_symbols_to_fetch.add(ticker)

    # Prioritize unique symbols and fetch data
    for symbol in list(potential_symbols_to_fetch): # Convert to list to iterate and modify the set if needed
        # Basic check to determine if it's a stock or crypto based on common patterns or predefined lists
        is_stock = symbol in settings.STOCK_SYMBOLS or \
                   (symbol not in settings.CRYPTO_PAIRS and len(symbol) <= 5 and symbol.isalpha()) # Heuristic for stock
        is_crypto = symbol in settings.CRYPTO_PAIRS or \
                    (symbol.endswith("USD") or symbol.endswith("USDT")) # Heuristic for crypto

        df = pd.DataFrame()
        if is_stock:
            df = await data_fetcher.fetch_stock_data(symbol)
            if not df.empty:
                processed_data = data_processor.calculate_technical_indicators(df.tail(1))
                if not processed_data.empty:
                    latest_data = processed_data.iloc[0].to_dict()
                    financial_data_context += f"Latest Stock Data for {symbol} (as of {latest_data.get('timestamp', 'N/A')}): " \
                                              f"Open: {latest_data.get('open', 'N/A')}, High: {latest_data.get('high', 'N/A')}, " \
                                              f"Low: {latest_data.get('low', 'N/A')}, Close: {latest_data.get('close', 'N/A')}, " \
                                              f"Volume: {latest_data.get('volume', 'N/A')}. "
                    if latest_data.get('ma_20') is not None:
                        financial_data_context += f"20-Day MA: {latest_data['ma_20']:.2f}. "
                    if latest_data.get('rsi_14') is not None:
                        financial_data_context += f"14-Day RSI: {latest_data['rsi_14']:.2f}. "
                    if latest_data.get('volatility') is not None:
                        financial_data_context += f"Volatility: {latest_data['volatility']:.2f}.\n"
                    sources.append(f"Real-time data: {symbol}")
                else:
                    financial_data_context += f"Could not process real-time data for {symbol}.\n"
            else:
                financial_data_context += f"Could not fetch real-time stock data for {symbol}.\n"
        elif is_crypto:
            df = await data_fetcher.fetch_crypto_data(symbol)
            if not df.empty:
                processed_data = data_processor.calculate_technical_indicators(df.tail(1))
                if not processed_data.empty:
                    latest_data = processed_data.iloc[0].to_dict()
                    financial_data_context += f"Latest Crypto Data for {symbol} (as of {latest_data.get('timestamp', 'N/A')}): " \
                                              f"Open: {latest_data.get('open', 'N/A')}, High: {latest_data.get('high', 'N/A')}, " \
                                              f"Low: {latest_data.get('low', 'N/A')}, Close: {latest_data.get('close', 'N/A')}. "
                    if latest_data.get('ma_20') is not None:
                        financial_data_context += f"20-Day MA: {latest_data['ma_20']:.2f}. "
                    if latest_data.get('rsi_14') is not None:
                        financial_data_context += f"14-Day RSI: {latest_data['rsi_14']:.2f}. "
                    if latest_data.get('volatility') is not None:
                        financial_data_context += f"Volatility: {latest_data['volatility']:.2f}.\n"
                    sources.append(f"Real-time data: {symbol}")
                else:
                    financial_data_context += f"Could not process real-time crypto data for {symbol}.\n"
            else:
                financial_data_context += f"Could not fetch real-time crypto data for {symbol}.\n"

    context = ""
    if financial_data_context:
        context += "Real-time Financial Data Context:\n" + financial_data_context + "\n"

    retrieved_docs = await vector_db_manager.retrieve_documents(query)
    if retrieved_docs:
        context += "Document Context:\n"
        for doc in retrieved_docs:
            context += f"- Source: {doc['metadata'].get('source', 'N/A')}\n"
            context += f"  Content: {doc['page_content']}\n"
            sources.append(doc['metadata'].get('source', 'N/A'))
        sources = list(set(sources)) # Remove duplicates

    if not context:
        logger.info("No specific context found. Relying on LLM general knowledge.")
        full_prompt = f"Given your general financial knowledge, answer the query accurately and comprehensively. If you don't have enough information, state that you cannot provide a precise answer.\n\nQuery: {query}\n\nAnswer:"
    else:
        full_prompt = f"Given the following context and your general financial knowledge, answer the query accurately and comprehensively. Prioritize information from the provided context. If the information is not in the context, use your general knowledge but clearly state if information is from a specific source.\n\nQuery: {query}\n\n{context}\n\nAnswer:"


    response_content = await llm_service.generate_response(full_prompt, conversation_history)

    end_time = time.time()
    performance_metrics = {
        "processing_time_seconds": round(end_time - start_time, 2),
        "llm_model_used": "llama3-8b-8192",
        "retrieved_documents_count": len(retrieved_docs)
    }

    return QueryResponse(
        response=response_content,
        sources=sources,
        metadata=metadata,
        performance_metrics=performance_metrics
    )

if __name__ == "__main__":
    import uvicorn
    # Create necessary directories at startup if they don't exist
    os.makedirs("data/financial_documents", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)