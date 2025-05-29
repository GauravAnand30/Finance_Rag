# frontend.py
import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import os

# Backend API URL (ensure this matches your backend's host and port)
BACKEND_URL = "http://localhost:8000/api"

st.set_page_config(layout="wide", page_title="Financial RAG System")

st.title("💰 Financial Intelligence RAG System")

# --- Sidebar for Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Intelligent Query", "Real-Time Data", "Upload Document", "System Health"])

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This system combines real-time financial data with a document knowledge base "
    "to provide intelligent analysis using a Large Language Model (LLM). "
    "Built with FastAPI and Streamlit."
)

# --- Page: Intelligent Query ---
if page == "Intelligent Query":
    st.header("Ask a Financial Question")
    st.write("Get insights by combining real-time data and stored financial documents.")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Type your financial question here (e.g., What is Apple's current stock performance? Summarize Google's Q4 earnings.)")

    if query:
        st.session_state.conversation_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/query/",
                        json={"query": query, "conversation_history": st.session_state.conversation_history}
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.write(result["response"])
                    st.session_state.conversation_history.append({"role": "assistant", "content": result["response"]})

                    if result["sources"]:
                        st.markdown("**Sources:**")
                        for source in result["sources"]:
                            st.write(f"- {source}")

                    st.markdown("**Performance Metrics:**")
                    st.json(result["performance_metrics"])

                except requests.exceptions.RequestException as e:
                    st.error(f"Error communicating with the backend: {e}")
                except json.JSONDecodeError:
                    st.error("Received an invalid response from the backend.")

# --- Page: Real-Time Data ---
elif page == "Real-Time Data":
    st.header("Live Financial Market Data")

    symbol_type = st.radio("Select Asset Type", ("Stocks", "Cryptocurrencies"))

    if symbol_type == "Stocks":
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL, MSFT)", "AAPL").upper()
        if st.button("Fetch Stock Data"):
            if symbol:
                try:
                    response = requests.get(f"{BACKEND_URL}/finance/stocks/{symbol}")
                    response.raise_for_status()
                    data = response.json()
                    if data:
                        df = pd.DataFrame(data)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                        st.subheader(f"Latest Data for {symbol}")
                        st.dataframe(df.tail())

                        st.subheader(f"Price Chart for {symbol}")
                        fig = go.Figure(data=[go.Candlestick(
                            x=df.index,
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close']
                        )])
                        fig.update_layout(xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader(f"Technical Indicators for {symbol}")
                        if 'ma_20' in df.columns and df['ma_20'].notna().any():
                            fig_ma = go.Figure()
                            fig_ma.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'))
                            fig_ma.add_trace(go.Scatter(x=df.index, y=df['ma_20'], mode='lines', name='20-Day MA'))
                            fig_ma.update_layout(title_text='Close Price and 20-Day Moving Average')
                            st.plotly_chart(fig_ma, use_container_width=True)
                        if 'rsi_14' in df.columns and df['rsi_14'].notna().any():
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], mode='lines', name='RSI (14)'))
                            fig_rsi.update_layout(title_text='Relative Strength Index (RSI)', yaxis_range=[0,100])
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                            st.plotly_chart(fig_rsi, use_container_width=True)


                    else:
                        st.warning(f"No data retrieved for {symbol}.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Error fetching data: {e.response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error communicating with the backend: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please enter a stock symbol.")

    elif symbol_type == "Cryptocurrencies":
        symbol = st.text_input("Enter Crypto Pair (e.g., BTCUSD, ETHUSD)", "BTCUSD").upper()
        if st.button("Fetch Crypto Data"):
            if symbol:
                try:
                    response = requests.get(f"{BACKEND_URL}/finance/crypto/{symbol}")
                    response.raise_for_status()
                    data = response.json()
                    if data:
                        df = pd.DataFrame(data)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                        st.subheader(f"Latest Data for {symbol}")
                        st.dataframe(df.tail())

                        st.subheader(f"Price Chart for {symbol}")
                        fig = go.Figure(data=[go.Candlestick(
                            x=df.index,
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close']
                        )])
                        fig.update_layout(xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

                        if 'ma_20' in df.columns and df['ma_20'].notna().any():
                            fig_ma = go.Figure()
                            fig_ma.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'))
                            fig_ma.add_trace(go.Scatter(x=df.index, y=df['ma_20'], mode='lines', name='20-Day MA'))
                            fig_ma.update_layout(title_text='Close Price and 20-Day Moving Average')
                            st.plotly_chart(fig_ma, use_container_width=True)
                        if 'rsi_14' in df.columns and df['rsi_14'].notna().any():
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], mode='lines', name='RSI (14)'))
                            fig_rsi.update_layout(title_text='Relative Strength Index (RSI)', yaxis_range=[0,100])
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                            st.plotly_chart(fig_rsi, use_container_width=True)
                    else:
                        st.warning(f"No data retrieved for {symbol}.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Error fetching data: {e.response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error communicating with the backend: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please enter a crypto pair.")

# --- Page: Upload Document ---
elif page == "Upload Document":
    st.header("Upload Financial Documents")
    st.write("Upload earnings reports, SEC filings, or research papers to expand the knowledge base.")

    uploaded_file = st.file_uploader("Choose a file (TXT or PDF)", type=["txt", "pdf"])

    if uploaded_file is not None:
        if st.button("Upload and Process"):
            with st.spinner("Uploading and processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/documents/upload", files=files)
                    response.raise_for_status()
                    result = response.json()
                    if result["status"] == "success":
                        st.success(f"Document '{result['filename']}' uploaded and processed successfully!")
                    else:
                        st.error(f"Failed to process document: {result.get('message', 'Unknown error')}")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Error uploading file: {e.response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error communicating with the backend: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

# --- Page: System Health ---
elif page == "System Health":
    st.header("System Health Check")
    st.write("Monitor the status of backend services.")

    if st.button("Check Health"):
        st.subheader("Financial Data Service Health")
        try:
            response = requests.get(f"{BACKEND_URL}/finance/health")
            response.raise_for_status()
            health_data = response.json()
            st.json(health_data)
            if health_data["status"] == "Healthy":
                st.success("Financial Data Service: Operational")
            else:
                st.warning(f"Financial Data Service: {health_data['status']} - {health_data['message']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error checking financial data service health: {e}")

        st.subheader("Document RAG Service Health")
        try:
            response = requests.get(f"{BACKEND_URL}/documents/health")
            response.raise_for_status()
            health_rag = response.json()
            st.json(health_rag)
            if health_rag["status"] == "Healthy":
                st.success("Document RAG Service: Operational")
            else:
                st.warning(f"Document RAG Service: {health_rag['status']} - {health_rag['message']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error checking document RAG service health: {e}")

        st.subheader("LLM Service (Groq) Connectivity")
        st.info("LLM service connectivity is implicitly checked during intelligent queries.")