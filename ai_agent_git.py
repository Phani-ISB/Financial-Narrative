#App.py file using Streamlit for dashboard, yahoo finance for stocks data and langchain for document processing.
# LLM used : Groq with llama 3.1 (8b)

# Importing all the dependencies
import numpy as np
import os
import streamlit as st
import yfinance as yf
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq

# Page Configuration 
st.set_page_config(page_title="Financial Report Agent", layout="wide")

# Initiallising LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.1-8b-instant"  # Strong reasoning model on Groq

# Defining Helper Functions ( 1. PDF Loader , Agents - Company name, Generate NSE ticker, Stock data)

# 1. PDF Loader from Langchain
def load_pdf_text(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text = "\n".join([p.page_content for p in pages])
    return text
# 2. Responses from Groq
def groq_completion(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a financial analyst AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()
# 3. Extract company name (Agent)
def extract_company_name(pdf_text):
    prompt = f"""
    From the following financial statement text, extract ONLY the company name.
    If not found, respond exactly with: NOT_FOUND

    Text:
    {pdf_text[:4000]}
    """
    result = groq_completion(prompt)
    return result
# 4. Generate NSE Ticker (for NSE stock check on yfinance) -(Agent)
def generate_nse_ticker(company_name):
    """
    Use Groq model to predict probable NSE ticker symbol.
    Returns ticker in format: XXXX.NS
    """
    prompt = f"""
    Predict the NSE stock ticker symbol for the Indian company: "{company_name}".
    Return ONLY the ticker symbol without exchange suffix.
    Example: Infosys -> INFY, Reliance Industries -> RELIANCE
    """

    try:
        ticker = groq_completion(prompt).strip().upper().replace(".NS", "")
        return f"{ticker}.NS"
    except Exception:
        return None

def validate_ticker(ticker):
    """
    Validate ticker by checking if price data exists.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if not hist.empty:
            return ticker
    except Exception:
        pass
    return None

# Stock Trends from 
def fetch_stock_trend(company_name):
    ticker_guess = generate_nse_ticker(company_name)
    valid_ticker = validate_ticker(ticker_guess) if ticker_guess else None

    # Fallback
    if not valid_ticker:
        fallback = company_name.split()[0].upper() + ".NS"
        valid_ticker = validate_ticker(fallback)

    # Manual intervention with confirmation button
    if not valid_ticker:
        st.warning("⚠️ Unable to auto-detect NSE ticker.")

        manual_ticker = st.text_input(
            "Enter NSE ticker manually (without .NS suffix):",
            key="manual_ticker_input"
        )

        confirm = st.button("Confirm Ticker")

        if not confirm:
            st.stop()  

        manual_ticker_full = manual_ticker.upper() + ".NS"
        valid_ticker = validate_ticker(manual_ticker_full)

        if not valid_ticker:
            st.error("Invalid ticker. Please check and re-enter.")
            st.stop()

    stock = yf.Ticker(valid_ticker)
    hist = stock.history(period="6mo")

    if hist.empty:
        return "UNKNOWN", 0, valid_ticker

    #Calcuate trend and pct change
    y = hist['Close'].dropna().values
    x = np.arrange(len(y))
    slope, intercept = np.polyfit(x,y,1)
    trend = "UPWARD" if slope > 0 else "DOWNWARD"
    pct_change = (slope / np.mean(y))*100
    return trend, round(pct_change, 2), valid_ticker

def create_financial_summary(pdf_text):
    prompt = f"""
    Summarize key financial highlights from this balance sheet.
    Focus on:
    - Assets
    - Liabilities
    - Equity
    - Leverage
    - Liquidity
    - Growth indicators

    Text:
    {pdf_text[:6000]}
    """
    return groq_completion(prompt)

def financial_narrative_agent(company, summary, trend, pct_change, ticker):
    prompt = f"""
    You are a financial narrative analyst AI.

    Company: {company}
    NSE Ticker: {ticker}

    Financial Summary:
    {summary}

    Stock Trend (Last 6 Months): {trend}
    Percentage Change: {pct_change}%

    Provide:
    1. Short Term Health Score (0-100)
    2. Long Term Health Score (0-100)
    3. Investment Inference (Short Term)
    4. Investment Inference (Long Term)

    Give concise and professional reasoning.
    """
    return groq_completion(prompt)

# Streamlit UI
st.title("Financial Health Narrative Agent")
st.write("Upload a company financial statement (PDF). The agent will auto-detect company name, analyze financials, fetch NSE stock trend, and generate investment insights.")

uploaded_file = st.file_uploader("Upload Financial Statement PDF", type=["pdf"])

if uploaded_file:
    with open("uploaded_statement.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing PDF and extracting financial text...")
    pdf_text = load_pdf_text("uploaded_statement.pdf")

    st.info("Detecting company name...")
    company_name = extract_company_name(pdf_text)

    if company_name == "NOT_FOUND":
        company_name = st.text_input("Company name not detected. Enter manually:")

    if company_name:
        st.success(f"Detected Company: {company_name}")

        st.info("Fetching NSE stock trend (last 6 months)...")
        trend, pct_change, ticker = fetch_stock_trend(company_name)
        st.write(f"**Ticker:** {ticker}")
        st.write(f"**Trend:** {trend} | **Change:** {pct_change}%")

        st.info("Generating financial summary...")
        summary = create_financial_summary(pdf_text)
        st.text_area("Financial Summary", summary, height=200)

        st.info("Running Narrative Health Analysis...")
        report = financial_narrative_agent(company_name, summary, trend, pct_change, ticker)

        st.subheader("📈 Financial Health & Investment Insights")
        st.write(report)

        st.download_button(
            label="Download Report",
            data=report,
            file_name="financial_health_report.txt",
            mime="text/plain"
        )

