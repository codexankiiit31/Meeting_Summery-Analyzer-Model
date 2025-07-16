import streamlit as st
import requests
import json
import pandas as pd
import io
from datetime import datetime

# Initialize session state variables
if 'input_method' not in st.session_state:
    st.session_state.input_method = "Upload File"
if 'processed_result' not in st.session_state:
    st.session_state.processed_result = None

# Backend URL
BACKEND_URL = "http://localhost:8000"

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Root Variables for Color Palette */
    :root {
        --primary-color: #3B82F6;      /* Vibrant Blue */
        --secondary-color: #10B981;    /* Emerald Green */
        --background-color: #F3F4F6;   /* Light Gray */
        --text-color: #1F2937;         /* Dark Gray */
        --card-background: #FFFFFF;    /* White */
    }

    /* Global Styling */
    body {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--background-color) !important;
        color: var(--text-color);
    }

    /* Container Styling */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem;
        background-color: var(--background-color);
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        color: var(--primary-color);
        font-weight: 700;
    }

    /* Card Styling */
    .stCard {
        background-color: var(--card-background);
        border-radius: 12px;
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }

    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 20px rgba(0,0,0,0.12);
    }

    /* Button Styling */
    .stButton>button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #2563EB !important;
        transform: scale(1.05);
    }

    /* Input Styling */
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        padding: 10px;
    }

    /* File Uploader Styling */
    .stFileUploader>div>button {
        background-color: var(--secondary-color) !important;
        color: white !important;
        border-radius: 8px;
    }

    /* Badge Styling */
    .badge {
        background-color: rgba(59, 130, 246, 0.1);
        color: var(--primary-color);
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 5px;
    }

    /* Sidebar Styling */
    .css-1aumxhk {
        background-color: var(--card-background);
    }

    /* Spinner Styling */
    .stSpinner > div {
        border-color: var(--primary-color) transparent var(--primary-color) transparent;
    }
</style>
""", unsafe_allow_html=True)

# Backend Status Check Function
def is_backend_running():
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

# Page Configuration
st.set_page_config(
    page_title="AI Meeting Insights",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main App Layout
def main():
    # Header
    st.markdown("# 🤝 AI Meeting Insights")
    st.markdown("### Intelligent Meeting Summary & CRM Note Generator")
    
    # Backend Status Check
    backend_status = is_backend_running()
    if not backend_status:
        st.error("❌ Backend server is not running. Please start the server.")
    
    # Input Section
    st.markdown("## 📝 Meeting Input")
    
    # Input Method Selection
    input_method = st.radio(
        "Choose Input Method", 
        ["Upload Audio/Text", "Paste Transcript"],
        horizontal=True
    )
    
    # File or Text Input
    if input_method == "Upload Audio/Text":
        uploaded_file = st.file_uploader(
            "Upload Meeting Recording or Transcript", 
            type=['mp3', 'wav', 'txt'],
            help="Supported formats: MP3, WAV, TXT"
        )
    else:
        uploaded_file = st.text_area(
            "Paste Meeting Transcript", 
            height=200,
            placeholder="Paste your meeting transcript here..."
        )
    
    # Process Button
    process_button = st.button(
        "Generate Insights", 
        disabled=not backend_status or not uploaded_file
    )
    
    # Processing Logic
    if process_button:
        with st.spinner("Analyzing meeting data..."):
            try:
                # API Call Logic (Similar to your existing code)
                response = process_meeting_data(uploaded_file)
                display_results(response)
            except Exception as e:
                st.error(f"Error processing meeting: {e}")

def process_meeting_data(uploaded_file):
    # Implement your existing API call logic here
    # Return the processed data
    pass

def display_results(result):
    # Create sections for Summary, Insights, and Actions
    st.markdown("## 📊 Meeting Summary")
    st.write(result['summary']['summary'])
    
    # Insights Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Key Topics")
        for topic in result['summary']['discussion_topics']:
            st.markdown(f"- {topic}")
    
    with col2:
        st.markdown("### 📋 Action Items")
        for item in result['crm_insights']['action_items']:
            st.markdown(f"- {item}")
    
    # Export Options
    st.markdown("## 📤 Export Options")
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        st.download_button(
            "Export JSON", 
            data=json.dumps(result, indent=4),
            file_name="meeting_insights.json",
            mime="application/json"
        )
    
    with export_col2:
        st.download_button(
            "Export CSV", 
            data=pd.DataFrame(result).to_csv(),
            file_name="meeting_insights.csv",
            mime="text/csv"
        )

# Sidebar
def sidebar():
    st.sidebar.title("🔍 Past Meetings")
    search_query = st.sidebar.text_input("Search Past Meetings")
    
    if st.sidebar.button("Search"):
        # Implement search logic
        pass

# Run the app
if __name__ == "__main__":
    main()
    sidebar()