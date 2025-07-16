# frontend/app.py

import streamlit as st
import requests
import json
import pandas as pd
import io
from datetime import datetime
import os
import logging

# Page Configuration
st.set_page_config(
    page_title="AI Meeting Insights",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Session State Initialization
def init_session_state():
    for key in ['input_method', 'processed_result']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'input_method' else "Upload File"

# Check Backend Status
def is_backend_running():
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Backend connection error: {e}")
        return False

# API Call to Process Meeting
def process_meeting_data(uploaded_file):
    try:
        if hasattr(uploaded_file, 'getvalue'):
            file_content = uploaded_file.getvalue()
            filename = uploaded_file.name
            files = {'file': (filename, file_content, uploaded_file.type)}
            data = {}
        else:
            files = None
            data = {'transcript_text': uploaded_file}

        response = requests.post(
            f"{BACKEND_URL}/process_meeting/",
            files=files,
            data=data,
            timeout=300
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        logger.error(f"API Request Error: {e}")
        raise
    except Exception as e:
        st.error(f"Processing error: {e}")
        logger.error(f"Processing error: {e}")
        raise

# Results Display
def display_results(result):
    st.markdown("## 📊 Meeting Summary")
    st.write(result.get('summary', {}).get('summary', 'No summary available'))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Key Discussion Topics")
        for topic in result.get('summary', {}).get('discussion_topics', []):
            st.markdown(f"- {topic}")

    with col2:
        st.markdown("### 📋 Action Items")
        for item in result.get('crm_insights', {}).get('action_items', []):
            st.markdown(f"- {item}")

    st.markdown("## 🔍 Detailed Insights")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 😣 Client Pain Points")
        for point in result.get('crm_insights', {}).get('pain_points', []):
            st.markdown(f"- {point}")

        st.markdown("### ❓ Objections")
        for obj in result.get('crm_insights', {}).get('objections', []):
            st.markdown(f"- {obj}")

    with col4:
        st.markdown("### ✅ Resolutions")
        for res in result.get('crm_insights', {}).get('resolutions', []):
            st.markdown(f"- {res}")

    # Export Section
    with st.expander("📤 Export Options"):
        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.download_button(
                "Export as JSON",
                data=json.dumps(result, indent=4),
                file_name=f"meeting_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with export_col2:
            csv_data = {
                "Summary": [result.get('summary', {}).get('summary', '')],
                "Discussion Topics": [', '.join(result.get('summary', {}).get('discussion_topics', []))],
                "Pain Points": [', '.join(result.get('crm_insights', {}).get('pain_points', []))],
                "Objections": [', '.join(result.get('crm_insights', {}).get('objections', []))],
                "Resolutions": [', '.join(result.get('crm_insights', {}).get('resolutions', []))],
                "Action Items": [', '.join(result.get('crm_insights', {}).get('action_items', []))]
            }
            st.download_button(
                "Export as CSV",
                data=pd.DataFrame(csv_data).to_csv(index=False),
                file_name=f"meeting_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Sidebar Search
def sidebar():
    st.sidebar.title("🔍 Past Meetings Search")

    search_query = st.sidebar.text_input("Search Keywords")

    if st.sidebar.button("Search Meetings"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/search_past_meetings/",
                json={"query": search_query}
            )
            response.raise_for_status()
            results = response.json().get('results', [])

            if results:
                st.sidebar.success(f"Found {len(results)} meetings")
                for i, result in enumerate(results, 1):
                    with st.sidebar.expander(f"Result {i}"):
                        st.write(result.get('content', 'No details available'))
            else:
                st.sidebar.info("No meetings found. Try different keywords.")
        except Exception as e:
            st.sidebar.error(f"Search failed: {e}")

# Main App Logic
def main():
    init_session_state()
    st.title("🤝 AI Meeting Insights")
    st.markdown("### Intelligent Meeting Summary & CRM Note Generator")

    backend_status = is_backend_running()
    if not backend_status:
        st.error("❌ Backend server is not running. Please start the server.")
        return

    st.markdown("---")

    tab1, tab2 = st.tabs(["📥 Input Meeting", "📊 Insights Output"])

    with tab1:
        st.markdown("#### 📝 Meeting Input")

        input_method = st.radio(
            "Choose Input Method", 
            ["Upload Audio/Text", "Paste Transcript"],
            horizontal=True
        )

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

        process_button = st.button(
            "🔍 Generate Insights", 
            disabled=not uploaded_file
        )

        if process_button and uploaded_file:
            with st.spinner("Analyzing meeting data..."):
                try:
                    result = process_meeting_data(uploaded_file)
                    st.session_state.processed_result = result
                    st.success("✅ Insights generated successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        if st.session_state.get("processed_result"):
            display_results(st.session_state["processed_result"])
        else:
            st.info("👈 Start by uploading a file or transcript in the Input tab.")

# Run App
if __name__ == "__main__":
    main()
    sidebar()
