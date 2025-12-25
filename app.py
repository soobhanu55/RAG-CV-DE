import streamlit as st
import os
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_engine import RAGEngine
from cv_parser import CVParser
from neural_classifier import CVClassifier
from api_handler import APIHandler
import tempfile
import json

# Page configuration
st.set_page_config(
    page_title="RAG CV Analysis System - Germany",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'cv_database' not in st.session_state:
    st.session_state.cv_database = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_system():
    """Initialize RAG system components"""
    if st.session_state.rag_engine is None:
        with st.spinner("Initializing RAG System..."):
            api_key = st.session_state.get('api_key', '')
            st.session_state.rag_engine = RAGEngine(api_key=api_key)
            st.success("✅ System initialized successfully!")

def main():
    st.markdown('<h1 class="main-header">🇩🇪 RAG Enterprise CV Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get('api_key', ''),
            help="Enter your OpenAI API key for LLM functionality"
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.divider()
        
        # Model selection
        model_choice = st.selectbox(
            "LLM Model",
            ["gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-4"],
            help="Select the LLM model for analysis"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-ada-002"],
            help="Select embedding model for vector search"
        )
        
        st.divider()
        
        # System stats
        st.header("📊 System Statistics")
        st.metric("CVs in Database", len(st.session_state.cv_database))
        st.metric("Conversations", len(st.session_state.chat_history))
        
        if st.button("🔄 Reset System"):
            st.session_state.rag_engine = None
            st.session_state.cv_database = []
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload CVs",
        "🔍 Search & Query",
        "🤖 AI Analysis",
        "📈 Neural Network Classifier",
        "🔌 API Integration"
    ])
    
    # Tab 1: Upload CVs
    with tab1:
        st.header("Upload German CVs")
        
        uploaded_files = st.file_uploader(
            "Upload CV files (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload one or multiple CV files in German"
        )
        
        if uploaded_files:
            if st.button("Process CVs"):
                initialize_system()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                        tmp_file.write(file.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Parse CV
                        parser = CVParser()
                        cv_data = parser.parse_cv(tmp_path, language='de')
                        
                        # Add to RAG system
                        st.session_state.rag_engine.add_document(cv_data, file.name)
                        st.session_state.cv_database.append({
                            'filename': file.name,
                            'data': cv_data
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                    finally:
                        os.unlink(tmp_path)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ All CVs processed successfully!")
                st.success(f"Processed {len(uploaded_files)} CV(s)")
        
        # Display uploaded CVs
        if st.session_state.cv_database:
            st.subheader("📋 Processed CVs")
            for cv in st.session_state.cv_database:
                with st.expander(f"📄 {cv['filename']}"):
                    st.json(cv['data'])
    
    # Tab 2: Search & Query
    with tab2:
        st.header("Semantic Search & Query")
        
        if not st.session_state.cv_database:
            st.warning("⚠️ Please upload CVs first in the 'Upload CVs' tab")
        else:
            search_query = st.text_input(
                "Enter search query (in German or English)",
                placeholder="e.g., 'Erfahrung mit Python und Machine Learning'"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                top_k = st.slider("Number of results", 1, 10, 3)
            with col2:
                search_button = st.button("🔍 Search", use_container_width=True)
            
            if search_button and search_query:
                initialize_system()
                
                with st.spinner("Searching CVs..."):
                    results = st.session_state.rag_engine.search(search_query, top_k=top_k)
                    
                    st.subheader("Search Results")
                    for idx, result in enumerate(results, 1):
                        with st.expander(f"Result {idx}: {result['filename']} (Score: {result['score']:.4f})"):
                            st.markdown(f"**Content:**\n{result['content']}")
                            st.markdown(f"**Metadata:**")
                            st.json(result['metadata'])
    
    # Tab 3: AI Analysis
    with tab3:
        st.header("AI-Powered CV Analysis")
        
        if not st.session_state.cv_database:
            st.warning("⚠️ Please upload CVs first in the 'Upload CVs' tab")
        else:
            st.subheader("Ask Questions About CVs")
            
            user_question = st.text_area(
                "Your question",
                placeholder="z.B. 'Welche Kandidaten haben Erfahrung mit Deep Learning?'",
                height=100
            )
            
            if st.button("💡 Get AI Answer"):
                if not st.session_state.get('api_key'):
                    st.error("❌ Please enter your API key in the sidebar")
                elif user_question:
                    initialize_system()
                    
                    with st.spinner("Generating AI response..."):
                        response = st.session_state.rag_engine.query(user_question)
                        
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': response
                        })
                        
                        st.markdown("### 🤖 AI Response")
                        st.markdown(f"<div class='metric-card'>{response['answer']}</div>", unsafe_allow_html=True)
                        
                        if response.get('sources'):
                            with st.expander("📚 Sources"):
                                for source in response['sources']:
                                    st.markdown(f"- **{source['filename']}** (Relevance: {source['score']:.2%})")
            
            # Chat history
            if st.session_state.chat_history:
                st.divider()
                st.subheader("💬 Conversation History")
                for idx, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                    with st.expander(f"Q{idx}: {chat['question'][:50]}..."):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown(f"**Answer:** {chat['answer']['answer']}")
    
    # Tab 4: Neural Network Classifier
    with tab4:
        st.header("Neural Network-Based CV Classification")
        
        if not st.session_state.cv_database:
            st.warning("⚠️ Please upload CVs first in the 'Upload CVs' tab")
        else:
            st.markdown("""
            The neural network classifier categorizes CVs based on:
            - **Job Category** (IT, Engineering, Management, etc.)
            - **Experience Level** (Junior, Mid-level, Senior)
            - **Skill Match Score** for specific requirements
            """)
            
            if st.button("🧠 Classify All CVs"):
                with st.spinner("Running neural network classification..."):
                    classifier = CVClassifier()
                    
                    results_data = []
                    for cv in st.session_state.cv_database:
                        classification = classifier.classify(cv['data'])
                        results_data.append({
                            'CV': cv['filename'],
                            'Category': classification['category'],
                            'Level': classification['experience_level'],
                            'Confidence': f"{classification['confidence']:.2%}"
                        })
                    
                    st.subheader("Classification Results")
                    st.table(results_data)
                    
                    # Visualization
                    st.subheader("📊 Distribution Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        categories = [r['Category'] for r in results_data]
                        st.markdown("**Job Categories**")
                        category_counts = {cat: categories.count(cat) for cat in set(categories)}
                        st.bar_chart(category_counts)
                    
                    with col2:
                        levels = [r['Level'] for r in results_data]
                        st.markdown("**Experience Levels**")
                        level_counts = {lvl: levels.count(lvl) for lvl in set(levels)}
                        st.bar_chart(level_counts)
    
    # Tab 5: API Integration
    with tab5:
        st.header("API Integration & Enterprise Endpoints")
        
        st.markdown("""
        ### Available REST API Endpoints
        
        This system provides RESTful APIs for enterprise integration:
        """)
        
        api_handler = APIHandler()
        
        endpoints = [
            {
                "Method": "POST",
                "Endpoint": "/api/v1/upload",
                "Description": "Upload and process CV files"
            },
            {
                "Method": "GET",
                "Endpoint": "/api/v1/search",
                "Description": "Semantic search across CVs"
            },
            {
                "Method": "POST",
                "Endpoint": "/api/v1/query",
                "Description": "AI-powered CV analysis"
            },
            {
                "Method": "POST",
                "Endpoint": "/api/v1/classify",
                "Description": "Neural network classification"
            },
            {
                "Method": "GET",
                "Endpoint": "/api/v1/cvs",
                "Description": "List all processed CVs"
            }
        ]
        
        st.table(endpoints)
        
        st.subheader("🧪 Test API Functionality")
        
        test_endpoint = st.selectbox("Select endpoint to test", [e["Endpoint"] for e in endpoints])
        
        if "search" in test_endpoint:
            test_query = st.text_input("Search query", "Python Machine Learning")
            if st.button("Test Search API"):
                result = api_handler.test_search(test_query, st.session_state.cv_database)
                st.json(result)
        
        elif "query" in test_endpoint:
            test_query = st.text_area("Query", "Welche Kandidaten haben AI Erfahrung?")
            if st.button("Test Query API"):
                if st.session_state.get('api_key'):
                    initialize_system()
                    result = api_handler.test_query(test_query, st.session_state.rag_engine)
                    st.json(result)
                else:
                    st.error("API key required")
        
        st.divider()
        
        st.subheader("📝 API Documentation")
        st.markdown("""
        **Authentication:** Bearer token required in header
        ```
        Authorization: Bearer YOUR_API_KEY
        ```
        
        **Example cURL Request:**
        ```
        curl -X POST https://your-api.com/api/v1/search \
          -H "Authorization: Bearer YOUR_KEY" \
          -H "Content-Type: application/json" \
          -d '{"query": "Python Developer", "top_k": 5}'
        ```
        """)

if __name__ == "__main__":
    main()
