import streamlit as st
import tempfile
from pathlib import Path
import re
import json
from datetime import datetime
import PyPDF2
from docx import Document

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cv_database' not in st.session_state:
    st.session_state.cv_database = []

# CV Parser Class (Lightweight)
class CVParser:
    def __init__(self):
        self.keywords = {
            'skills': ['fähigkeiten', 'skills', 'kenntnisse', 'kompetenzen'],
            'experience': ['berufserfahrung', 'experience', 'arbeitserfahrung'],
            'education': ['ausbildung', 'education', 'bildung', 'studium'],
        }
    
    def parse_cv(self, file_path, language='de'):
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            text = self._extract_pdf(file_path)
        elif file_ext == '.docx':
            text = self._extract_docx(file_path)
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = "Unsupported format"
        
        return self._parse_text(text)
    
    def _extract_pdf(self, pdf_path):
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except:
            text = "Error extracting PDF"
        return text
    
    def _extract_docx(self, docx_path):
        try:
            doc = Document(docx_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        except:
            text = "Error extracting DOCX"
        return text
    
    def _parse_text(self, text):
        return {
            'personal_info': self._extract_personal(text),
            'skills': self._extract_skills(text),
            'experience': self._extract_experience(text),
            'education': self._extract_education(text),
            'raw_text': text
        }
    
    def _extract_personal(self, text):
        info = {}
        lines = text.split('\n')
        if lines:
            info['name'] = lines[0].strip()
        
        email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email:
            info['email'] = email.group()
        
        phone = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', text)
        if phone:
            info['phone'] = phone.group()
        
        return info
    
    def _extract_skills(self, text):
        tech_skills = ['python', 'java', 'javascript', 'c++', 'sql', 'machine learning',
                      'deep learning', 'tensorflow', 'pytorch', 'docker', 'kubernetes']
        
        skills = [skill.title() for skill in tech_skills if skill in text.lower()]
        return skills[:15]
    
    def _extract_experience(self, text):
        experiences = []
        date_pattern = r'(\d{2}[-/.]\d{4}|\d{4})\s*[-–]\s*(\d{2}[-/.]\d{4}|\d{4}|present|heute)'
        matches = re.finditer(date_pattern, text, re.IGNORECASE)
        
        for match in list(matches)[:5]:
            context = text[max(0, match.start()-100):match.start()+300]
            experiences.append({
                'duration': match.group(),
                'description': context[:200]
            })
        
        return experiences
    
    def _extract_education(self, text):
        education = []
        degrees = ['bachelor', 'master', 'phd', 'diploma', 'doktor']
        
        for degree in degrees:
            if degree in text.lower():
                idx = text.lower().index(degree)
                education.append({
                    'degree': degree.title(),
                    'context': text[idx:idx+150]
                })
        
        return education[:3]

# Simple Search Function
def simple_search(query, cv_database):
    results = []
    query_lower = query.lower()
    
    for cv in cv_database:
        text = cv['data']['raw_text'].lower()
        if query_lower in text:
            score = text.count(query_lower) / max(len(text.split()), 1)
            results.append({
                'filename': cv['filename'],
                'score': score,
                'snippet': cv['data']['raw_text'][:300]
            })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

# Main App
def main():
    st.markdown('<h1 class="main-header">🇩🇪 RAG CV Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Info")
        st.metric("CVs in Database", len(st.session_state.cv_database))
        
        st.info("💡 Lightweight version - Optimized for Streamlit Cloud")
        
        if st.button("🔄 Reset System"):
            st.session_state.cv_database = []
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload CVs",
        "🔍 Search",
        "📊 Analytics",
        "🔌 API Info"
    ])
    
    # Tab 1: Upload
    with tab1:
        st.header("Upload German CVs")
        
        uploaded_files = st.file_uploader(
            "Upload CV files (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process CVs"):
            parser = CVParser()
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                
                try:
                    cv_data = parser.parse_cv(tmp_path, 'de')
                    st.session_state.cv_database.append({
                        'filename': file.name,
                        'data': cv_data,
                        'uploaded': datetime.now().isoformat()
                    })
                except Exception as e:
                    st.error(f"Error: {file.name} - {str(e)}")
                finally:
                    Path(tmp_path).unlink()
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            st.success(f"✅ Processed {len(uploaded_files)} CV(s)")
        
        # Display CVs
        if st.session_state.cv_database:
            st.subheader("📋 Processed CVs")
            for cv in st.session_state.cv_database:
                with st.expander(f"📄 {cv['filename']}"):
                    data = cv['data']
                    st.markdown(f"**Name:** {data['personal_info'].get('name', 'N/A')}")
                    st.markdown(f"**Email:** {data['personal_info'].get('email', 'N/A')}")
                    st.markdown(f"**Skills:** {', '.join(data['skills'][:10])}")
                    st.markdown(f"**Experience:** {len(data['experience'])} positions found")
                    st.markdown(f"**Education:** {len(data['education'])} entries found")
    
    # Tab 2: Search
    with tab2:
        st.header("Search CVs")
        
        if not st.session_state.cv_database:
            st.warning("⚠️ Please upload CVs first")
        else:
            search_query = st.text_input(
                "Search query",
                placeholder="e.g., Python, Machine Learning, Berlin"
            )
            
            if st.button("🔍 Search") and search_query:
                results = simple_search(search_query, st.session_state.cv_database)
                
                st.subheader(f"Found {len(results)} Results")
                
                for idx, result in enumerate(results[:10], 1):
                    with st.expander(f"Result {idx}: {result['filename']} (Score: {result['score']:.4f})"):
                        st.markdown(result['snippet'])
    
    # Tab 3: Analytics
    with tab3:
        st.header("CV Analytics")
        
        if not st.session_state.cv_database:
            st.warning("⚠️ Please upload CVs first")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total CVs", len(st.session_state.cv_database))
            
            with col2:
                total_skills = sum(len(cv['data']['skills']) for cv in st.session_state.cv_database)
                st.metric("Total Skills", total_skills)
            
            with col3:
                avg_exp = sum(len(cv['data']['experience']) for cv in st.session_state.cv_database) / max(len(st.session_state.cv_database), 1)
                st.metric("Avg. Positions", f"{avg_exp:.1f}")
            
            st.subheader("📊 Skill Distribution")
            
            # Collect all skills
            all_skills = {}
            for cv in st.session_state.cv_database:
                for skill in cv['data']['skills']:
                    all_skills[skill] = all_skills.get(skill, 0) + 1
            
            if all_skills:
                sorted_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:10]
                st.bar_chart({skill: count for skill, count in sorted_skills})
            
            st.subheader("📋 CV Summary Table")
            
            table_data = []
            for cv in st.session_state.cv_database:
                table_data.append({
                    "CV": cv['filename'],
                    "Name": cv['data']['personal_info'].get('name', 'N/A'),
                    "Skills": len(cv['data']['skills']),
                    "Experience": len(cv['data']['experience']),
                    "Education": len(cv['data']['education'])
                })
            
            st.table(table_data)
    
    # Tab 4: API Info
    with tab4:
        st.header("API Integration")
        
        st.markdown("""
        ### Available REST API Endpoints
        
        This system can be extended with the following APIs:
        """)
        
        endpoints = [
            {"Method": "POST", "Endpoint": "/api/v1/upload", "Description": "Upload CV files"},
            {"Method": "GET", "Endpoint": "/api/v1/search", "Description": "Search CVs"},
            {"Method": "GET", "Endpoint": "/api/v1/cvs", "Description": "List all CVs"},
            {"Method": "GET", "Endpoint": "/api/v1/analytics", "Description": "Get analytics"},
        ]
        
        st.table(endpoints)
        
        st.markdown("""
        ### Export Data
        
        Download your CV database:
        """)
        
        if st.session_state.cv_database:
            json_data = json.dumps(st.session_state.cv_database, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download CV Database (JSON)",
                data=json_data,
                file_name="cv_database.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
