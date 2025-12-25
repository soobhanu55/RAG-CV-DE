import re
from typing import Dict, Any, List
from pathlib import Path
import PyPDF2
from docx import Document

class CVParser:
    """
    CV Parser for German CVs
    Supports PDF, DOCX, and TXT formats
    """
    
    def __init__(self):
        # German CV keywords
        self.keywords = {
            'skills': ['fähigkeiten', 'skills', 'kenntnisse', 'kompetenzen'],
            'experience': ['berufserfahrung', 'experience', 'arbeitserfahrung', 'beruflicher werdegang'],
            'education': ['ausbildung', 'education', 'bildung', 'studium'],
            'personal': ['persönliche daten', 'personal information', 'kontakt']
        }
        
    def parse_cv(self, file_path: str, language: str = 'de') -> Dict[str, Any]:
        """Main parsing function"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            text = self._extract_pdf(file_path)
        elif file_ext == '.docx':
            text = self._extract_docx(file_path)
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return self._parse_text(text, language)
    
    def _extract_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            text = f"Error extracting PDF: {str(e)}"
        return text
    
    def _extract_docx(self, docx_path: str) -> str:
        """Extract text from DOCX"""
        try:
            doc = Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            text = f"Error extracting DOCX: {str(e)}"
        return text
    
    def _parse_text(self, text: str, language: str) -> Dict[str, Any]:
        """Parse CV text into structured data"""
        cv_data = {
            'personal_info': self._extract_personal_info(text),
            'summary': self._extract_summary(text),
            'skills': self._extract_skills(text),
            'experience': self._extract_experience(text),
            'education': self._extract_education(text),
            'total_experience_years': self._calculate_experience(text),
            'raw_text': text
        }
        
        return cv_data
    
    def _extract_personal_info(self, text: str) -> Dict[str, str]:
        """Extract personal information"""
        info = {}
        
        # Extract name (typically at the start)
        lines = text.split('\n')
        if lines:
            info['name'] = lines[0].strip()
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            info['email'] = email_match.group()
        
        # Extract phone
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            info['phone'] = phone_match.group()
        
        return info
    
    def _extract_summary(self, text: str) -> str:
        """Extract professional summary"""
        summary_keywords = ['zusammenfassung', 'profil', 'summary', 'profile', 'über mich']
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in summary_keywords):
                # Get next few lines as summary
                summary_lines = lines[i+1:i+5]
                return ' '.join([l.strip() for l in summary_lines if l.strip()])
        
        return text[:200]  # Default: first 200 chars
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills"""
        skills = []
        text_lower = text.lower()
        
        # Common technical skills
        tech_skills = [
            'python', 'java', 'javascript', 'c++', 'sql', 'machine learning',
            'deep learning', 'neural networks', 'tensorflow', 'pytorch',
            'docker', 'kubernetes', 'aws', 'azure', 'git', 'agile',
            'künstliche intelligenz', 'maschinelles lernen', 'datenanalyse'
        ]
        
        for skill in tech_skills:
            if skill in text_lower:
                skills.append(skill.title())
        
        # Extract from skills section
        for keyword in self.keywords['skills']:
            if keyword in text_lower:
                idx = text_lower.index(keyword)
                section = text[idx:idx+500]
                # Extract words that look like skills
                words = re.findall(r'\b[A-Za-z+#.-]+\b', section)
                skills.extend([w for w in words if len(w) > 2])[:10]
                break
        
        return list(set(skills))[:15]
    
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience"""
        experiences = []
        
        # Look for date patterns
        date_pattern = r'(\d{2}[-/.]\d{4}|\d{4})\s*[-–—]\s*(\d{2}[-/.]\d{4}|\d{4}|present|heute|current)'
        date_matches = re.finditer(date_pattern, text, re.IGNORECASE)
        
        for match in date_matches:
            start_pos = match.start()
            context = text[max(0, start_pos-100):start_pos+300]
            
            experience = {
                'duration': match.group(),
                'title': self._extract_job_title(context),
                'company': self._extract_company(context),
                'description': context[:200]
            }
            experiences.append(experience)
        
        return experiences[:5]
    
    def _extract_job_title(self, text: str) -> str:
        """Extract job title from context"""
        common_titles = [
            'developer', 'engineer', 'manager', 'analyst', 'consultant',
            'entwickler', 'ingenieur', 'manager', 'analyst', 'berater'
        ]
        
        text_lower = text.lower()
        for title in common_titles:
            if title in text_lower:
                idx = text_lower.index(title)
                return text[max(0, idx-20):idx+30].strip()
        
        return "Position"
    
    def _extract_company(self, text: str) -> str:
        """Extract company name"""
        # Look for patterns like "at Company" or "bei Firma"
        patterns = [r'bei\s+([A-Z][A-Za-z\s&]+)', r'at\s+([A-Z][A-Za-z\s&]+)']
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return "Company"
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information"""
        education = []
        
        # Common degrees
        degrees = [
            'bachelor', 'master', 'phd', 'diploma', 'mba',
            'bachelor', 'master', 'doktor', 'diplom'
        ]
        
        text_lower = text.lower()
        for degree in degrees:
            if degree in text_lower:
                idx = text_lower.index(degree)
                context = text[idx:idx+200]
                
                education.append({
                    'degree': degree.title(),
                    'institution': self._extract_institution(context),
                    'year': self._extract_year(context)
                })
        
        return education[:3]
    
    def _extract_institution(self, text: str) -> str:
        """Extract university/institution name"""
        keywords = ['universität', 'university', 'hochschule', 'college', 'institut']
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                idx = text_lower.index(keyword)
                return text[max(0, idx-10):idx+50].strip()
        
        return "Educational Institution"
    
    def _extract_year(self, text: str) -> str:
        """Extract year from text"""
        year_pattern = r'\b(19|20)\d{2}\b'
        match = re.search(year_pattern, text)
        return match.group() if match else "Year unknown"
    
    def _calculate_experience(self, text: str) -> int:
        """Calculate total years of experience"""
        # Find all years mentioned
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        if years:
            years_int = [int(y) for y in years]
            return max(years_int) - min(years_int)
        return 0
