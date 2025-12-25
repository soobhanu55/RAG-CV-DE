from typing import Dict, Any, List
import json
from datetime import datetime

class APIHandler:
    """
    REST API Handler for enterprise integration
    Provides endpoints for external systems
    """
    
    def __init__(self):
        self.api_version = "v1"
        self.base_url = "/api/v1"
    
    def test_search(self, query: str, cv_database: List[Dict]) -> Dict[str, Any]:
        """Test search API endpoint"""
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results": []
        }
        
        # Simple keyword search for testing
        for cv in cv_database[:3]:
            if query.lower() in str(cv.get('data', {})).lower():
                response["results"].append({
                    "filename": cv['filename'],
                    "match_score": 0.85,
                    "snippet": str(cv['data'])[:200]
                })
        
        response["total_results"] = len(response["results"])
        return response
    
    def test_query(self, query: str, rag_engine: Any) -> Dict[str, Any]:
        """Test query API endpoint"""
        try:
            result = rag_engine.query(query)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "answer": result.get('answer', ''),
                "sources": [
                    {
                        "filename": s['filename'],
                        "relevance_score": s['score']
                    } for s in result.get('sources', [])[:3]
                ]
            }
        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def format_response(self, data: Any, status_code: int = 200) -> Dict[str, Any]:
        """Format API response"""
        return {
            "status_code": status_code,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "api_version": self.api_version
        }
    
    def get_api_documentation(self) -> Dict[str, Any]:
        """Return API documentation"""
        return {
            "api_version": self.api_version,
            "endpoints": {
                f"{self.base_url}/upload": {
                    "method": "POST",
                    "description": "Upload CV files for processing",
                    "parameters": {
                        "file": "multipart/form-data",
                        "language": "string (default: 'de')"
                    }
                },
                f"{self.base_url}/search": {
                    "method": "GET",
                    "description": "Semantic search across CVs",
                    "parameters": {
                        "query": "string (required)",
                        "top_k": "integer (default: 5)"
                    }
                },
                f"{self.base_url}/query": {
                    "method": "POST",
                    "description": "AI-powered CV query",
                    "parameters": {
                        "question": "string (required)",
                        "context_size": "integer (default: 3)"
                    }
                },
                f"{self.base_url}/classify": {
                    "method": "POST",
                    "description": "Classify CV using neural network",
                    "parameters": {
                        "cv_id": "string (required)"
                    }
                }
            }
        }
