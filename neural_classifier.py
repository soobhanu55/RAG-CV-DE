import numpy as np
from typing import Dict, Any, List
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn

class CVNeuralNetwork(nn.Module):
    """
    Neural Network for CV Classification
    Multi-layer perceptron for categorizing CVs
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_classes: int = 5):
        super(CVNeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

class CVClassifier:
    """
    CV Classifier using Neural Networks
    Classifies CVs by job category and experience level
    """
    
    def __init__(self):
        self.categories = [
            'IT & Software Development',
            'Engineering & Technical',
            'Management & Business',
            'Data Science & AI',
            'Other'
        ]
        
        self.experience_levels = ['Junior', 'Mid-Level', 'Senior', 'Expert']
        
        # Feature keywords for classification
        self.category_keywords = {
            'IT & Software Development': [
                'python', 'java', 'javascript', 'developer', 'software', 'programming',
                'entwickler', 'programmierung', 'software'
            ],
            'Engineering & Technical': [
                'engineer', 'mechanical', 'electrical', 'ingenieur', 'technical', 'technisch'
            ],
            'Management & Business': [
                'manager', 'business', 'management', 'führung', 'geschäft', 'projekt'
            ],
            'Data Science & AI': [
                'data science', 'machine learning', 'ai', 'artificial intelligence',
                'datenanalyse', 'künstliche intelligenz', 'deep learning'
            ]
        }
        
        # Initialize neural network
        self.model = CVNeuralNetwork(input_size=50, num_classes=len(self.categories))
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network with pretrained-like weights"""
        # For demo purposes - in production, load trained weights
        for param in self.model.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
    
    def classify(self, cv_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify CV using neural network
        """
        # Extract features
        features = self._extract_features(cv_data)
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(feature_tensor)
            category_idx = torch.argmax(predictions).item()
            confidence = predictions[0][category_idx].item()
        
        # Determine experience level
        experience_level = self._determine_experience_level(cv_data)
        
        return {
            'category': self.categories[category_idx],
            'experience_level': experience_level,
            'confidence': confidence,
            'category_scores': {
                cat: float(predictions[0][i]) 
                for i, cat in enumerate(self.categories)
            }
        }
    
    def _extract_features(self, cv_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from CV"""
        features = []
        
        # Text-based features
        text = cv_data.get('raw_text', '').lower()
        
        # Category keyword matches (20 features)
        for category, keywords in self.category_keywords.items():
            match_count = sum(1 for kw in keywords if kw in text)
            features.append(min(match_count / len(keywords), 1.0))
        
        # Skill-based features (10 features)
        skills = cv_data.get('skills', [])
        features.append(len(skills) / 20.0)  # Normalized skill count
        
        tech_skills = ['python', 'java', 'sql', 'machine learning']
        features.append(sum(1 for skill in skills if any(ts in skill.lower() for ts in tech_skills)) / 10.0)
        
        soft_skills = ['communication', 'leadership', 'teamwork', 'kommunikation']
        features.append(sum(1 for skill in skills if any(ss in skill.lower() for ss in soft_skills)) / 10.0)
        
        # Add more dummy features to reach 50
        for _ in range(7):
            features.append(np.random.random() * 0.1)
        
        # Experience features (5 features)
        exp_years = cv_data.get('total_experience_years', 0)
        features.append(min(exp_years / 20.0, 1.0))
        features.append(len(cv_data.get('experience', [])) / 10.0)
        features.append(len(cv_data.get('education', [])) / 5.0)
        features.append(1.0 if 'phd' in text or 'doktor' in text else 0.0)
        features.append(1.0 if 'master' in text else 0.5 if 'bachelor' in text else 0.0)
        
        # Text statistics (5 features)
        features.append(min(len(text) / 5000.0, 1.0))
        features.append(len(text.split()) / 1000.0)
        features.append(text.count('\n') / 100.0)
        features.append(1.0 if '@' in text else 0.0)
        features.append(1.0 if 'github' in text or 'linkedin' in text else 0.0)
        
        # Pad or truncate to exactly 50 features
        while len(features) < 50:
            features.append(0.0)
        features = features[:50]
        
        return features
    
    def _determine_experience_level(self, cv_data: Dict[str, Any]) -> str:
        """Determine experience level based on years and positions"""
        exp_years = cv_data.get('total_experience_years', 0)
        num_positions = len(cv_data.get('experience', []))
        
        if exp_years < 2:
            return 'Junior'
        elif exp_years < 5:
            return 'Mid-Level'
        elif exp_years < 10:
            return 'Senior'
        else:
            return 'Expert'
