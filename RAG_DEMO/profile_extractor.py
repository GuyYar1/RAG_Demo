import re
import logging

logger = logging.getLogger(__name__)

class UserProfile:
    def __init__(self):
        self.age = None
        self.gender = None
        self.medical_history = []
        self.severity_level = None
        self.condition = None
        
    def to_dict(self):
        return {
            'age': self.age,
            'gender': self.gender,
            'medical_history': self.medical_history,
            'severity_level': self.severity_level,
            'condition': self.condition
        }
    
    def __str__(self):
        return f"Profile(age={self.age}, gender={self.gender}, conditions={self.medical_history}, severity={self.severity_level})"


def extract_profile_from_query(query: str, conversation_history: list = None) -> UserProfile:
    """
    Extract user profile information from query and conversation history.
    """
    profile = UserProfile()
    query_lower = query.lower()
    
    # Extract age (handle typos like "yers")
    age_patterns = [
        r'(?:i am|i\'m|im)\s+(\d{1,3})\s+(?:years?|yers?)\s+old',
        r'(\d{1,3})\s+(?:years?|yers?)\s+old',
        r'(\d{1,3})\s*y\.?o\.?',
        r'age\s*[:\-]?\s*(\d{1,3})'
    ]
    for pattern in age_patterns:
        match = re.search(pattern, query_lower)
        if match:
            age = int(match.group(1))
            if 0 < age < 120:  # Sanity check
                profile.age = age
                logger.info(f"Extracted age: {age}")
                break
    
    # Extract gender
    gender_patterns = {
        'male': r'\b(?:male|man|he|his|him|mr\.?)\b',
        'female': r'\b(?:female|woman|she|her|hers|ms\.?|mrs\.?)\b'
    }
    for gender, pattern in gender_patterns.items():
        if re.search(pattern, query_lower):
            profile.gender = gender
            logger.info(f"Extracted gender: {gender}")
            break
    
    # Extract medical conditions
    condition_keywords = {
    'diabetes': ['diabetes', 'diabetic', 'diabities', 'diabeties', 'type 1', 'type 2', 't1d', 't2d'],
    'hypertension': ['hypertension', 'high blood pressure', 'hbp'],
    'diabetic_retinopathy': ['diabetic retinopathy', 'retinopathy', 'DR', 'NPDR', 'PDR'],
    'macular_edema': ['macular edema', 'DME', 'edema']
    }
    
    for condition, keywords in condition_keywords.items():
        for keyword in keywords:
            if keyword.lower() in query_lower:
                if condition not in profile.medical_history:
                    profile.medical_history.append(condition)
                    logger.info(f"Extracted condition: {condition}")
    
    # Extract severity level
    severity_patterns = [
        r'(?:classification\s*)?level\s*(\d+)\s*(?:of|out of|/)\s*(\d+)',
        r'(?:grade|stage|severity)\s*(\d+)',
        r'\b(mild|moderate|severe|proliferative)\b'
    ]
    
    severity_mapping = {
        'mild': 1,
        'moderate': 2,
        'severe': 3,
        'proliferative': 4,
        'advanced': 4
    }
    
    for pattern in severity_patterns:
        match = re.search(pattern, query_lower)
        if match:
            if 'mild' in match.group(0) or 'moderate' in match.group(0) or 'severe' in match.group(0) or 'proliferative' in match.group(0):
                severity_word = match.group(1).lower()
                profile.severity_level = severity_mapping.get(severity_word, None)
                profile.condition = f"{severity_word.capitalize()} Diabetic Retinopathy"
            else:
                try:
                    level = int(match.group(1))
                    total = int(match.group(2)) if len(match.groups()) > 1 else 4
                    profile.severity_level = level
                    profile.condition = f"Level {level}/{total} Diabetic Retinopathy"
                    logger.info(f"Extracted severity: Level {level}/{total}")
                except (ValueError, IndexError):
                    pass
            break
    
    # Check for severe keywords even if no explicit level mentioned
    if profile.severity_level is None:
        if any(term in query_lower for term in ['severe', 'advanced', 'proliferative', 'pdr']):
            profile.severity_level = 3
            profile.condition = "Severe Diabetic Retinopathy"
            logger.info("Inferred severe condition from keywords")
    
    # Extract from conversation history if available
    if conversation_history and profile.age is None:
        for msg in reversed(conversation_history[-5:]):  # Last 5 messages
            if msg.get('role') == 'user':
                hist_profile = extract_profile_from_query(msg.get('content', ''))
                if hist_profile.age and not profile.age:
                    profile.age = hist_profile.age
                if hist_profile.gender and not profile.gender:
                    profile.gender = hist_profile.gender
                for condition in hist_profile.medical_history:
                    if condition not in profile.medical_history:
                        profile.medical_history.append(condition)
    
    logger.info(f"Final extracted profile: {profile}")
    return profile


def is_severe_condition(profile: UserProfile) -> bool:
    """
    Determine if the user's condition is severe and requires urgent care.
    """
    if profile.severity_level is not None and profile.severity_level >= 3:
        return True
    
    severe_keywords = ['severe', 'proliferative', 'PDR', 'advanced']
    if profile.condition:
        return any(keyword.lower() in profile.condition.lower() for keyword in severe_keywords)
    
    return False    