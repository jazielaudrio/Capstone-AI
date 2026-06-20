import re

def recommend_complexity(title: str, description: str) -> int:
    """
    Estimates task complexity (1-5) based on keywords and text length.
    """
    text = (title + " " + description).lower()
    
    # Keyword sets mapped to complexity scores
    level_5_keywords = ['architecture', 'microservices', 'refactor', 'infrastructure', 'payment gateway', 'security', 'oauth', 'ml model', 'pipeline', 'deployment', 'kubernetes', 'k8s', 'docker swarm']
    level_4_keywords = ['database', 'schema', 'migration', 'integration', 'api', 'webhook', 'cache', 'redis', 'elasticsearch', 'performance', 'optimize', 'socket', 'realtime']
    level_3_keywords = ['crud', 'form', 'component', 'endpoint', 'query', 'filter', 'pagination', 'state', 'redux', 'context', 'service', 'handler']
    level_2_keywords = ['bug', 'fix', 'ui', 'css', 'style', 'color', 'alignment', 'typo', 'update text', 'button', 'modal']
    
    score = 1 # Default minimum complexity
    
    # 1. Keyword based scoring
    if any(kw in text for kw in level_5_keywords):
        score = 5
    elif any(kw in text for kw in level_4_keywords):
        score = 4
    elif any(kw in text for kw in level_3_keywords):
        score = 3
    elif any(kw in text for kw in level_2_keywords):
        score = 2
        
    # 2. Text length based adjustment
    # A very long description usually indicates more complex requirements
    word_count = len(text.split())
    if word_count > 100 and score < 4:
        score += 1
    elif word_count > 200 and score < 5:
        score += 1
        
    # 3. Short description penalty
    # If it's just "fix bug" (2 words), it's likely very simple
    if word_count < 5 and score > 2:
        score -= 1
        
    # Ensure score is within 1-5 bounds
    return max(1, min(5, score))
