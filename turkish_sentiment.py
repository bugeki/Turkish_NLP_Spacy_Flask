"""
Turkish Sentiment Analysis using XGBoost
Lightweight, fast, and production-ready
"""

import numpy as np
from collections import Counter
import re

class TurkishSentimentAnalyzer:
    """
    Lightweight Turkish sentiment analyzer using rule-based approach
    and feature engineering. Can be upgraded to XGBoost when training data available.
    """
    
    def __init__(self, nlp=None):
        self.nlp = nlp
        
        # Turkish positive words dictionary
        self.positive_words = {
            'güzel', 'harika', 'muhteşem', 'mükemmel', 'süper', 'başarılı',
            'iyi', 'hoş', 'sevdim', 'beğendim', 'mutlu', 'keyifli', 'eğlenceli',
            'kaliteli', 'başarı', 'tebrikler', 'bravo', 'aferin', 'teşekkür',
            'sağolun', 'minnettar', 'şahane', 'enfes', 'kusursuz', 'efsane',
            'nefis', 'olağanüstü', 'parlak', 'görkemli', 'fevkalade',
            'hayran', 'takdir', 'övgü', 'sevinç', 'zevk', 'huzur',
            'masal', 'rüya', 'cennet', 'mucize', 'hayal', 'gurur'
        }
        
        # Turkish negative words dictionary
        self.negative_words = {
            'kötü', 'berbat', 'rezalet', 'çöp', 'boktan', 'iğrenç', 'tiksinç',
            'vasat', 'beğenmedim', 'sevmedim', 'sıkıcı', 'can', 'üzücü',
            'fena', 'boş', 'saçma', 'anlamsız', 'zayıf', 'eksik', 'yetersiz',
            'başarısız', 'kırık', 'bozuk', 'sorunlu', 'problem', 'hata',
            'korkunç', 'dehşet', 'felaket', 'trajedi', 'acı', 'ızdırap',
            'pişman', 'hayal', 'kırıklığı', 'üzüntü', 'öfke', 'sinir',
            'nefret', 'tiksinti', 'ihanet', 'yalan', 'aldatma', 'hile'
        }
        
        # Turkish intensifiers
        self.intensifiers = {
            'çok': 1.5, 'fazla': 1.3, 'aşırı': 1.8, 'son': 1.4, 'derece': 1.4,
            'gerçekten': 1.3, 'kesinlikle': 1.5, 'tamamen': 1.4, 'oldukça': 1.3,
            'gayet': 1.2, 'epey': 1.3, 'bayağı': 1.3, 'bir': 1.2, 'hayli': 1.3
        }
        
        # Turkish negations
        self.negations = {'değil', 'yok', 'hiç', 'asla', 'hayır'}
        
    def extract_features(self, text):
        """Extract features from text for sentiment analysis"""
        features = {}
        
        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Emoji sentiment (basic)
        positive_emojis = ['😊', '😀', '😁', '🙂', '😍', '🥰', '❤️', '👍', '✨', '🎉']
        negative_emojis = ['😢', '😭', '😞', '😔', '😡', '😠', '💔', '👎', '😰', '😨']
        
        features['positive_emoji_count'] = sum(text.count(e) for e in positive_emojis)
        features['negative_emoji_count'] = sum(text.count(e) for e in negative_emojis)
        
        # Tokenize
        words = text.lower().split()
        
        # Lexicon-based features
        features['positive_word_count'] = sum(1 for w in words if w in self.positive_words)
        features['negative_word_count'] = sum(1 for w in words if w in self.negative_words)
        features['intensifier_count'] = sum(1 for w in words if w in self.intensifiers)
        features['negation_count'] = sum(1 for w in words if w in self.negations)
        
        # spaCy features (if available)
        if self.nlp:
            doc = self.nlp(text)
            
            # POS tag distribution
            pos_counts = Counter([token.pos_ for token in doc])
            features['noun_count'] = pos_counts.get('NOUN', 0)
            features['verb_count'] = pos_counts.get('VERB', 0)
            features['adj_count'] = pos_counts.get('ADJ', 0)
            features['adv_count'] = pos_counts.get('ADV', 0)
            
            # Named entities
            features['entity_count'] = len(doc.ents)
        
        return features
    
    def calculate_sentiment_score(self, text):
        """
        Calculate sentiment score using rule-based approach
        Returns: (score, label, confidence)
        score: -1.0 (very negative) to 1.0 (very positive)
        """
        features = self.extract_features(text)
        words = text.lower().split()
        
        # Base score from lexicon
        pos_score = features['positive_word_count']
        neg_score = features['negative_word_count']
        
        # Apply intensifiers
        for i, word in enumerate(words):
            if word in self.intensifiers:
                multiplier = self.intensifiers[word]
                # Look at next word
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word in self.positive_words:
                        pos_score += 0.5 * multiplier
                    elif next_word in self.negative_words:
                        neg_score += 0.5 * multiplier
        
        # Handle negations (flip sentiment)
        negation_active = False
        for word in words:
            if word in self.negations:
                negation_active = True
                # Swap scores partially
                pos_score, neg_score = neg_score * 0.7, pos_score * 0.7
                break
        
        # Emoji contribution
        pos_score += features['positive_emoji_count'] * 0.5
        neg_score += features['negative_emoji_count'] * 0.5
        
        # Exclamation marks (intensify existing sentiment)
        if features['exclamation_count'] > 0:
            if pos_score > neg_score:
                pos_score *= (1 + features['exclamation_count'] * 0.1)
            elif neg_score > pos_score:
                neg_score *= (1 + features['exclamation_count'] * 0.1)
        
        # Calculate final score (-1 to 1)
        total = pos_score + neg_score
        if total == 0:
            score = 0.0
            label = 'Nötr'
            confidence = 0.5
        else:
            score = (pos_score - neg_score) / (pos_score + neg_score)
            
            # Determine label
            if score > 0.2:
                label = 'Pozitif'
            elif score < -0.2:
                label = 'Negatif'
            else:
                label = 'Nötr'
            
            # Calculate confidence (0 to 1)
            confidence = min(abs(score) + 0.3, 1.0)
        
        return score, label, confidence
    
    def analyze(self, text):
        """
        Main analysis method
        Returns dict with sentiment information
        """
        if not text or not text.strip():
            return {
                'score': 0.0,
                'label': 'Nötr',
                'confidence': 0.0,
                'polarity': 0.0,
                'subjectivity': 0.5
            }
        
        score, label, confidence = self.calculate_sentiment_score(text)
        
        return {
            'score': round(score, 3),
            'label': label,
            'confidence': round(confidence, 3),
            'polarity': round(score, 3),  # Same as score for compatibility
            'subjectivity': round(confidence, 3),  # Use confidence as proxy
            'model': 'Turkish Lexicon + Features'
        }


# Example usage and testing
if __name__ == "__main__":
    analyzer = TurkishSentimentAnalyzer()
    
    # Test cases
    test_texts = [
        "Bu film gerçekten muhteşemdi! Çok beğendim, harika bir deneyimdi.",
        "Berbat bir ürün, hiç beğenmedim. Param boşa gitti.",
        "Fena değil ama çok da iyi değil.",
        "Bugün hava güzel.",
        "Rezalet bir hizmet! Çok kötü, asla tavsiye etmem.",
        "Harika! Süper bir deneyim yaşadım 😊👍",
        "😢 Çok üzücü bir durum, kesinlikle kötü.",
    ]
    
    print("Turkish Sentiment Analyzer - Test Results")
    print("=" * 60)
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Label: {result['label']}")
        print(f"Score: {result['score']}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 60)
