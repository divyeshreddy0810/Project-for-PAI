#!/usr/bin/env python3
"""
Sentiment Cache Manager
-----------------------
Speeds up sentiment analysis by caching results and using fast pretrained models.
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np


class SentimentCache:
    """Manages caching of sentiment analysis results"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.index_file = os.path.join(cache_dir, "sentiment_index.json")
        os.makedirs(cache_dir, exist_ok=True)
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load cache index"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_index(self):
        """Save cache index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _hash_headline(self, headline: str) -> str:
        """Create hash of headline"""
        return hashlib.md5(headline.lower().strip().encode()).hexdigest()
    
    def get(self, headline: str) -> Optional[Dict]:
        """Get cached sentiment for headline"""
        h_hash = self._hash_headline(headline)
        
        if h_hash in self.index:
            entry = self.index[h_hash]
            # Check if cache is fresh (within 30 days)
            cached_date = datetime.fromisoformat(entry['cached_at'])
            if datetime.now() - cached_date < timedelta(days=30):
                return entry['sentiment']
            else:
                # Remove stale cache
                del self.index[h_hash]
                self._save_index()
        
        return None
    
    def set(self, headline: str, sentiment: Dict):
        """Cache sentiment result"""
        h_hash = self._hash_headline(headline)
        self.index[h_hash] = {
            'headline': headline[:100],
            'sentiment': sentiment,
            'cached_at': datetime.now().isoformat()
        }
        self._save_index()
    
    def clear_stale(self):
        """Remove stale cache entries (>30 days)"""
        cutoff_date = datetime.now() - timedelta(days=30)
        stale_hashes = []
        
        for h_hash, entry in self.index.items():
            cached_date = datetime.fromisoformat(entry['cached_at'])
            if cached_date < cutoff_date:
                stale_hashes.append(h_hash)
        
        for h_hash in stale_hashes:
            del self.index[h_hash]
        
        if stale_hashes:
            self._save_index()
            print(f"🧹 Cleared {len(stale_hashes)} stale cache entries")


class FastSentimentAnalyzer:
    """
    Fast sentiment analyzer using VADER (Valence Aware Dictionary and sEntiment Reasoner)
    as a quick fallback when FinBERT is slow.
    """
    
    @staticmethod
    def analyze_with_vader(text: str) -> Dict:
        """
        Use VADER sentiment analyzer (fast, no model downloading required)
        Falls back to regex-based sentiment if VADER not available
        """
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk
            
            # Download required VADER lexicon
            try:
                nltk.data.find('sentiment/vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(text)
            
            # Convert to FinBERT-like format (positive prob 0-1)
            positive = max(0, min(1, (scores['compound'] + 1) / 2))
            negative = 1 - positive
            neutral = 0.5 - abs(scores['compound']) / 2
            
            return {
                'positive': float(positive),
                'negative': float(negative),
                'neutral': float(neutral),
                'confidence': abs(scores['compound']),
                'label': 'POSITIVE' if positive > 0.55 else ('NEGATIVE' if negative > 0.55 else 'NEUTRAL'),
                'method': 'VADER'
            }
        except ImportError:
            # Fallback: regex-based sentiment
            return FastSentimentAnalyzer.analyze_with_regex(text)
    
    @staticmethod
    def analyze_with_regex(text: str) -> Dict:
        """
        Ultra-fast regex-based sentiment (no dependencies)
        """
        text_lower = text.lower()
        
        positive_words = [
            'positive', 'good', 'great', 'excellent', 'amazing', 'wonderful',
            'strong', 'gain', 'growth', 'bull', 'up', 'rise', 'rally',
            'surge', 'boom', 'profit', 'success', 'win'
        ]
        
        negative_words = [
            'negative', 'bad', 'poor', 'terrible', 'awful', 'lose', 'loss',
            'decline', 'fall', 'bear', 'down', 'crash', 'plunge', 'fail',
            'risk', 'warning', 'concern', 'trouble'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        
        if total > 0:
            positive = pos_count / total
            negative = neg_count / total
        else:
            positive = 0.5
            negative = 0.5
        
        neutral = 1 - abs(positive - negative)
        confidence = abs(positive - negative)
        
        return {
            'positive': float(positive),
            'negative': float(negative),
            'neutral': float(neutral),
            'confidence': float(confidence),
            'label': 'POSITIVE' if positive > 0.55 else ('NEGATIVE' if negative > 0.55 else 'NEUTRAL'),
            'method': 'REGEX'
        }


class HybridSentimentAnalyzer:
    """
    Uses caching + VADER for speed, FinBERT for accuracy
    """
    
    def __init__(self, use_finbert: bool = True, cache_enabled: bool = True):
        self.use_finbert = use_finbert
        self.cache = SentimentCache() if cache_enabled else None
        
        if use_finbert:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                print("✅ FinBERT loaded successfully")
            except Exception as e:
                print(f"⚠️  FinBERT load failed: {e}. Using VADER instead.")
                self.use_finbert = False
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment with caching and fallback strategies
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(text)
            if cached:
                return {**cached, 'cached': True}
        
        # Use FinBERT if available and enabled
        if self.use_finbert:
            try:
                import torch
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)[0]
                
                labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
                label = labels[probs.argmax().item()]
                
                sentiment = {
                    'negative': float(probs[0]),
                    'neutral': float(probs[1]),
                    'positive': float(probs[2]),
                    'confidence': float(probs.max()),
                    'label': label,
                    'method': 'FinBERT',
                    'cached': False
                }
            except Exception as e:
                print(f"⚠️  FinBERT inference failed: {e}. Using VADER.")
                sentiment = FastSentimentAnalyzer.analyze_with_vader(text)
                sentiment['cached'] = False
        else:
            # Use VADER (fast, no GPU needed)
            sentiment = FastSentimentAnalyzer.analyze_with_vader(text)
            sentiment['cached'] = False
        
        # Cache result
        if self.cache:
            self.cache.set(text, sentiment)
        
        return sentiment
    
    def analyze_batch(self, texts: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Analyze multiple texts with progress tracking
        """
        results = []
        
        for i, text in enumerate(texts):
            result = self.analyze(text)
            results.append(result)
            
            if (i + 1) % batch_size == 0:
                print(f"   Processed {i + 1}/{len(texts)}")
        
        if len(texts) % batch_size != 0:
            print(f"   Processed {len(texts)}/{len(texts)}")
        
        return results


if __name__ == "__main__":
    # Test the hybrid analyzer
    analyzer = HybridSentimentAnalyzer(use_finbert=False)  # Use VADER for speed
    
    test_headlines = [
        "Stock market surges on positive earnings reports",
        "Tech stocks plunge amid recession fears",
        "Bitcoin rallies after regulatory approval",
    ]
    
    for headline in test_headlines:
        result = analyzer.analyze(headline)
        print(f"\n'{headline[:50]}...'")
        print(f"  Positive: {result['positive']:.3f}")
        print(f"  Negative: {result['negative']:.3f}")
        print(f"  Neutral: {result['neutral']:.3f}")
        print(f"  Label: {result['label']} (Method: {result.get('method', 'Unknown')})")
