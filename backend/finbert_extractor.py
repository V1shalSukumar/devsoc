"""
FinBERT Financial Term Extractor

Uses ProsusAI/finbert to:
1. Identify important financial segments in transcripts
2. Extract key financial terms using hybrid approach (sentiment + vocabulary)
3. Prepare terms for LLM explanation

Memory optimized for low-resource environments.
"""

import re
import gc
import os
from functools import lru_cache
from typing import Optional
import psutil

# Set environment variables for memory optimization BEFORE importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Limit OpenMP threads to 1
os.environ.setdefault("MKL_NUM_THREADS", "1")  # Limit MKL threads
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Disable tokenizer parallelism

import torch
torch.set_num_threads(1)  # Limit torch threads for memory

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# ── FinBERT Model Configuration ────────────────────────────────────────────

FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Enable lightweight mode to skip model loading (for memory-constrained environments)
# When True, only vocabulary-based extraction is used
# DEFAULT: True to prevent OOM crashes on systems with limited RAM
LIGHTWEIGHT_MODE = True

# Minimum RAM required to load FinBERT model (in GB)
MIN_RAM_FOR_MODEL_GB = 4.0

# Financial vocabulary for hybrid extraction
FINANCIAL_TERMS_VOCABULARY = {
    # Banking & Accounts
    "account balance", "savings account", "current account", "fixed deposit",
    "recurring deposit", "overdraft", "credit limit", "debit card", "credit card",
    "net banking", "mobile banking", "NEFT", "RTGS", "IMPS", "UPI", "SWIFT",
    
    # Loans & Credit
    "EMI", "equated monthly installment", "principal", "interest rate", "APR",
    "annual percentage rate", "tenure", "prepayment", "foreclosure", "collateral",
    "mortgage", "home loan", "personal loan", "auto loan", "education loan",
    "credit score", "CIBIL", "creditworthiness", "debt-to-income",
    
    # Investment (removed ambiguous single words)
    "mutual fund", "SIP", "systematic investment plan", "NAV", "net asset value",
    "equity fund", "debt fund", "hybrid fund", "index fund", "ETF", "exchange traded fund",
    "investment portfolio", "diversification", "asset allocation", "risk appetite",
    "investment returns", "CAGR", "compound annual growth rate", "dividend yield", "capital gains",
    "stock market", "share price", "stock price", "shares outstanding",
    
    # Insurance
    "insurance premium", "sum assured", "maturity value", "surrender value", "policy term",
    "beneficiary", "nominee", "claim settlement", "exclusions", "insurance coverage",
    
    # Compliance & Regulatory
    "KYC", "know your customer", "AML", "anti-money laundering", "PAN",
    "Aadhaar", "compliance", "disclosure", "terms and conditions",
    "cooling-off period", "grievance redressal", "ombudsman",
    
    # Fees & Charges
    "processing fee", "service charge", "late payment fee", "penalty",
    "foreclosure charges", "prepayment penalty", "annual fee", "maintenance charge",
    
    # Trading & Markets (more specific terms)
    "stock exchange", "bond market", "debenture", "derivative", "futures contract", "options contract",
    "margin trading", "leverage ratio", "stop loss", "limit order", "market order",
    "bull market", "bear market", "market volatility", "market liquidity",
    
    # Financial Metrics
    "ROI", "return on investment", "ROE", "return on equity", "P/E ratio",
    "price to earnings", "EPS", "earnings per share", "book value",
}

# Terms that are ambiguous and need context validation
# These words have common non-financial meanings
AMBIGUOUS_TERMS = {
    "share": ["screen share", "share screen", "what to share", "share link", "share with", 
              "share it", "share your", "share the", "will share", "can share", "to share",
              "share this", "share my", "share our", "share a", "share an", "share some",
              "share that", "share those", "please share", "could share", "would share",
              "want to share", "like to share", "going to share", "need to share"],
    "stock": ["in stock", "out of stock", "stock up", "stock room", "take stock",
              "laughing stock", "stock phrase", "stock answer", "stock character"],
    "premium": ["premium quality", "premium service", "premium support", "premium plan", "premium tier",
                "premium features", "premium subscription", "premium member", "premium access",
                "premium package", "premium version", "premium content"],
    "bond": ["bond with", "bonding", "bond between", "bond together", "emotional bond",
             "family bond", "team bond", "social bond"],
    "margin": ["margin of", "profit margin", "margin note", "page margin", "error margin",
               "margin for error", "safety margin", "narrow margin", "slim margin"],
    "equity": ["home equity", "equity in", "brand equity", "social equity", "pay equity",
               "racial equity", "gender equity"],
    "returns": ["returns policy", "product returns", "return the", "returns it", "return policy",
                "return item", "return product", "return or exchange", "free returns"],
    "coverage": ["network coverage", "signal coverage", "media coverage", "news coverage",
                 "press coverage", "wifi coverage", "cellular coverage", "broad coverage"],
    "maturity": ["emotional maturity", "product maturity", "mental maturity", "personal maturity"],
    "principal": ["school principal", "principal reason", "principal cause", "principal concern"],
    "dividend": [],  # Usually financial, but check context
    "portfolio": ["art portfolio", "design portfolio", "work portfolio", "creative portfolio",
                  "photography portfolio", "project portfolio"],
    "fund": ["fun day", "fun time", "fundraiser", "fund raiser", "fun"],  # Watch for "fund" vs "fun"
    "yield": ["yield sign", "yield to", "yield right", "yield results", "high yield"],
    "rate": ["at this rate", "rate of", "heart rate", "rate this", "rate my"],
    "balance": ["work life balance", "balance beam", "balance out", "off balance", "lose balance",
                "keep balance", "sense of balance"],
    "term": ["long term", "short term", "terms of use", "search term", "term paper"],
    "credit": ["give credit", "credit for", "to your credit", "extra credit", "take credit",
               "full credit"],
    "interest": ["point of interest", "show interest", "lose interest", "of interest",
                 "personal interest", "conflict of interest", "best interest"],
    "value": ["value for money", "values matter", "family values", "core values", "good value"],
    "capital": ["capital city", "capital letter", "death capital", "capital punishment"],
    "asset": ["asset to", "valuable asset", "team asset"],
}

# Phrases that indicate NON-financial usage
FALSE_POSITIVE_PATTERNS = [
    # Share patterns
    r"(?:choose|select|pick)\s+(?:what\s+)?to\s+share",
    r"share\s+(?:screen|link|file|document|photo|video|image|location|contact|details)",
    r"screen\s+share",
    r"(?:want|like|going|need|would|could|can|will)\s+to\s+share",
    r"share\s+(?:with|it|this|that|my|your|our|some)",
    r"(?:please|could\s+you|would\s+you)\s+share",
    # Stock patterns
    r"(?:in|out\s+of)\s+stock",
    r"(?:take|check)\s+stock\s+of",
    r"laughing\s+stock",
    # Premium patterns
    r"premium\s+(?:quality|service|support|version|tier|plan|features|subscription|member|access|package|content)",
    # Bond patterns  
    r"bond\s+(?:with|between|together)",
    r"(?:emotional|family|team|social)\s+bond",
    # Margin patterns
    r"(?:profit|error|safety|page)\s+margin",
    r"margin\s+(?:of|for|note)",
    # Returns patterns
    r"(?:product|order|item|free)\s+returns?",
    r"(?:return|send)\s+(?:it|the|this|that|item|product)",
    r"return\s+(?:policy|or\s+exchange)",
    # Coverage patterns
    r"(?:network|signal|media|news|press|wifi|cellular)\s+coverage",
    # Principal patterns
    r"(?:school|vice|assistant)\s+principal",
    r"principal\s+(?:reason|cause|concern)",
    # Portfolio patterns
    r"(?:art|design|work|creative|photography|project)\s+portfolio",
    # Balance patterns
    r"(?:work\s+life|life\s+work)\s+balance",
    r"(?:keep|lose|off|sense\s+of)\s+balance",
    r"balance\s+(?:beam|out)",
    # Credit patterns
    r"(?:give|take|full|extra)\s+credit",
    r"credit\s+(?:for|to)",
    # Interest patterns (non-financial)
    r"(?:point|place)\s+of\s+interest",
    r"(?:show|lose|personal|conflict\s+of|best)\s+interest",
    # Value patterns
    r"(?:family|core|good)\s+values?",
    # Capital patterns
    r"capital\s+(?:city|letter|punishment)",
    # Asset patterns (non-financial)
    r"(?:valuable|team)\s+asset",
    r"asset\s+to\s+(?:the|our|your)",
]

# Compile false positive patterns
_FALSE_POSITIVE_COMPILED: list = []

def _get_false_positive_patterns() -> list:
    """Lazily compile false positive regex patterns."""
    global _FALSE_POSITIVE_COMPILED
    if not _FALSE_POSITIVE_COMPILED:
        _FALSE_POSITIVE_COMPILED = [
            re.compile(p, re.IGNORECASE) for p in FALSE_POSITIVE_PATTERNS
        ]
    return _FALSE_POSITIVE_COMPILED

# Financial context indicators - if ambiguous term is near these, it's more likely financial
FINANCIAL_CONTEXT_INDICATORS = {
    # Currency and money
    "money", "rupees", "dollars", "rs", "inr", "usd", "eur", "gbp", "percent", "%",
    "₹", "$", "€", "£", "cents", "paise", "lakh", "crore", "thousand", "million",
    # Banking
    "bank", "banking", "loan", "account", "payment", "deposit", "withdraw", "transfer",
    "savings", "current", "fixed", "recurring", "fd", "rd", "cheque", "check",
    # Investment
    "invest", "investing", "investment", "trading", "trade", "market", "markets",
    "equity", "equities", "mutual", "sip", "nav", "etf", "nifty", "sensex",
    # Finance terms
    "finance", "financial", "fund", "funds", "asset", "assets", "liability", "liabilities",
    "capital", "interest", "rate", "rates", "fee", "fees", "charge", "charges",
    "tax", "taxes", "income", "expense", "expenses", "profit", "loss", "gains",
    # Participants
    "broker", "brokers", "investor", "investors", "trader", "traders",
    "lender", "borrower", "creditor", "debtor", "shareholder", "shareholders",
    # Transactions
    "transaction", "transactions", "portfolio", "wealth", "holdings",
    "buy", "sell", "purchase", "redeem", "redemption",
    # Regulatory (India)
    "nse", "bse", "sebi", "rbi", "irda", "pfrda", "nps", "ppf", "epf",
    # Loan terms
    "emi", "tenure", "principal", "apr", "roi", "cibil", "credit score",
    "collateral", "mortgage", "foreclosure", "prepayment",
    # Insurance
    "insurance", "premium", "policy", "claim", "coverage", "nominee", "beneficiary",
    # Financial verbs
    "borrow", "lend", "repay", "default", "mature", "compound", "accrue",
    # KYC/Compliance
    "kyc", "aml", "pan", "aadhaar", "compliance", "disclosure",
}

# Compile regex patterns for term matching (case-insensitive)
_TERM_PATTERNS: dict[str, re.Pattern] = {}


def _get_term_patterns() -> dict[str, re.Pattern]:
    """Lazily compile regex patterns for financial terms."""
    global _TERM_PATTERNS
    if not _TERM_PATTERNS:
        for term in FINANCIAL_TERMS_VOCABULARY:
            # Word boundary matching
            pattern = r'(?<!\w)' + re.escape(term) + r'(?!\w)'
            _TERM_PATTERNS[term] = re.compile(pattern, re.IGNORECASE)
    return _TERM_PATTERNS


# ── Model Loading with Memory Optimization ────────────────────────────────

# Global model cache (allows explicit cleanup)
_MODEL_CACHE: dict = {}


def _clear_model_cache():
    """Clear the model cache and free memory."""
    global _MODEL_CACHE
    if _MODEL_CACHE:
        _MODEL_CACHE.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def check_memory_available() -> bool:
    """
    Check if there's enough memory available to load the FinBERT model.
    
    Returns:
        True if sufficient memory is available, False otherwise
    """
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        print(f"Available RAM: {available_gb:.2f} GB (minimum required: {MIN_RAM_FOR_MODEL_GB} GB)")
        return available_gb >= MIN_RAM_FOR_MODEL_GB
    except Exception as e:
        print(f"⚠ Could not check memory: {e}")
        return False  # Assume not enough memory if we can't check


def load_finbert_model(force_reload: bool = False) -> tuple:
    """
    Load and cache FinBERT model and tokenizer with memory optimization.
    Uses CPU for inference with low memory settings.
    
    Args:
        force_reload: If True, reload the model even if cached
    
    Returns:
        tuple: (tokenizer, model, device)
        
    Raises:
        MemoryError: If insufficient RAM is available
        RuntimeError: If model loading fails
    """
    global _MODEL_CACHE
    
    if not force_reload and "model" in _MODEL_CACHE:
        return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"], _MODEL_CACHE["device"]
    
    # Check memory before attempting to load
    if not check_memory_available():
        raise MemoryError(
            f"Insufficient RAM to load FinBERT model. "
            f"Need at least {MIN_RAM_FOR_MODEL_GB} GB available. "
            f"Using lightweight vocabulary-only mode instead."
        )
    
    # Clear any existing cache first
    _clear_model_cache()
    gc.collect()
    
    print("Loading FinBERT model (memory-optimized, first run downloads ~440MB)...")
    
    try:
        # Load tokenizer (lightweight)
        tokenizer = AutoTokenizer.from_pretrained(
            FINBERT_MODEL_NAME,
            model_max_length=128,  # Reduced from 256 to save memory
        )
        
        # Load model with memory optimizations
        model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_MODEL_NAME,
            low_cpu_mem_usage=True,  # Reduce CPU memory during loading
            torch_dtype=torch.float32,  # Keep float32 for CPU
        )
        
        # Force CPU usage
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()
        
        # Disable gradient computation globally for inference
        for param in model.parameters():
            param.requires_grad = False
        
        # Cache the model
        _MODEL_CACHE["tokenizer"] = tokenizer
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["device"] = device
        
        gc.collect()
        print("✓ FinBERT model loaded successfully (memory-optimized)")
        return tokenizer, model, device
        
    except (MemoryError, RuntimeError, OSError) as e:
        print(f"⚠ Error loading FinBERT: {e}")
        _clear_model_cache()
        gc.collect()
        raise MemoryError(f"Failed to load FinBERT model: {e}")


# ── Segment Splitting ──────────────────────────────────────────────────────

def split_into_segments(text: str, max_length: int = 400) -> list[dict]:
    """
    Split transcript into analyzable segments.
    Tries to split on sentence boundaries.
    
    Args:
        text: Full transcript text
        max_length: Maximum characters per segment
        
    Returns:
        List of segment dicts with 'text', 'start_char', 'end_char'
    """
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    segments = []
    current_segment = ""
    current_start = 0
    
    for sentence in sentences:
        if len(current_segment) + len(sentence) <= max_length:
            current_segment += (" " if current_segment else "") + sentence
        else:
            if current_segment:
                end_pos = current_start + len(current_segment)
                segments.append({
                    "text": current_segment.strip(),
                    "start_char": current_start,
                    "end_char": end_pos,
                })
                current_start = end_pos + 1
            current_segment = sentence
    
    # Add remaining segment
    if current_segment.strip():
        segments.append({
            "text": current_segment.strip(),
            "start_char": current_start,
            "end_char": current_start + len(current_segment),
        })
    
    return segments


# ── FinBERT Sentiment Analysis ─────────────────────────────────────────────

def analyze_segment_sentiment(
    segments: list[dict],
    tokenizer,
    model,
    device,
    batch_size: int = 2,  # Small batch size for memory efficiency
) -> list[dict]:
    """
    Analyze financial sentiment for each segment using FinBERT.
    Memory-optimized with batch processing and explicit cleanup.
    
    FinBERT outputs: positive, negative, neutral
    High positive/negative scores indicate financially significant content.
    
    Args:
        segments: List of segment dicts with 'text' key
        tokenizer: FinBERT tokenizer
        model: FinBERT model
        device: torch device
        batch_size: Number of segments to process at once (default 2 for low memory)
        
    Returns:
        List of segments with added sentiment analysis
    """
    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    
    analyzed_segments = []
    total_segments = len(segments)
    
    # Process in small batches to manage memory
    for batch_start in range(0, total_segments, batch_size):
        batch_end = min(batch_start + batch_size, total_segments)
        batch = segments[batch_start:batch_end]
        
        for seg in batch:
            text = seg["text"]
            
            try:
                # Tokenize with reduced max_length
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,  # Reduced from 512 to save memory
                    padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Inference with no_grad for memory efficiency
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                
                # Extract results
                predicted_class = int(np.argmax(probs))
                sentiment = label_map[predicted_class]
                confidence = float(probs[predicted_class])
                
                # Financial importance score (non-neutral = more important)
                importance_score = float(1.0 - probs[2])  # 1 - neutral probability
                
                analyzed_segments.append({
                    **seg,
                    "sentiment": sentiment,
                    "sentiment_confidence": round(confidence, 3),
                    "financial_importance": round(importance_score, 3),
                    "sentiment_scores": {
                        "positive": round(float(probs[0]), 3),
                        "negative": round(float(probs[1]), 3),
                        "neutral": round(float(probs[2]), 3),
                    },
                })
                
                # Explicit cleanup of tensors
                del inputs, outputs, probs
                
            except (MemoryError, RuntimeError) as e:
                # On OOM, add segment with default values
                print(f"⚠ Memory error processing segment, using defaults: {e}")
                analyzed_segments.append({
                    **seg,
                    "sentiment": "neutral",
                    "sentiment_confidence": 0.5,
                    "financial_importance": 0.5,
                    "sentiment_scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                })
        
        # Clean up after each batch
        gc.collect()
    
    return analyzed_segments


# ── Vocabulary-Based Term Extraction ───────────────────────────────────────

def _calculate_context_score(text: str, match_start: int, match_end: int, term: str) -> float:
    """
    Calculate a context score (0.0-1.0) indicating how likely this term is financial.
    
    Uses multiple signals:
    1. Presence of financial context indicators nearby
    2. Proximity to other confirmed financial vocabulary terms
    3. Sentence structure analysis
    
    Args:
        text: Full text being analyzed
        match_start: Start position of the match
        match_end: End position of the match  
        term: The term that was matched
        
    Returns:
        Score between 0.0 (definitely not financial) and 1.0 (definitely financial)
    """
    score = 0.5  # Start neutral
    
    # Get extended context (150 chars each side for better analysis)
    context_start = max(0, match_start - 150)
    context_end = min(len(text), match_end + 150)
    context = text[context_start:context_end].lower()
    
    # Get immediate context (50 chars) for proximity scoring
    immediate_start = max(0, match_start - 50)
    immediate_end = min(len(text), match_end + 50)
    immediate_context = text[immediate_start:immediate_end].lower()
    
    # Signal 1: Count financial indicators in extended context
    indicator_count = sum(1 for ind in FINANCIAL_CONTEXT_INDICATORS if ind in context)
    if indicator_count >= 3:
        score += 0.3  # Strong financial context
    elif indicator_count >= 1:
        score += 0.15  # Some financial context
    else:
        score -= 0.2  # No financial context = less likely
    
    # Signal 2: Check for other vocabulary terms nearby (proximity boost)
    patterns = _get_term_patterns()
    nearby_terms = 0
    for other_term, pattern in patterns.items():
        if other_term.lower() != term.lower():
            if pattern.search(context):
                nearby_terms += 1
                if nearby_terms >= 2:
                    break  # Enough to boost score
    
    if nearby_terms >= 2:
        score += 0.2  # Multiple financial terms nearby = strong signal
    elif nearby_terms == 1:
        score += 0.1  # One other term = moderate signal
    
    # Signal 3: Check for financial action verbs in immediate context
    financial_verbs = {
        "pay", "paid", "paying", "invest", "invested", "investing",
        "borrow", "borrowed", "borrowing", "lend", "lending", "lent",
        "deposit", "deposited", "withdraw", "withdrawn", "transfer",
        "buy", "buying", "bought", "sell", "selling", "sold",
        "charge", "charged", "charging", "owe", "owed", "owing",
        "earn", "earned", "earning", "redeem", "redeemed",
    }
    if any(verb in immediate_context for verb in financial_verbs):
        score += 0.15
    
    # Clamp score to valid range
    return max(0.0, min(1.0, score))


def _is_false_positive(text: str, match_start: int, match_end: int, term: str) -> bool:
    """
    Check if a matched term is actually a false positive based on surrounding context.
    Uses multi-layered analysis including pattern matching and context scoring.
    
    Args:
        text: Full text being analyzed
        match_start: Start position of the match
        match_end: End position of the match
        term: The term that was matched
        
    Returns:
        True if this is likely a false positive (non-financial usage)
    """
    # Get extended context around the match (150 chars each side)
    context_start = max(0, match_start - 150)
    context_end = min(len(text), match_end + 150)
    context = text[context_start:context_end].lower()
    
    # Layer 1: Check against explicit false positive patterns
    for pattern in _get_false_positive_patterns():
        if pattern.search(context):
            return True
    
    term_lower = term.lower()
    
    # Layer 2: For ambiguous terms, check exclusion phrases
    if term_lower in AMBIGUOUS_TERMS:
        exclusion_phrases = AMBIGUOUS_TERMS[term_lower]
        for phrase in exclusion_phrases:
            if phrase.lower() in context:
                return True
        
        # Layer 3: Calculate context score for ambiguous terms
        context_score = _calculate_context_score(text, match_start, match_end, term)
        
        # Threshold: require at least 0.5 score for ambiguous terms
        if context_score < 0.5:
            return True  # Low context score = likely false positive
    
    return False


def extract_vocabulary_terms(text: str) -> list[dict]:
    """
    Extract financial terms using vocabulary matching with context validation.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of found terms with positions
    """
    patterns = _get_term_patterns()
    found_terms = []
    text_lower = text.lower()
    
    for term, pattern in patterns.items():
        for match in pattern.finditer(text):
            # Get the actual matched text (preserves case)
            matched_text = match.group()
            
            # Skip false positives using context analysis
            if _is_false_positive(text, match.start(), match.end(), term):
                continue
            
            # Extract context (surrounding words)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            found_terms.append({
                "term": term,
                "matched_text": matched_text,
                "start": match.start(),
                "end": match.end(),
                "context": f"...{context}..." if start > 0 or end < len(text) else context,
            })
    
    # Remove duplicates (same term, same position)
    seen = set()
    unique_terms = []
    for t in found_terms:
        key = (t["term"], t["start"])
        if key not in seen:
            seen.add(key)
            unique_terms.append(t)
    
    return sorted(unique_terms, key=lambda x: x["start"])


# ── Hybrid Term Extraction ─────────────────────────────────────────────────

def extract_key_financial_terms(
    text: str,
    analyzed_segments: list[dict],
    importance_threshold: float = 0.3,
) -> list[dict]:
    """
    Hybrid extraction: Vocabulary terms + importance filtering from FinBERT.
    
    Args:
        text: Full transcript
        analyzed_segments: Segments with FinBERT analysis
        importance_threshold: Minimum financial importance score
        
    Returns:
        List of key financial terms with context and importance
    """
    # Step 1: Get vocabulary-matched terms
    vocab_terms = extract_vocabulary_terms(text)
    
    # Step 2: Enhance with segment importance scores
    enhanced_terms = []
    
    for term in vocab_terms:
        term_start = term["start"]
        term_end = term["end"]
        
        # Find which segment this term belongs to
        segment_importance = 0.0
        segment_sentiment = "neutral"
        
        for seg in analyzed_segments:
            if seg["start_char"] <= term_start < seg["end_char"]:
                segment_importance = seg["financial_importance"]
                segment_sentiment = seg["sentiment"]
                break
        
        # Only include terms from financially important segments
        # OR terms that are inherently important (in vocab)
        if segment_importance >= importance_threshold or term["term"] in FINANCIAL_TERMS_VOCABULARY:
            enhanced_terms.append({
                **term,
                "importance_score": round(max(segment_importance, 0.5), 3),  # Min 0.5 for vocab matches
                "segment_sentiment": segment_sentiment,
            })
    
    # Step 3: Deduplicate by term (keep highest importance)
    term_map: dict[str, dict] = {}
    for t in enhanced_terms:
        key = t["term"].lower()
        if key not in term_map or t["importance_score"] > term_map[key]["importance_score"]:
            term_map[key] = t
    
    # Sort by importance
    return sorted(term_map.values(), key=lambda x: -x["importance_score"])


# ── Important Segments Identification ──────────────────────────────────────

def identify_important_segments(
    analyzed_segments: list[dict],
    importance_threshold: float = 0.4,
    max_segments: int = 10,
) -> list[dict]:
    """
    Identify the most financially important segments.
    
    Args:
        analyzed_segments: Segments with FinBERT analysis
        importance_threshold: Minimum importance score
        max_segments: Maximum number of segments to return
        
    Returns:
        List of important segments sorted by importance
    """
    important = [
        seg for seg in analyzed_segments
        if seg["financial_importance"] >= importance_threshold
    ]
    
    # Sort by importance descending
    important.sort(key=lambda x: -x["financial_importance"])
    
    return important[:max_segments]


# ── Main Entry Point ───────────────────────────────────────────────────────

def run_finbert_analysis(transcript: str, use_model: bool = True) -> dict:
    """
    Run complete FinBERT analysis on a transcript.
    
    1. Splits transcript into segments
    2. Analyzes each segment with FinBERT (if use_model=True and not LIGHTWEIGHT_MODE)
    3. Identifies important financial segments
    4. Extracts key financial terms (hybrid approach with context awareness)
    
    Args:
        transcript: Full call transcript text
        use_model: Whether to load and use the FinBERT model. Set to False for
                   vocabulary-only extraction (faster, less memory).
        
    Returns:
        Dict with analysis results:
        - important_segments: Financially significant segments
        - financial_terms: Extracted terms with importance scores
        - overall_financial_tone: Summary of financial sentiment
        - segment_count: Total segments analyzed
        - mode: 'full' or 'lightweight' indicating which mode was used
    """
    if not transcript or not transcript.strip():
        return {
            "important_segments": [],
            "financial_terms": [],
            "overall_financial_tone": "neutral",
            "segment_count": 0,
            "mode": "lightweight",
            "error": "Empty transcript",
        }
    
    # Determine if we should use the model
    should_use_model = use_model and not LIGHTWEIGHT_MODE
    
    # Split into segments
    segments = split_into_segments(transcript)
    
    if not segments:
        return {
            "important_segments": [],
            "financial_terms": [],
            "overall_financial_tone": "neutral",
            "segment_count": 0,
            "mode": "lightweight" if not should_use_model else "full",
        }
    
    if should_use_model:
        # Full mode: Load model and analyze with FinBERT
        try:
            tokenizer, model, device = load_finbert_model()
            analyzed_segments = analyze_segment_sentiment(segments, tokenizer, model, device)
            important_segments = identify_important_segments(analyzed_segments)
            financial_terms = extract_key_financial_terms(transcript, analyzed_segments)
            mode = "full"
        except (MemoryError, RuntimeError) as e:
            # Fall back to lightweight mode on memory issues
            print(f"⚠ FinBERT model loading failed ({e}), falling back to lightweight mode")
            should_use_model = False
    
    if not should_use_model:
        # Lightweight mode: vocabulary extraction only
        analyzed_segments = []
        important_segments = []
        # Use vocabulary extraction with context awareness
        vocab_terms = extract_vocabulary_terms(transcript)
        financial_terms = [
            {
                **t,
                "importance_score": 0.6,  # Default importance for vocab matches
                "segment_sentiment": "neutral",
            }
            for t in vocab_terms
        ]
        mode = "lightweight"
    
    # Calculate overall financial tone
    if analyzed_segments:
        avg_positive = sum(s["sentiment_scores"]["positive"] for s in analyzed_segments) / len(analyzed_segments)
        avg_negative = sum(s["sentiment_scores"]["negative"] for s in analyzed_segments) / len(analyzed_segments)
        
        if avg_positive > 0.4:
            overall_tone = "positive"
        elif avg_negative > 0.4:
            overall_tone = "negative"
        else:
            overall_tone = "neutral"
    else:
        overall_tone = "neutral"
    
    return {
        "important_segments": [
            {
                "text": seg["text"],
                "sentiment": seg["sentiment"],
                "importance": seg["financial_importance"],
            }
            for seg in important_segments
        ],
        "financial_terms": [
            {
                "term": t["term"],
                "context": t.get("context", ""),
                "importance": t.get("importance_score", t.get("importance", 0.5)),
            }
            for t in financial_terms[:20]  # Limit to top 20 terms
        ],
        "overall_financial_tone": overall_tone,
        "segment_count": len(segments),
        "important_segment_count": len(important_segments),
        "unique_terms_found": len(financial_terms),
        "mode": mode,
    }


# ── Prepare Terms for LLM Explanation ──────────────────────────────────────

def prepare_terms_for_explanation(finbert_result: dict) -> list[str]:
    """
    Extract unique term names for LLM explanation.
    
    Args:
        finbert_result: Result from run_finbert_analysis()
        
    Returns:
        List of unique term names to explain
    """
    terms = finbert_result.get("financial_terms", [])
    # Get unique terms, sorted by importance
    return [t["term"] for t in terms]


if __name__ == "__main__":
    # Quick test
    test_text = """
    Good morning, I'm calling about the EMI payment for my home loan. 
    The interest rate seems higher than what was mentioned during the loan processing.
    I was told the APR would be around 8.5% but I'm being charged more.
    Can you explain the processing fee and prepayment penalty as well?
    I want to understand my credit score requirements before applying for a top-up loan.
    """
    
    print("Running FinBERT analysis test...")
    result = run_finbert_analysis(test_text)
    
    print(f"\nAnalyzed {result['segment_count']} segments")
    print(f"Found {result['important_segment_count']} important segments")
    print(f"Extracted {result['unique_terms_found']} unique financial terms")
    print(f"Overall tone: {result['overall_financial_tone']}")
    
    print("\nImportant Segments:")
    for seg in result["important_segments"]:
        print(f"  [{seg['sentiment']}] {seg['text'][:80]}...")
    
    print("\nFinancial Terms:")
    for term in result["financial_terms"][:10]:
        print(f"  - {term['term']} (importance: {term['importance']})")
