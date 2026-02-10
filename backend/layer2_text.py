"""
LAYER 2: Multilingual Text Processing (Local)
- Language-aware PII detection (Presidio for English, regex for all languages)
- Financial entity extraction (English, Hindi, Russian)
- Named entity recognition (spaCy for English)
- Profanity / prohibited phrase detection (multilingual)
- Obligation keyword extraction (multilingual)

Key fix: Presidio ONLY runs on English text. For non-English (Hindi, etc.),
we use regex-only PII detection to avoid false positives like Hindi words
being misclassified as PERSON/LOCATION.
"""

import re
import spacy
from pathlib import Path

# Microsoft Presidio for PII detection (English only)
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------
# LANGUAGE DETECTION HELPER
# ---------------------------------------------------------------------------

_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')   # Hindi, Marathi, Sanskrit
_CYRILLIC_RE = re.compile(r'[\u0400-\u04FF]')     # Russian, etc.
_TAMIL_RE = re.compile(r'[\u0B80-\u0BFF]')         # Tamil
_TELUGU_RE = re.compile(r'[\u0C00-\u0C7F]')        # Telugu
_KANNADA_RE = re.compile(r'[\u0C80-\u0CFF]')       # Kannada
_MALAYALAM_RE = re.compile(r'[\u0D00-\u0D7F]')     # Malayalam
_BENGALI_RE = re.compile(r'[\u0980-\u09FF]')       # Bengali, Assamese
_GUJARATI_RE = re.compile(r'[\u0A80-\u0AFF]')      # Gujarati
_GURMUKHI_RE = re.compile(r'[\u0A00-\u0A7F]')      # Punjabi
_ARABIC_RE = re.compile(r'[\u0600-\u06FF]')        # Arabic, Urdu
_CJK_RE = re.compile(r'[\u4E00-\u9FFF]')           # Chinese
_HANGUL_RE = re.compile(r'[\uAC00-\uD7AF]')        # Korean
_KANA_RE = re.compile(r'[\u3040-\u30FF]')          # Japanese Hiragana/Katakana
_LATIN_RE = re.compile(r'[a-zA-Z]')


def _detect_script(text: str) -> str:
    """Detect dominant script in text. Returns a language code.
    Any non-Latin script → returns a non-'en' code so Presidio is skipped."""
    scripts = {
        "hi": len(_DEVANAGARI_RE.findall(text)),
        "ta": len(_TAMIL_RE.findall(text)),
        "te": len(_TELUGU_RE.findall(text)),
        "kn": len(_KANNADA_RE.findall(text)),
        "ml": len(_MALAYALAM_RE.findall(text)),
        "bn": len(_BENGALI_RE.findall(text)),
        "gu": len(_GUJARATI_RE.findall(text)),
        "pa": len(_GURMUKHI_RE.findall(text)),
        "ar": len(_ARABIC_RE.findall(text)),
        "zh": len(_CJK_RE.findall(text)),
        "ko": len(_HANGUL_RE.findall(text)),
        "ja": len(_KANA_RE.findall(text)),
        "ru": len(_CYRILLIC_RE.findall(text)),
    }
    latin_count = len(_LATIN_RE.findall(text))
    non_latin_total = sum(scripts.values())
    total = non_latin_total + latin_count

    if total == 0:
        return "en"

    # If any non-Latin script dominates (>20% of chars), it's not English
    if non_latin_total > 0:
        dominant = max(scripts, key=scripts.get)
        if scripts[dominant] / max(total, 1) > 0.2:
            return dominant

    return "en"


# Map of known language codes/names → normalized code
_LANG_MAP = {
    # English
    "en": "en", "english": "en", "eng": "en",
    # Hindi
    "hi": "hi", "hindi": "hi", "hin": "hi",
    # Tamil
    "ta": "ta", "tam": "ta", "tamil": "ta",
    # Telugu
    "te": "te", "tel": "te", "telugu": "te",
    # Kannada
    "kn": "kn", "kan": "kn", "kannada": "kn",
    # Malayalam
    "ml": "ml", "mal": "ml", "malayalam": "ml",
    # Bengali
    "bn": "bn", "ben": "bn", "bengali": "bn", "bangla": "bn",
    # Gujarati
    "gu": "gu", "guj": "gu", "gujarati": "gu",
    # Punjabi
    "pa": "pa", "pan": "pa", "punjabi": "pa",
    # Marathi
    "mr": "hi", "mar": "hi", "marathi": "hi",  # Devanagari — same treatment as Hindi
    # Russian
    "ru": "ru", "russian": "ru", "rus": "ru",
    # Arabic / Urdu
    "ar": "ar", "arabic": "ar", "ara": "ar",
    "ur": "ar", "urdu": "ar", "urd": "ar",
    # Chinese
    "zh": "zh", "chinese": "zh", "zho": "zh", "cmn": "zh",
    # Japanese
    "ja": "ja", "japanese": "ja", "jpn": "ja",
    # Korean
    "ko": "ko", "korean": "ko", "kor": "ko",
    # European — treat these as non-English too (Presidio only has English NLP)
    "es": "es", "spanish": "es", "spa": "es",
    "fr": "fr", "french": "fr", "fra": "fr",
    "de": "de", "german": "de", "deu": "de",
    "pt": "pt", "portuguese": "pt", "por": "pt",
    "it": "it", "italian": "it", "ita": "it",
}


def _normalize_language(language: str, text: str = "") -> str:
    """Normalize language code. Returns 'en' ONLY for verified English.
    Any unrecognized or non-English language → returns a non-'en' code
    so that Presidio (English-only) is never run on non-English text."""
    lang = language.lower().strip()

    mapped = _LANG_MAP.get(lang)

    if mapped and mapped != "en":
        # Explicitly non-English — trust it
        return mapped

    if mapped == "en":
        # Caller says English — verify via script detection
        # (catches the case where Layer 1 returns "en" but text is Devanagari/Tamil/etc.)
        if text:
            detected = _detect_script(text)
            if detected != "en":
                return detected
        return "en"

    # Unknown language code or 'auto' — detect from text script
    if text:
        detected = _detect_script(text)
        # If script detection can't tell, and the language code is something
        # we don't recognize, default to NON-English to be safe
        if detected == "en" and lang not in ("", "auto", "en", "english", "eng"):
            return "other"
        return detected

    # No text, no recognized language — default English
    return "en"


# ---------------------------------------------------------------------------
# CUSTOM RECOGNIZERS FOR SENSITIVE DATA (English Presidio)
# ---------------------------------------------------------------------------

remote_access_patterns = [
    Pattern(name="remote_access_dashed", regex=r"\b\d(?:[-.\s]\d){8,11}\b", score=0.9),
    Pattern(name="remote_access_continuous", regex=r"\b\d{9,12}\b", score=0.7),
]
remote_access_recognizer = PatternRecognizer(
    supported_entity="REMOTE_ACCESS_CODE",
    patterns=remote_access_patterns,
    name="RemoteAccessRecognizer",
    supported_language="en",
)

sensitive_number_patterns = [
    Pattern(name="numeric_sequence_6_plus", regex=r"\b\d{6,}\b", score=0.5),
    Pattern(name="formatted_numeric_code",
            regex=r"\b\d{2,4}[-.\s]\d{2,4}[-.\s]\d{2,4}(?:[-.\s]\d{2,4})?\b", score=0.6),
]
sensitive_number_recognizer = PatternRecognizer(
    supported_entity="SENSITIVE_NUMBER",
    patterns=sensitive_number_patterns,
    name="SensitiveNumberRecognizer",
    supported_language="en",
)

phone_patterns = [
    Pattern(name="phone_intl",
            regex=r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", score=0.85),
    Pattern(name="phone_spoken",
            regex=r"\b(?:call|phone|mobile|cell|contact)(?:\s+(?:me|us|at))?\s*:?\s*\+?\d[\d\s.-]{8,15}\b",
            score=0.9),
]
phone_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER_EXTENDED",
    patterns=phone_patterns,
    name="ExtendedPhoneRecognizer",
    supported_language="en",
)

india_id_patterns = [
    Pattern(name="aadhaar", regex=r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b", score=0.85),
    Pattern(name="pan_card", regex=r"\b[A-Z]{5}\d{4}[A-Z]\b", score=0.95),
]
india_id_recognizer = PatternRecognizer(
    supported_entity="INDIA_ID",
    patterns=india_id_patterns,
    name="IndiaIDRecognizer",
    supported_language="en",
)

# ---------------------------------------------------------------------------
# PRESIDIO ANALYZER SETUP (English only)
# ---------------------------------------------------------------------------
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}

provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

registry = RecognizerRegistry()
registry.load_predefined_recognizers(nlp_engine=nlp_engine)
registry.add_recognizer(remote_access_recognizer)
registry.add_recognizer(sensitive_number_recognizer)
registry.add_recognizer(phone_recognizer)
registry.add_recognizer(india_id_recognizer)

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["en"],
    registry=registry,
)

PRESIDIO_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
    "CREDIT_CARD", "US_BANK_NUMBER", "IP_ADDRESS", "DATE_TIME",
    "LOCATION", "US_DRIVER_LICENSE", "US_PASSPORT", "IBAN_CODE",
    "NRP", "MEDICAL_LICENSE", "URL",
    "REMOTE_ACCESS_CODE", "SENSITIVE_NUMBER",
    "PHONE_NUMBER_EXTENDED", "INDIA_ID",
]

# ---------------------------------------------------------------------------
# UNIVERSAL REGEX PII PATTERNS (work on ANY language / script)
# These detect structured data patterns that are language-agnostic
# ---------------------------------------------------------------------------
UNIVERSAL_PII_PATTERNS = {
    "PHONE_NUMBER": re.compile(
        r"(?:\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4})"
        r"|(?:\b(?:\+?91[-.\s]?)?\d{10}\b)"
        r"|(?:\+?7[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2})",
    ),
    "EMAIL_ADDRESS": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "CREDIT_CARD": re.compile(
        r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"
    ),
    "AADHAAR": re.compile(
        r"\b\d{4}[-.\s]\d{4}[-.\s]\d{4}\b"
    ),
    "PAN_CARD": re.compile(
        r"\b[A-Z]{5}\d{4}[A-Z]\b"
    ),
    "ACCOUNT_NUMBER": re.compile(
        r"(?:a/?c|account|खाता|अकाउंट)\s*(?:no\.?|number|#|नंबर|नम्बर)?\s*:?\s*\d{9,18}\b",
        re.IGNORECASE,
    ),
    "US_SSN": re.compile(
        r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b"
    ),
    "IFSC_CODE": re.compile(
        r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
    ),
    "REMOTE_ACCESS_CODE": re.compile(
        r"\b\d(?:[-.\s]\d){8,11}\b"
    ),
    "IP_ADDRESS": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    "URL": re.compile(
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    ),
    "IBAN_CODE": re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"
    ),
    "PASSPORT_RU": re.compile(
        r"\b\d{4}\s?\d{6}\b"
    ),
    "INN_RU": re.compile(
        r"\b\d{10}(?:\d{2})?\b"
    ),
    "SNILS_RU": re.compile(
        r"\b\d{3}[-.\s]\d{3}[-.\s]\d{3}[-.\s]\d{2}\b"
    ),
}

# Contextual PII patterns for Hindi (need keyword context to avoid false positives)
CONTEXTUAL_PII_PATTERNS_HI = {
    "PHONE_NUMBER": re.compile(
        r"(?:फ़ोन|फोन|मोबाइल|नंबर|नम्बर|कॉल|call|phone|mobile)\s*"
        r"(?:नंबर|नम्बर|number|no\.?)?\s*:?\s*\+?\d[\d\s.-]{8,15}",
        re.IGNORECASE,
    ),
    "ACCOUNT_NUMBER": re.compile(
        r"(?:खाता|अकाउंट|account|a/?c)\s*(?:नंबर|नम्बर|number|no\.?|#)?\s*:?\s*\d{9,18}",
        re.IGNORECASE,
    ),
    "OTP": re.compile(
        r"(?:OTP|ओटीपी|कोड|code|वेरिफ़िकेशन|verification)\s*:?\s*\d{4,8}",
        re.IGNORECASE,
    ),
    "CVV": re.compile(
        r"(?:CVV|सीवीवी|CVC)\s*:?\s*\d{3,4}",
        re.IGNORECASE,
    ),
    "PIN": re.compile(
        r"(?:PIN|पिन|ATM\s*पिन|ATM\s*PIN)\s*:?\s*\d{4,6}",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# FINANCIAL ENTITY PATTERNS (multilingual)
# ---------------------------------------------------------------------------
FINANCIAL_PATTERNS_EN = {
    "currency_amount": re.compile(
        r"(?:(?:Rs\.?|INR|USD|\$|€|£|₹)\s*\d[\d,]*(?:\.\d{1,2})?)"
        r"|(?:\d[\d,]*(?:\.\d{1,2})?\s*(?:rupees|dollars|euros|pounds|lakhs?|crores?"
        r"|thousand|hundred|million|billion))",
        re.IGNORECASE,
    ),
    "percentage": re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|percent|per\s*cent)\b", re.IGNORECASE
    ),
    "date_reference": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        r"|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{2,4})"
        r"|(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
        re.IGNORECASE,
    ),
    "loan_term": re.compile(
        r"\b\d+\s*(?:months?|years?|days?|EMI|installments?)\b",
        re.IGNORECASE,
    ),
}

FINANCIAL_PATTERNS_HI = {
    "currency_amount_hi": re.compile(
        # Symbol + digits: "₹500", "Rs. 1000"
        r"(?:(?:Rs\.?|INR|₹)\s*\d[\d,]*(?:\.\d{1,2})?)"
        # Digits + Hindi units: "4 लाख", "10 हज़ार"
        r"|(?:\d[\d,]*(?:\.\d{1,2})?\s*(?:रुपये|रुपए|रुपया|लाख|करोड़|हज़ार|हजार|सौ|पैसे?))"
        # Hindi number words + units: "चार लाख", "दो करोड़"
        r"|(?:(?:एक|दो|तीन|चार|पाँच|पांच|छह|छः|सात|आठ|नौ|दस"
        r"|ग्यारह|बारह|तेरह|चौदह|पंद्रह|सोलह|सत्रह|अठारह|उन्नीस"
        r"|बीस|तीस|चालीस|पचास|साठ|सत्तर|अस्सी|नब्बे|सौ)"
        r"\s+(?:लाख|करोड़|हज़ार|हजार|सौ|रुपये|रुपए|रुपया))"
        # "X लाख से ऊपर"
        r"|(?:\d+\s*लाख(?:\s+(?:से\s+)?(?:ऊपर|नीचे|तक|के))?)",
        re.IGNORECASE,
    ),
    "percentage_hi": re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|प्रतिशत|परसेंट|फीसदी)\b",
        re.IGNORECASE,
    ),
    "loan_term_hi": re.compile(
        r"\b\d+\s*(?:महीने?|साल|दिन|वर्ष|ईएमआई|EMI|किस्त(?:ों|ें)?)\b",
        re.IGNORECASE,
    ),
    "interest_rate_hi": re.compile(
        r"(?:ब्याज|interest|rate)\s*(?:दर|rate)?\s*:?\s*\d+(?:\.\d+)?\s*(?:%|प्रतिशत|परसेंट)?",
        re.IGNORECASE,
    ),
}

FINANCIAL_PATTERNS_RU = {
    "currency_rub": re.compile(
        r"(?:\d[\d\s,]*(?:[.,]\d{1,2})?\s*(?:рублей|руб\.?|₽))"
        r"|(?:\d[\d\s,]*(?:[.,]\d{1,2})?\s*(?:тысяч|миллион(?:ов|а)?|млн)"
        r"\s*(?:рублей|долларов|евро)?)"
        r"|(?:\d[\d\s,]*(?:[.,]\d{1,2})?\s*(?:долларов|евро))",
        re.IGNORECASE,
    ),
    "percentage_ru": re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:%|процент(?:ов|а)?)\b", re.IGNORECASE
    ),
    "date_ru": re.compile(
        r"\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа"
        r"|сентября|октября|ноября|декабря)\s*\d{2,4}?\b",
        re.IGNORECASE,
    ),
    "loan_term_ru": re.compile(
        r"\b\d+\s*(?:месяц(?:ев|а)?|лет|год(?:а|ов)?|дней|дня)\b",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# PROFANITY / PROHIBITED PHRASES (multilingual)
# ---------------------------------------------------------------------------
PROFANITY_WORDS_EN = {
    "damn", "hell", "shit", "fuck", "bastard", "ass", "crap",
    "idiot", "stupid", "dumb", "moron", "shut up",
}

PROFANITY_WORDS_HI = {
    "बेवकूफ", "गधा", "कमीना", "हरामी", "साला", "कुत्ता",
    "चुप", "बकवास", "पागल", "मूर्ख", "उल्लू", "गधे",
    "चोर", "झूठा", "नालायक", "निकम्मा", "बदतमीज़", "बदतमीज",
}

PROFANITY_WORDS_RU = {
    "дурак", "идиот", "тупой", "мошенник",
}

PROHIBITED_PHRASES_EN = [
    "we will take legal action immediately",
    "your credit score will be ruined",
    "we'll seize your assets",
    "you must decide right now",
    "this is your last chance",
    "don't tell anyone about this offer",
    "zero fees",
    "guaranteed approval",
    "risk-free investment",
]

PROHIBITED_PHRASES_HI = [
    "हम कानूनी कार्रवाई करेंगे",
    "कानूनी कार्रवाई",
    "आपकी प्रॉपर्टी जब्त",
    "संपत्ति जब्त",
    "अभी फैसला करो",
    "अभी तय करो",
    "आखिरी मौका",
    "किसी को मत बताना",
    "कोई फीस नहीं",
    "गारंटी",
    "बिना रिस्क",
    "ज़ीरो फीस",
    "झूठ बोल रहे",
    "legal action",
    "police complaint",
    "FIR",
    "arrest",
    "jail",
]

PROHIBITED_PHRASES_RU = [
    "мы подадим в суд",
    "ваш кредитный рейтинг будет испорчен",
    "мы арестуем ваше имущество",
    "вы должны решить прямо сейчас",
    "это ваш последний шанс",
    "гарантированное одобрение",
]


# ---------------------------------------------------------------------------
# OBLIGATION KEYWORDS (multilingual — expanded Hindi)
# ---------------------------------------------------------------------------
OBLIGATION_KEYWORDS_EN = [
    "must", "shall", "required", "mandatory", "obligated",
    "need to", "have to", "should", "will be charged",
    "agree to", "consent", "acknowledge", "confirm",
    "i promise", "we guarantee", "committed to",
    "by signing", "terms and conditions", "cooling off",
    "within 30 days", "penalty", "fee", "interest rate",
    "liable", "binding", "enforceable", "forfeit", "waive",
]

OBLIGATION_KEYWORDS_HI = [
    "ज़रूरी", "ज़रूरी है", "जरूरी", "अनिवार्य", "वादा", "सहमत", "शर्तें",
    "शर्तों", "ज़िम्मेदारी", "जिम्मेदारी", "बाध्य", "कानूनी",
    "भुगतान", "पेमेंट", "किस्त", "ब्याज", "ब्याज दर", "फीस",
    "चार्ज", "जुर्माना", "पेनल्टी", "देना होगा", "देना पड़ेगा",
    "चुकाना", "भरना होगा", "भरना पड़ेगा", "जमा करना",
    "गारंटी", "वारंटी", "बीमा", "प्रीमियम",
    "EMI", "ईएमआई", "लोन", "ऋण", "कर्ज", "कर्ज़",
    "अकाउंट", "खाता", "बैलेंस", "बकाया",
    "नियम", "agreement", "सहमति", "अनुबंध",
    "डिफॉल्ट", "default", "overdue", "ओवरड्यू",
    "purchase", "पर्चेस", "खरीद", "stock", "स्टॉक",
    "target", "टारगेट", "टार्गेट",
]

OBLIGATION_KEYWORDS_RU = [
    "должен", "обязан", "необходимо", "обещаю", "гарантирую",
    "подтверждаю", "согласен", "обязательно", "штраф", "комиссия",
    "процент", "условия", "договор", "контракт",
    "в течение", "обязуюсь", "ответственность",
]


# ---------------------------------------------------------------------------
# DENY LISTS — words that must NEVER be flagged as PII
# ---------------------------------------------------------------------------
DENY_LIST_EN = {
    "anydesk", "teamviewer", "zoom", "skype", "whatsapp", "telegram",
    "chrome", "firefox", "safari", "edge", "opera",
    "windows", "macos", "linux", "ubuntu",
    "pixel", "iphone", "samsung", "motorola", "oneplus", "xiaomi",
    "huawei", "nokia", "sony", "lg", "oppo", "vivo", "realme",
    "android", "ios",
    "google", "apple", "microsoft", "amazon", "facebook", "meta",
    "paypal", "venmo", "cashapp", "cash app", "zelle",
    "bitcoin", "ethereum", "crypto",
    "support", "help", "hello", "ok", "okay", "yes", "no",
    "vpn", "qr", "wifi", "usb", "sim",
}

# Common Hindi words/particles that must NEVER be flagged as PII
DENY_LIST_HI = {
    # Particles, postpositions, pronouns
    "है", "हैं", "हो", "था", "थी", "थे", "हूँ", "हूं",
    "का", "की", "के", "को", "से", "में", "पर", "तक", "ने",
    "और", "या", "भी", "तो", "ही", "ना", "नहीं", "नही", "मत",
    "यह", "वह", "ये", "वो", "इस", "उस", "जो", "कि", "जी",
    "कब", "कहाँ", "कहां", "कैसे", "क्या", "कौन", "क्यों", "कितना", "कितने",
    "अब", "तब", "यहाँ", "वहाँ", "यहां", "वहां",
    "मैं", "तुम", "आप", "हम", "उन", "इन",
    "एक", "दो", "तीन", "चार", "पाँच", "पांच",
    # Common verbs / forms
    "करो", "करें", "करना", "करते", "करती", "किया", "गया", "गई",
    "बोल", "बोलो", "बोला", "बोली", "बोले",
    "देखो", "देखिए", "देखें", "देखा", "देखी",
    "बताएं", "बताओ", "बताया", "बताइए",
    "लगाइए", "लगाओ", "लगा", "लगी", "लगे",
    "रखे", "रखा", "रखी", "रखो", "रखें",
    "लिए", "लिया", "लेना", "लेते", "लेती",
    "चलो", "चलें", "चला", "चली",
    "आया", "आई", "आए", "आता", "आती", "आते",
    "जाओ", "जाना", "जाएं", "जाते", "जाती",
    "होगा", "होगी", "होंगे", "होता", "होती",
    # Common nouns — never PII
    "सामान", "पैसा", "पैसे", "लोग", "लोगों", "बात",
    "काम", "दिन", "रात", "घर", "दुकान", "दुकानदार",
    "भैया", "भाई", "बहन", "दीदी", "अंकल", "आंटी",
    "साहब", "मैडम", "सर", "जी",
    "अच्छा", "बहुत", "थोड़ा", "ज्यादा", "कम", "ऊपर", "नीचे",
    "खुश", "खाली", "महंगा", "सस्ता",
    "line", "negative", "purchase", "stock", "target", "average",
    "total", "staff", "employee",
    # Common financial call words (not PII themselves)  
    "एंप्लॉई", "स्टाफ", "स्टॉक", "पर्चेस", "टोटल", "एवरेज",
    "टारगेट", "बिजनेस", "मार्केट",
    # Additional particles that get detected as NRP
    "वगैरा", "वगैरह", "इत्यादि",
}

# Presidio filter for English
PERSON_NOT_LOCATION = {
    "maryam", "stephen", "michael", "omar", "adam", "sarah", "john",
    "david", "james", "robert", "william", "joseph", "charles",
    "mary", "patricia", "jennifer", "linda", "elizabeth", "susan",
    "ali", "ahmed", "mohammed", "fatima", "ayesha", "hassan",
    "raj", "priya", "amit", "sunita", "vikram", "anita",
}

CONTEXT_CORRECTIONS = {
    "speaking to": "PERSON",
    "my name is": "PERSON",
    "name is": "PERSON",
    "this is": "PERSON",
    "manager is": "PERSON",
    "colleague": "PERSON",
    "daughter": "PERSON",
    "son": "PERSON",
    "mr.": "PERSON",
    "mrs.": "PERSON",
    "ms.": "PERSON",
    "dr.": "PERSON",
}


# ---------------------------------------------------------------------------
# PII DETECTION FUNCTIONS
# ---------------------------------------------------------------------------

def _should_filter_entity(value: str, entity_type: str) -> bool:
    """Check if an entity should be filtered out (false positive)."""
    value_lower = value.lower().strip()

    if value_lower in DENY_LIST_EN or value_lower in DENY_LIST_HI:
        return True

    if len(value_lower) < 2:
        return True

    if entity_type in ("PERSON", "LOCATION", "NRP") and value_lower.isdigit():
        return True

    common_words = {
        "one", "two", "three", "four", "five", "the", "and", "for",
        "sir", "madam", "dear", "hello", "please", "thank", "thanks",
        "good", "morning", "afternoon", "evening", "day",
    }
    if entity_type in ("PERSON", "LOCATION", "NRP") and value_lower in common_words:
        return True

    return False


def _correct_entity_type(value: str, entity_type: str, text: str, start: int) -> str:
    """Correct entity type using surrounding context (English only)."""
    value_lower = value.lower().strip()

    if entity_type == "LOCATION" and value_lower in PERSON_NOT_LOCATION:
        return "PERSON"

    context_start = max(0, start - 50)
    context = text[context_start:start].lower()

    for keyword, correct_type in CONTEXT_CORRECTIONS.items():
        if keyword in context:
            return correct_type

    return entity_type


def _calculate_confidence(result, text: str) -> float:
    """Adjusted confidence scoring for Presidio results."""
    base_score = result.score
    value = text[result.start:result.end].lower()
    entity_type = result.entity_type

    if entity_type in ("REMOTE_ACCESS_CODE", "INDIA_ID", "CREDIT_CARD", "US_SSN"):
        return min(0.95, base_score + 0.1)

    if len(value) > 10:
        base_score = min(0.95, base_score + 0.05)

    if len(value) < 4:
        base_score = max(0.4, base_score - 0.15)

    common_words = {"one", "two", "three", "four", "five", "the", "and", "for"}
    if value in common_words:
        base_score = max(0.3, base_score - 0.3)

    return round(base_score, 3)


def detect_pii_presidio(text: str, score_threshold: float = 0.5) -> list[dict]:
    """Detect PII using Microsoft Presidio. ONLY for English text."""
    findings = []

    try:
        results = analyzer.analyze(
            text=text,
            entities=PRESIDIO_ENTITIES,
            language="en",
            score_threshold=score_threshold,
        )

        seen_positions = set()

        for result in results:
            value = text[result.start:result.end]
            entity_type = result.entity_type

            pos_key = (result.start, result.end)
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)

            if _should_filter_entity(value, entity_type):
                continue

            corrected_type = _correct_entity_type(value, entity_type, text, result.start)
            confidence = _calculate_confidence(result, text)

            if confidence < score_threshold:
                continue

            findings.append({
                "type": corrected_type,
                "value": value,
                "start": result.start,
                "end": result.end,
                "confidence": confidence,
                "risk": "high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low",
                "method": "presidio",
            })

    except Exception as e:
        print(f"Presidio error: {e}")

    return findings


def detect_pii_regex(text: str, language: str = "en") -> list[dict]:
    """
    Detect PII using universal regex patterns. Works for ALL languages.
    Catches structured data: phone numbers, emails, Aadhaar, PAN, etc.
    """
    findings = []
    seen_spans = set()

    for pii_type, pattern in UNIVERSAL_PII_PATTERNS.items():
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if any(s[0] <= span[0] and s[1] >= span[1] for s in seen_spans):
                continue
            seen_spans.add(span)

            value = match.group().strip()
            if len(value) < 3:
                continue

            # Validate IP addresses
            if pii_type == "IP_ADDRESS":
                parts = value.split(".")
                if not all(p.isdigit() and int(p) < 256 for p in parts):
                    continue

            findings.append({
                "type": pii_type,
                "value": value,
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.85,
                "risk": "high",
                "method": "regex",
            })

    # Hindi contextual PII (needs keyword context to avoid false positives)
    if language == "hi":
        for pii_type, pattern in CONTEXTUAL_PII_PATTERNS_HI.items():
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if any(s[0] <= span[0] and s[1] >= span[1] for s in seen_spans):
                    continue
                seen_spans.add(span)

                findings.append({
                    "type": pii_type,
                    "value": match.group().strip(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9,
                    "risk": "high",
                    "method": "regex_contextual",
                })

    findings.sort(key=lambda x: x["start"])
    return findings


def detect_pii(text: str, language: str = "en", score_threshold: float = 0.5) -> list[dict]:
    """
    Language-aware PII detection.
    
    - English: Presidio (ML-based) + regex patterns
    - Hindi/Russian/Other: Regex-only (avoids Presidio false positives)
    
    This is the KEY fix: Presidio's English NLP model produces garbage results
    on non-Latin scripts, so we skip it entirely for non-English.
    """
    lang = _normalize_language(language, text)

    # Always run universal regex patterns
    regex_findings = detect_pii_regex(text, language=lang)

    if lang == "en":
        # English: also run Presidio for ML-based detection
        presidio_findings = detect_pii_presidio(text, score_threshold)

        # Merge: prefer Presidio results for overlapping spans
        merged = []
        presidio_spans = {(f["start"], f["end"]) for f in presidio_findings}

        for f in presidio_findings:
            merged.append(f)

        for f in regex_findings:
            span = (f["start"], f["end"])
            if not any(ps[0] <= span[0] and ps[1] >= span[1] for ps in presidio_spans):
                merged.append(f)

        merged.sort(key=lambda x: x["start"])
        return merged
    else:
        # Non-English: regex-only — NO Presidio
        return regex_findings


# ---------------------------------------------------------------------------
# FINANCIAL ENTITY EXTRACTION (language-aware)
# ---------------------------------------------------------------------------

def extract_financial_entities(text: str, language: str = "en") -> list[dict]:
    """Extract financial entities using language-appropriate patterns."""
    lang = _normalize_language(language, text)
    entities = []

    if lang == "hi":
        pattern_sets = {**FINANCIAL_PATTERNS_HI, **FINANCIAL_PATTERNS_EN}
    elif lang == "ru":
        pattern_sets = {**FINANCIAL_PATTERNS_RU, **FINANCIAL_PATTERNS_EN}
    else:
        pattern_sets = FINANCIAL_PATTERNS_EN

    seen_spans = set()
    for ent_type, pattern in pattern_sets.items():
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)

            entities.append({
                "type": ent_type,
                "value": match.group().strip(),
                "start": match.start(),
                "end": match.end(),
            })

    entities.sort(key=lambda x: x["start"])
    return entities


# ---------------------------------------------------------------------------
# NAMED ENTITY RECOGNITION (spaCy for English only)
# ---------------------------------------------------------------------------

def extract_named_entities(text: str, language: str = "en") -> list[dict]:
    """
    Extract named entities using spaCy (English only).
    For non-English, returns empty list — running en_core_web_sm on Hindi
    produces garbage results.
    """
    lang = _normalize_language(language, text)

    if lang != "en":
        return []

    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
        if ent.label_ in {"PERSON", "ORG", "GPE", "MONEY", "DATE", "CARDINAL", "PERCENT"}
    ]


# ---------------------------------------------------------------------------
# PROFANITY / PROHIBITED PHRASES (language-aware)
# ---------------------------------------------------------------------------

def detect_profanity(text: str, language: str = "en") -> list[dict]:
    """Detect profanity and prohibited phrases in the correct language."""
    lang = _normalize_language(language, text)
    findings = []
    text_lower = text.lower()

    if lang == "hi":
        profanity_words = PROFANITY_WORDS_HI | PROFANITY_WORDS_EN
        prohibited_phrases = PROHIBITED_PHRASES_HI + PROHIBITED_PHRASES_EN
    elif lang == "ru":
        profanity_words = PROFANITY_WORDS_RU | PROFANITY_WORDS_EN
        prohibited_phrases = PROHIBITED_PHRASES_RU + PROHIBITED_PHRASES_EN
    else:
        profanity_words = PROFANITY_WORDS_EN
        prohibited_phrases = PROHIBITED_PHRASES_EN

    # Check prohibited phrases
    for phrase in prohibited_phrases:
        phrase_lower = phrase.lower()
        idx = text_lower.find(phrase_lower)
        if idx != -1:
            findings.append({
                "type": "prohibited_phrase",
                "value": phrase,
                "start": idx,
                "severity": "high",
            })

    # Check profanity words (Unicode-aware word splitting)
    words = re.findall(r'[\w\u0900-\u097F\u0400-\u04FF]+', text_lower)
    for word in words:
        if word in profanity_words:
            findings.append({
                "type": "profanity",
                "value": word,
                "severity": "medium",
            })

    return findings


# ---------------------------------------------------------------------------
# OBLIGATION EXTRACTION (language-aware sentence splitting)
# ---------------------------------------------------------------------------

def _split_sentences_universal(text: str, language: str = "en") -> list[str]:
    """
    Split text into sentences. Uses spaCy for English,
    regex-based splitting for Hindi/Russian.
    """
    lang = _normalize_language(language, text)

    if lang == "en":
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Hindi uses '।' (purna viram) as period, plus standard '.', '?', '!'
    sentences = re.split(r'[।\.\?\!]+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_obligations(text: str, language: str = "en") -> list[dict]:
    """Extract sentences containing obligation keywords (language-aware)."""
    lang = _normalize_language(language, text)
    obligations = []

    if lang == "hi":
        keywords = OBLIGATION_KEYWORDS_HI + OBLIGATION_KEYWORDS_EN
    elif lang == "ru":
        keywords = OBLIGATION_KEYWORDS_RU + OBLIGATION_KEYWORDS_EN
    else:
        keywords = OBLIGATION_KEYWORDS_EN

    sentences = _split_sentences_universal(text, lang)

    for sent in sentences:
        sent_lower = sent.lower()
        matched_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            pattern = r'(?<!\w)' + re.escape(kw_lower) + r'(?!\w)'
            if re.search(pattern, sent_lower):
                matched_keywords.append(kw)

        if matched_keywords:
            idx = text.find(sent)
            obligations.append({
                "sentence": sent,
                "keywords": matched_keywords,
                "start": idx if idx != -1 else 0,
                "end": (idx + len(sent)) if idx != -1 else len(sent),
            })

    return obligations


# ---------------------------------------------------------------------------
# HINDI TEXT SENTIMENT (simple lexicon)
# ---------------------------------------------------------------------------

HINDI_NEGATIVE_WORDS = {
    "झूठ", "झूठा", "झूठी", "गलत", "बुरा", "खराब", "परेशान",
    "शिकायत", "दिक्कत", "समस्या", "नाराज़", "नाराज", "गुस्सा",
    "धमकी", "डर", "डरा", "लूट", "चोरी", "फ्रॉड", "fraud",
    "scam", "स्कैम", "ठगी", "धोखा",
}

HINDI_POSITIVE_WORDS = {
    "अच्छा", "बढ़िया", "सही", "ठीक", "धन्यवाद", "शुक्रिया",
    "मदद", "सहायता", "खुश", "संतुष्ट",
}


def analyze_text_sentiment(text: str, language: str = "en") -> dict:
    """Simple lexicon-based sentiment for Hindi."""
    lang = _normalize_language(language, text)

    if lang != "hi":
        return {"sentiment": "neutral", "negative_words": [], "positive_words": []}

    words = set(re.findall(r'[\w\u0900-\u097F]+', text.lower()))
    neg = words & HINDI_NEGATIVE_WORDS
    pos = words & HINDI_POSITIVE_WORDS

    if len(neg) > len(pos) + 1:
        sentiment = "negative"
    elif len(pos) > len(neg) + 1:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "negative_words": list(neg),
        "positive_words": list(pos),
    }


# ---------------------------------------------------------------------------
# POLICY LOADER
# ---------------------------------------------------------------------------

def load_policy_rules(policy_dir: str = "../data/policies") -> dict:
    """Load policy documents for reference."""
    rules = {}
    policy_path = Path(policy_dir)
    if policy_path.exists():
        for f in policy_path.iterdir():
            if f.suffix in (".txt", ".ttx"):
                rules[f.stem] = f.read_text(encoding="utf-8", errors="ignore")
    return rules


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def run_layer2(transcript: str, language: str = "en") -> dict:
    """
    Run complete Layer 2 pipeline on transcript text.
    
    Language routing:
    - English: Presidio (ML) + regex PII + spaCy NER + EN patterns
    - Hindi:   Regex-only PII + Hindi financial patterns + Hindi profanity/obligations
    - Russian: Regex-only PII + Russian financial patterns + Russian profanity/obligations
    """
    lang = _normalize_language(language, transcript)

    print(f"[Layer 2] Language: {lang} (input: '{language}')")
    print(f"[Layer 2] Transcript length: {len(transcript)} chars")

    # 1. PII Detection (language-aware routing)
    pii = detect_pii(transcript, language=lang)

    # 2. Financial Entity Extraction
    financial_entities = extract_financial_entities(transcript, language=lang)

    # 3. Named Entity Recognition (English only)
    named_entities = extract_named_entities(transcript, language=lang)

    # 4. Profanity / Prohibited Phrases
    profanity = detect_profanity(transcript, language=lang)

    # 5. Obligation Keywords
    obligations = extract_obligations(transcript, language=lang)

    # 6. Quick sentiment (Hindi lexicon)
    sentiment = analyze_text_sentiment(transcript, language=lang)

    # Risk summary
    risk_level = "low"
    if len(pii) > 0:
        risk_level = "medium"
    if any(p.get("severity") == "high" for p in profanity) or len(pii) > 3:
        risk_level = "high"

    print(f"[Layer 2] Results — PII: {len(pii)}, Financial: {len(financial_entities)}, "
          f"NER: {len(named_entities)}, Profanity: {len(profanity)}, "
          f"Obligations: {len(obligations)}, Risk: {risk_level}")

    return {
        "layer": "text_processing",
        "language_detected": lang,
        "pii_detected": pii,
        "pii_count": len(pii),
        "financial_entities": financial_entities,
        "named_entities": named_entities,
        "profanity_findings": profanity,
        "obligation_sentences": obligations,
        "text_sentiment": sentiment,
        "risk_level": risk_level,
    }
