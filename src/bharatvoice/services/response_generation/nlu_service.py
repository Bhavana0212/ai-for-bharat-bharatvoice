"""
Natural Language Understanding (NLU) Service for BharatVoice Assistant.

This module implements comprehensive NLU capabilities specifically designed for
Indian cultural context, including intent recognition, entity extraction,
colloquial term understanding, and cultural context interpretation.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum

from bharatvoice.core.models import (
    Intent,
    Entity,
    LanguageCode,
    ConversationState,
    RegionalContextData,
    CulturalEvent,
    UserProfile
)


class IntentCategory(str, Enum):
    """Categories of user intents for Indian context."""
    
    # General intents
    GREETING = "greeting"
    FAREWELL = "farewell"
    HELP = "help"
    CONFIRMATION = "confirmation"
    NEGATION = "negation"
    
    # Information seeking
    WEATHER_INQUIRY = "weather_inquiry"
    TIME_INQUIRY = "time_inquiry"
    DATE_INQUIRY = "date_inquiry"
    NEWS_INQUIRY = "news_inquiry"
    
    # Transportation
    TRAIN_INQUIRY = "train_inquiry"
    BUS_INQUIRY = "bus_inquiry"
    FLIGHT_INQUIRY = "flight_inquiry"
    TRAFFIC_INQUIRY = "traffic_inquiry"
    
    # Cultural and festivals
    FESTIVAL_INQUIRY = "festival_inquiry"
    CULTURAL_EVENT = "cultural_event"
    RELIGIOUS_INQUIRY = "religious_inquiry"
    
    # Services
    FOOD_ORDER = "food_order"
    RIDE_BOOKING = "ride_booking"
    PAYMENT_UPI = "payment_upi"
    GOVERNMENT_SERVICE = "government_service"
    
    # Entertainment
    MUSIC_REQUEST = "music_request"
    CRICKET_SCORES = "cricket_scores"
    BOLLYWOOD_NEWS = "bollywood_news"
    
    # Local services
    HOSPITAL_INQUIRY = "hospital_inquiry"
    SHOPPING_INQUIRY = "shopping_inquiry"
    RESTAURANT_INQUIRY = "restaurant_inquiry"
    
    # Personal
    REMINDER_SET = "reminder_set"
    ALARM_SET = "alarm_set"
    CALL_REQUEST = "call_request"
    
    # Unknown
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """Types of entities for Indian context."""
    
    # Location entities
    CITY = "city"
    STATE = "state"
    LANDMARK = "landmark"
    PINCODE = "pincode"
    
    # Time entities
    DATE = "date"
    TIME = "time"
    DURATION = "duration"
    
    # Cultural entities
    FESTIVAL = "festival"
    DEITY = "deity"
    CULTURAL_TERM = "cultural_term"
    
    # Transportation
    TRAIN_NAME = "train_name"
    STATION = "station"
    ROUTE = "route"
    
    # Food and cuisine
    DISH = "dish"
    CUISINE_TYPE = "cuisine_type"
    RESTAURANT = "restaurant"
    
    # People and relationships
    PERSON_NAME = "person_name"
    RELATIONSHIP = "relationship"
    
    # Numbers and quantities
    NUMBER = "number"
    CURRENCY = "currency"
    QUANTITY = "quantity"
    
    # Services
    SERVICE_TYPE = "service_type"
    GOVERNMENT_DOC = "government_doc"
    
    # Entertainment
    MOVIE = "movie"
    ACTOR = "actor"
    SONG = "song"
    CRICKET_TEAM = "cricket_team"


class ColloquialTermMapper:
    """Maps colloquial Indian terms to standard meanings."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._colloquial_mappings = self._initialize_colloquial_terms()
        self._regional_variations = self._initialize_regional_variations()
    
    def _initialize_colloquial_terms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize colloquial term mappings."""
        return {
            # Greetings and common expressions
            "namaste": {"standard": "hello", "context": "respectful_greeting", "languages": ["hi", "en-IN"]},
            "namaskar": {"standard": "hello", "context": "respectful_greeting", "languages": ["hi", "mr"]},
            "adab": {"standard": "hello", "context": "respectful_greeting", "languages": ["ur", "hi"]},
            "sat sri akal": {"standard": "hello", "context": "sikh_greeting", "languages": ["pa"]},
            "vanakkam": {"standard": "hello", "context": "tamil_greeting", "languages": ["ta"]},
            "namaskara": {"standard": "hello", "context": "kannada_greeting", "languages": ["kn"]},
            
            # Family relationships
            "mummy": {"standard": "mother", "context": "family", "languages": ["hi", "en-IN"]},
            "papa": {"standard": "father", "context": "family", "languages": ["hi", "en-IN"]},
            "didi": {"standard": "elder_sister", "context": "family", "languages": ["hi"]},
            "bhai": {"standard": "brother", "context": "family", "languages": ["hi"]},
            "dada": {"standard": "elder_brother", "context": "family", "languages": ["hi", "bn"]},
            "nana": {"standard": "maternal_grandfather", "context": "family", "languages": ["hi"]},
            "nani": {"standard": "maternal_grandmother", "context": "family", "languages": ["hi"]},
            "chacha": {"standard": "paternal_uncle", "context": "family", "languages": ["hi"]},
            "mama": {"standard": "maternal_uncle", "context": "family", "languages": ["hi"]},
            "mausi": {"standard": "maternal_aunt", "context": "family", "languages": ["hi"]},
            
            # Food and dining
            "khana": {"standard": "food", "context": "dining", "languages": ["hi", "ur"]},
            "paani": {"standard": "water", "context": "dining", "languages": ["hi"]},
            "chai": {"standard": "tea", "context": "beverage", "languages": ["hi", "en-IN"]},
            "roti": {"standard": "bread", "context": "food", "languages": ["hi"]},
            "sabzi": {"standard": "vegetables", "context": "food", "languages": ["hi"]},
            "dal": {"standard": "lentils", "context": "food", "languages": ["hi"]},
            "mithai": {"standard": "sweets", "context": "food", "languages": ["hi"]},
            "nashta": {"standard": "breakfast", "context": "meal", "languages": ["hi"]},
            
            # Transportation
            "gaadi": {"standard": "vehicle", "context": "transport", "languages": ["hi"]},
            "rickshaw": {"standard": "auto_rickshaw", "context": "transport", "languages": ["hi", "en-IN"]},
            "tempo": {"standard": "shared_auto", "context": "transport", "languages": ["hi"]},
            "bus": {"standard": "bus", "context": "transport", "languages": ["hi", "en-IN"]},
            "train": {"standard": "train", "context": "transport", "languages": ["hi", "en-IN"]},
            
            # Money and shopping
            "paisa": {"standard": "money", "context": "currency", "languages": ["hi"]},
            "rupiya": {"standard": "rupee", "context": "currency", "languages": ["hi"]},
            "bazaar": {"standard": "market", "context": "shopping", "languages": ["hi", "ur"]},
            "dukan": {"standard": "shop", "context": "shopping", "languages": ["hi"]},
            
            # Time expressions
            "abhi": {"standard": "now", "context": "time", "languages": ["hi"]},
            "kal": {"standard": "yesterday_or_tomorrow", "context": "time", "languages": ["hi"]},
            "parso": {"standard": "day_after_tomorrow", "context": "time", "languages": ["hi"]},
            "subah": {"standard": "morning", "context": "time", "languages": ["hi"]},
            "shaam": {"standard": "evening", "context": "time", "languages": ["hi"]},
            "raat": {"standard": "night", "context": "time", "languages": ["hi"]},
            
            # Common expressions
            "accha": {"standard": "good_or_okay", "context": "agreement", "languages": ["hi"]},
            "theek hai": {"standard": "okay", "context": "agreement", "languages": ["hi"]},
            "kya baat": {"standard": "what_matter", "context": "inquiry", "languages": ["hi"]},
            "kaise ho": {"standard": "how_are_you", "context": "greeting", "languages": ["hi"]},
            "kya haal": {"standard": "how_are_things", "context": "greeting", "languages": ["hi"]},
            
            # Religious and cultural terms
            "mandir": {"standard": "temple", "context": "religious", "languages": ["hi"]},
            "masjid": {"standard": "mosque", "context": "religious", "languages": ["hi", "ur"]},
            "gurudwara": {"standard": "sikh_temple", "context": "religious", "languages": ["pa", "hi"]},
            "church": {"standard": "church", "context": "religious", "languages": ["en-IN"]},
            "puja": {"standard": "worship", "context": "religious", "languages": ["hi"]},
            "aarti": {"standard": "prayer_ritual", "context": "religious", "languages": ["hi"]},
            "prasad": {"standard": "blessed_food", "context": "religious", "languages": ["hi"]},
            
            # Festivals
            "diwali": {"standard": "festival_of_lights", "context": "festival", "languages": ["hi", "en-IN"]},
            "holi": {"standard": "festival_of_colors", "context": "festival", "languages": ["hi", "en-IN"]},
            "eid": {"standard": "islamic_festival", "context": "festival", "languages": ["ur", "hi"]},
            "dussehra": {"standard": "victory_festival", "context": "festival", "languages": ["hi"]},
            "ganpati": {"standard": "ganesha_festival", "context": "festival", "languages": ["hi", "mr"]},
            
            # Modern slang and expressions
            "yaar": {"standard": "friend", "context": "casual", "languages": ["hi", "en-IN"]},
            "boss": {"standard": "sir_or_friend", "context": "casual", "languages": ["en-IN", "hi"]},
            "bro": {"standard": "brother_friend", "context": "casual", "languages": ["en-IN"]},
            "dude": {"standard": "friend", "context": "casual", "languages": ["en-IN"]},
            "bindaas": {"standard": "carefree", "context": "attitude", "languages": ["hi"]},
            "jugaad": {"standard": "innovative_solution", "context": "problem_solving", "languages": ["hi"]},
            "timepass": {"standard": "leisure_activity", "context": "activity", "languages": ["hi", "en-IN"]},
        }
    
    def _initialize_regional_variations(self) -> Dict[str, Dict[str, str]]:
        """Initialize regional variations of common terms."""
        return {
            "water": {
                "hi": "paani",
                "ta": "thanni",
                "te": "neeru",
                "bn": "jol",
                "mr": "paani",
                "gu": "paani",
                "kn": "neeru",
                "ml": "vellam",
                "pa": "paani"
            },
            "food": {
                "hi": "khana",
                "ta": "saapadu",
                "te": "bhojanam",
                "bn": "khawa",
                "mr": "jevan",
                "gu": "khavanu",
                "kn": "oota",
                "ml": "bhojanam",
                "pa": "khana"
            },
            "house": {
                "hi": "ghar",
                "ta": "veedu",
                "te": "illu",
                "bn": "bari",
                "mr": "ghar",
                "gu": "ghar",
                "kn": "mane",
                "ml": "veedu",
                "pa": "ghar"
            }
        }
    
    async def map_colloquial_terms(self, text: str, language: LanguageCode) -> str:
        """Map colloquial terms to standard meanings."""
        try:
            mapped_text = text.lower()
            
            for colloquial, mapping in self._colloquial_mappings.items():
                if language.value in mapping["languages"]:
                    # Use word boundaries to avoid partial matches
                    pattern = r'\b' + re.escape(colloquial) + r'\b'
                    mapped_text = re.sub(pattern, mapping["standard"], mapped_text, flags=re.IGNORECASE)
            
            return mapped_text
            
        except Exception as e:
            self.logger.error(f"Error mapping colloquial terms: {e}")
            return text
    
    async def get_cultural_context(self, term: str) -> Optional[Dict[str, Any]]:
        """Get cultural context for a term."""
        try:
            term_lower = term.lower()
            if term_lower in self._colloquial_mappings:
                return {
                    "term": term,
                    "standard_meaning": self._colloquial_mappings[term_lower]["standard"],
                    "context": self._colloquial_mappings[term_lower]["context"],
                    "languages": self._colloquial_mappings[term_lower]["languages"]
                }
            return None
        except Exception as e:
            self.logger.error(f"Error getting cultural context: {e}")
            return None


class IndianEntityExtractor:
    """Extracts Indian-specific entities from text."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._entity_patterns = self._initialize_entity_patterns()
        self._indian_cities = self._initialize_indian_cities()
        self._indian_states = self._initialize_indian_states()
        self._festivals = self._initialize_festivals()
        self._dishes = self._initialize_indian_dishes()
        self._relationships = self._initialize_relationships()
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for entity extraction."""
        return {
            "pincode": [r'\b\d{6}\b'],
            "phone": [r'\b[6-9]\d{9}\b', r'\b\+91[6-9]\d{9}\b'],
            "currency": [r'₹\s*\d+(?:,\d+)*(?:\.\d+)?', r'rs\.?\s*\d+', r'rupees?\s*\d+'],
            "time": [r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b', r'\b\d{1,2}\s*(?:am|pm)\b'],
            "date": [r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', r'\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4}\b'],
            "train_number": [r'\b\d{5}\b'],
            "government_doc": [r'\baadhaar\s*(?:card|number)?\b', r'\bpan\s*(?:card|number)?\b', r'\bpassport\b', r'\bvoter\s*id\b', r'\bdriving\s*licen[cs]e\b']
        }
    
    def _initialize_indian_cities(self) -> Set[str]:
        """Initialize set of major Indian cities."""
        return {
            # Metro cities
            "mumbai", "delhi", "bangalore", "hyderabad", "ahmedabad", "chennai", "kolkata", "pune",
            # State capitals
            "lucknow", "jaipur", "bhopal", "gandhinagar", "thiruvananthapuram", "panaji", "shimla",
            "chandigarh", "dehradun", "ranchi", "raipur", "bhubaneswar", "guwahati", "imphal",
            "aizawl", "kohima", "gangtok", "itanagar", "dispur", "shillong", "agartala",
            # Major cities
            "kanpur", "nagpur", "indore", "thane", "visakhapatnam", "vadodara", "ghaziabad",
            "ludhiana", "agra", "nashik", "faridabad", "meerut", "rajkot", "kalyan", "vasai",
            "varanasi", "srinagar", "aurangabad", "dhanbad", "amritsar", "navi mumbai", "allahabad",
            "howrah", "gwalior", "jabalpur", "coimbatore", "vijayawada", "jodhpur", "madurai",
            "raipur", "kota", "guwahati", "chandigarh", "solapur", "hubli", "tiruchirappalli",
            "bareilly", "mysore", "tiruppur", "gurgaon", "aligarh", "jalandhar", "bhubaneswar",
            "salem", "warangal", "mira", "bhiwandi", "saharanpur", "gorakhpur", "bikaner",
            "amravati", "noida", "jamshedpur", "bhilai", "cuttack", "firozabad", "kochi",
            "nellore", "bhavnagar", "dehradun", "durgapur", "asansol", "rourkela", "nanded",
            "kolhapur", "ajmer", "akola", "gulbarga", "jamnagar", "ujjain", "loni", "siliguri",
            "jhansi", "ulhasnagar", "jammu", "sangli", "mangalore", "erode", "belgaum", "ambattur",
            "tirunelveli", "malegaon", "gaya", "jalgaon", "udaipur", "maheshtala"
        }
    
    def _initialize_indian_states(self) -> Set[str]:
        """Initialize set of Indian states and union territories."""
        return {
            # States
            "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
            "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
            "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
            "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
            "uttar pradesh", "uttarakhand", "west bengal",
            # Union Territories
            "andaman and nicobar islands", "chandigarh", "dadra and nagar haveli and daman and diu",
            "delhi", "jammu and kashmir", "ladakh", "lakshadweep", "puducherry"
        }
    
    def _initialize_festivals(self) -> Set[str]:
        """Initialize set of Indian festivals."""
        return {
            "diwali", "deepavali", "holi", "eid", "eid ul fitr", "eid ul adha", "dussehra",
            "vijayadashami", "ganesh chaturthi", "ganpati", "navratri", "durga puja",
            "karva chauth", "karwa chauth", "raksha bandhan", "janmashtami", "ram navami",
            "maha shivratri", "saraswati puja", "kali puja", "onam", "pongal", "baisakhi",
            "vaisakhi", "lohri", "makar sankranti", "gudi padwa", "ugadi", "vishu",
            "poila boishakh", "bihu", "teej", "chhath puja", "dhanteras", "bhai dooj",
            "govardhan puja", "akshaya tritiya", "hanuman jayanti", "guru nanak jayanti",
            "christmas", "good friday", "easter", "buddha purnima", "mahavir jayanti"
        }
    
    def _initialize_indian_dishes(self) -> Set[str]:
        """Initialize set of popular Indian dishes."""
        return {
            # North Indian
            "roti", "chapati", "naan", "paratha", "dal", "rajma", "chole", "paneer",
            "butter chicken", "chicken tikka", "biryani", "pulao", "samosa", "pakora",
            "aloo gobi", "palak paneer", "dal makhani", "tandoori chicken", "kebab",
            # South Indian
            "dosa", "idli", "vada", "sambar", "rasam", "uttapam", "appam", "puttu",
            "coconut rice", "lemon rice", "curd rice", "fish curry", "chicken curry",
            "masala dosa", "rava dosa", "medu vada", "upma", "pongal",
            # Snacks and sweets
            "chaat", "bhel puri", "pani puri", "gol gappa", "aloo tikki", "dhokla",
            "kachori", "jalebi", "gulab jamun", "rasgulla", "sandesh", "laddu",
            "barfi", "halwa", "kheer", "kulfi", "falooda", "lassi", "chai",
            # Regional specialties
            "vada pav", "pav bhaji", "misal pav", "poha", "upma", "thali", "kadhi",
            "khichdi", "paratha", "stuffed paratha", "makki ki roti", "sarson ka saag"
        }
    
    def _initialize_relationships(self) -> Set[str]:
        """Initialize Indian family relationships."""
        return {
            "mummy", "mama", "papa", "dad", "bhai", "brother", "sister", "didi", "behan",
            "dada", "nana", "nani", "dadi", "chacha", "tau", "mausa", "mama", "mausi",
            "chachi", "tayi", "bua", "mami", "sasur", "saas", "jeth", "devar", "nanad",
            "bhabhi", "jija", "sala", "saali", "beta", "beti", "grandson", "granddaughter",
            "nephew", "niece", "cousin", "husband", "wife", "pati", "patni", "dulha", "dulhan"
        }
    
    async def extract_entities(self, text: str, language: LanguageCode) -> List[Entity]:
        """Extract entities from text."""
        try:
            entities = []
            text_lower = text.lower()
            
            # Extract pattern-based entities
            for entity_type, patterns in self._entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity = Entity(
                            name=entity_type,
                            value=match.group(),
                            type=entity_type,
                            confidence=0.9,
                            start_pos=match.start(),
                            end_pos=match.end()
                        )
                        entities.append(entity)
            
            # Extract cities
            for city in self._indian_cities:
                if city in text_lower:
                    start_pos = text_lower.find(city)
                    entity = Entity(
                        name="city",
                        value=city.title(),
                        type=EntityType.CITY.value,
                        confidence=0.85,
                        start_pos=start_pos,
                        end_pos=start_pos + len(city)
                    )
                    entities.append(entity)
            
            # Extract states
            for state in self._indian_states:
                if state in text_lower:
                    start_pos = text_lower.find(state)
                    entity = Entity(
                        name="state",
                        value=state.title(),
                        type=EntityType.STATE.value,
                        confidence=0.85,
                        start_pos=start_pos,
                        end_pos=start_pos + len(state)
                    )
                    entities.append(entity)
            
            # Extract festivals
            for festival in self._festivals:
                if festival in text_lower:
                    start_pos = text_lower.find(festival)
                    entity = Entity(
                        name="festival",
                        value=festival.title(),
                        type=EntityType.FESTIVAL.value,
                        confidence=0.9,
                        start_pos=start_pos,
                        end_pos=start_pos + len(festival)
                    )
                    entities.append(entity)
            
            # Extract dishes
            for dish in self._dishes:
                if dish in text_lower:
                    start_pos = text_lower.find(dish)
                    entity = Entity(
                        name="dish",
                        value=dish.title(),
                        type=EntityType.DISH.value,
                        confidence=0.8,
                        start_pos=start_pos,
                        end_pos=start_pos + len(dish)
                    )
                    entities.append(entity)
            
            # Extract relationships
            for relationship in self._relationships:
                if relationship in text_lower:
                    start_pos = text_lower.find(relationship)
                    entity = Entity(
                        name="relationship",
                        value=relationship.title(),
                        type=EntityType.RELATIONSHIP.value,
                        confidence=0.8,
                        start_pos=start_pos,
                        end_pos=start_pos + len(relationship)
                    )
                    entities.append(entity)
            
            # Remove duplicate entities (same position)
            unique_entities = []
            seen_positions = set()
            for entity in entities:
                pos_key = (entity.start_pos, entity.end_pos, entity.type)
                if pos_key not in seen_positions:
                    unique_entities.append(entity)
                    seen_positions.add(pos_key)
            
            return unique_entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []


class IndianIntentClassifier:
    """Classifies user intents with Indian cultural context."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._intent_patterns = self._initialize_intent_patterns()
        self._cultural_keywords = self._initialize_cultural_keywords()
    
    def _initialize_intent_patterns(self) -> Dict[IntentCategory, List[str]]:
        """Initialize intent classification patterns."""
        return {
            IntentCategory.GREETING: [
                r'\b(?:namaste|namaskar|hello|hi|hey|good morning|good evening|adab|sat sri akal|vanakkam)\b',
                r'\b(?:kaise ho|kya haal|how are you|what\'s up)\b'
            ],
            IntentCategory.FAREWELL: [
                r'\b(?:bye|goodbye|alvida|see you|take care|good night|ram ram)\b',
                r'\b(?:phir milenge|see you later|catch you later)\b'
            ],
            IntentCategory.WEATHER_INQUIRY: [
                r'\b(?:weather|mausam|barish|rain|temperature|garmi|sardi|cold|hot)\b',
                r'\b(?:how is the weather|what\'s the weather|weather forecast|monsoon)\b'
            ],
            IntentCategory.TRAIN_INQUIRY: [
                r'\b(?:train|railway|irctc|station|platform|ticket|reservation)\b',
                r'\b(?:train schedule|train timing|book train|train status)\b'
            ],
            IntentCategory.FESTIVAL_INQUIRY: [
                r'\b(?:festival|tyohar|celebration|diwali|holi|eid|dussehra|ganpati)\b',
                r'\b(?:when is|festival date|celebration|puja|aarti)\b'
            ],
            IntentCategory.FOOD_ORDER: [
                r'\b(?:food|khana|order|delivery|restaurant|hotel|dhaba)\b',
                r'\b(?:hungry|bhookh|khana chahiye|food delivery|zomato|swiggy)\b'
            ],
            IntentCategory.RIDE_BOOKING: [
                r'\b(?:cab|taxi|auto|rickshaw|ola|uber|ride|book)\b',
                r'\b(?:need a ride|book cab|auto chahiye|taxi book karo)\b'
            ],
            IntentCategory.PAYMENT_UPI: [
                r'\b(?:payment|paisa|money|upi|paytm|gpay|phonepe|pay)\b',
                r'\b(?:send money|payment karo|paisa bhejo|transfer)\b'
            ],
            IntentCategory.CRICKET_SCORES: [
                r'\b(?:cricket|score|match|ipl|team india|world cup)\b',
                r'\b(?:cricket score|match result|who won|live score)\b'
            ],
            IntentCategory.BOLLYWOOD_NEWS: [
                r'\b(?:bollywood|movie|film|actor|actress|cinema)\b',
                r'\b(?:latest movie|film news|bollywood news|new release)\b'
            ],
            IntentCategory.TIME_INQUIRY: [
                r'\b(?:time|samay|what time|kitna baja|clock)\b',
                r'\b(?:current time|time kya hai|what\'s the time)\b'
            ],
            IntentCategory.GOVERNMENT_SERVICE: [
                r'\b(?:aadhaar|pan card|passport|license|government|sarkari)\b',
                r'\b(?:government service|document|certificate|apply)\b'
            ],
            IntentCategory.HOSPITAL_INQUIRY: [
                r'\b(?:hospital|doctor|medical|health|dawai|medicine)\b',
                r'\b(?:need doctor|hospital near|medical help|emergency)\b'
            ],
            IntentCategory.CONFIRMATION: [
                r'\b(?:yes|haan|ji haan|theek hai|accha|okay|right|correct)\b'
            ],
            IntentCategory.NEGATION: [
                r'\b(?:no|nahi|nahin|mat karo|don\'t|stop|cancel)\b'
            ],
            IntentCategory.HELP: [
                r'\b(?:help|madad|sahayata|guide|kaise|how to|what can you do)\b'
            ]
        }
    
    def _initialize_cultural_keywords(self) -> Dict[str, List[str]]:
        """Initialize cultural context keywords."""
        return {
            "religious": ["mandir", "masjid", "church", "gurudwara", "puja", "namaz", "prayer", "bhagwan", "allah", "god"],
            "family": ["mummy", "papa", "bhai", "didi", "nana", "nani", "chacha", "mama", "family", "ghar"],
            "food": ["khana", "chai", "roti", "dal", "sabzi", "mithai", "nashta", "dinner", "lunch"],
            "festival": ["diwali", "holi", "eid", "dussehra", "ganpati", "navratri", "celebration", "tyohar"],
            "transport": ["train", "bus", "auto", "rickshaw", "metro", "flight", "gaadi", "station"],
            "shopping": ["bazaar", "market", "dukan", "shop", "mall", "buy", "purchase", "kharidna"],
            "entertainment": ["movie", "film", "cricket", "music", "dance", "party", "fun", "masti"]
        }
    
    async def classify_intent(self, text: str, context: Optional[ConversationState] = None) -> Intent:
        """Classify user intent from text."""
        try:
            text_lower = text.lower()
            best_intent = IntentCategory.UNKNOWN
            best_confidence = 0.0
            intent_parameters = {}
            
            # Check each intent category
            for intent_category, patterns in self._intent_patterns.items():
                confidence = 0.0
                matches = 0
                
                for pattern in patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        matches += 1
                        confidence += 0.3
                
                # Boost confidence based on cultural keywords
                for cultural_context, keywords in self._cultural_keywords.items():
                    for keyword in keywords:
                        if keyword in text_lower:
                            if self._is_intent_related_to_context(intent_category, cultural_context):
                                confidence += 0.1
                
                # Normalize confidence
                confidence = min(confidence, 1.0)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_intent = intent_category
            
            # Add context-based adjustments
            if context and context.conversation_history:
                last_interaction = context.conversation_history[-1]
                if last_interaction.intent:
                    # Boost related intents
                    if self._are_intents_related(best_intent, last_interaction.intent):
                        best_confidence = min(best_confidence + 0.1, 1.0)
            
            # Set minimum confidence threshold
            if best_confidence < 0.3:
                best_intent = IntentCategory.UNKNOWN
                best_confidence = 0.1
            
            return Intent(
                name=best_intent.value,
                confidence=best_confidence,
                category=best_intent.value,
                parameters=intent_parameters
            )
            
        except Exception as e:
            self.logger.error(f"Error classifying intent: {e}")
            return Intent(
                name=IntentCategory.UNKNOWN.value,
                confidence=0.1,
                category=IntentCategory.UNKNOWN.value,
                parameters={}
            )
    
    def _is_intent_related_to_context(self, intent: IntentCategory, context: str) -> bool:
        """Check if intent is related to cultural context."""
        relations = {
            IntentCategory.FESTIVAL_INQUIRY: ["religious", "festival"],
            IntentCategory.FOOD_ORDER: ["food"],
            IntentCategory.TRAIN_INQUIRY: ["transport"],
            IntentCategory.RIDE_BOOKING: ["transport"],
            IntentCategory.BOLLYWOOD_NEWS: ["entertainment"],
            IntentCategory.CRICKET_SCORES: ["entertainment"],
            IntentCategory.HOSPITAL_INQUIRY: ["family"],
            IntentCategory.SHOPPING_INQUIRY: ["shopping"]
        }
        
        return context in relations.get(intent, [])
    
    def _are_intents_related(self, intent1: IntentCategory, intent2: str) -> bool:
        """Check if two intents are related."""
        related_groups = [
            [IntentCategory.TRAIN_INQUIRY, IntentCategory.BUS_INQUIRY, IntentCategory.RIDE_BOOKING],
            [IntentCategory.FOOD_ORDER, IntentCategory.RESTAURANT_INQUIRY],
            [IntentCategory.WEATHER_INQUIRY, IntentCategory.TIME_INQUIRY, IntentCategory.DATE_INQUIRY],
            [IntentCategory.FESTIVAL_INQUIRY, IntentCategory.CULTURAL_EVENT, IntentCategory.RELIGIOUS_INQUIRY],
            [IntentCategory.CRICKET_SCORES, IntentCategory.BOLLYWOOD_NEWS, IntentCategory.MUSIC_REQUEST]
        ]
        
        for group in related_groups:
            if intent1 in group and intent2 in [i.value for i in group]:
                return True
        
        return False


class CulturalContextInterpreter:
    """Interprets cultural context from user input."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cultural_patterns = self._initialize_cultural_patterns()
        self._regional_contexts = self._initialize_regional_contexts()
    
    def _initialize_cultural_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural interpretation patterns."""
        return {
            "respect_indicators": {
                "patterns": [r'\bji\b', r'\bsir\b', r'\bmadam\b', r'\bsahab\b', r'\bsahib\b'],
                "context": "formal_respectful",
                "response_style": "formal"
            },
            "casual_indicators": {
                "patterns": [r'\byaar\b', r'\bbro\b', r'\bdude\b', r'\bboss\b'],
                "context": "casual_friendly",
                "response_style": "casual"
            },
            "family_context": {
                "patterns": [r'\bmummy\b', r'\bpapa\b', r'\bbhai\b', r'\bdidi\b', r'\bfamily\b'],
                "context": "family_oriented",
                "response_style": "warm_personal"
            },
            "religious_context": {
                "patterns": [r'\bmandir\b', r'\bmasjid\b', r'\bpuja\b', r'\bnamaz\b', r'\bprayer\b'],
                "context": "religious_spiritual",
                "response_style": "respectful_spiritual"
            },
            "festival_context": {
                "patterns": [r'\bdiwali\b', r'\bholi\b', r'\beid\b', r'\bfestival\b', r'\btyohar\b'],
                "context": "celebratory_festive",
                "response_style": "enthusiastic_cultural"
            },
            "urgency_indicators": {
                "patterns": [r'\bjaldi\b', r'\bquick\b', r'\burgent\b', r'\bemergency\b', r'\bturant\b'],
                "context": "urgent_immediate",
                "response_style": "quick_efficient"
            }
        }
    
    def _initialize_regional_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regional cultural contexts."""
        return {
            "north_india": {
                "states": ["delhi", "punjab", "haryana", "uttar pradesh", "rajasthan", "himachal pradesh"],
                "greetings": ["namaste", "sat sri akal", "adab"],
                "cultural_traits": ["direct_communication", "family_oriented", "festival_loving"],
                "common_terms": ["ji", "sahab", "yaar", "bhai"]
            },
            "south_india": {
                "states": ["tamil nadu", "karnataka", "kerala", "andhra pradesh", "telangana"],
                "greetings": ["vanakkam", "namaskara", "namasthe"],
                "cultural_traits": ["respectful_communication", "traditional_values", "education_focused"],
                "common_terms": ["sir", "madam", "anna", "akka"]
            },
            "west_india": {
                "states": ["maharashtra", "gujarat", "goa", "rajasthan"],
                "greetings": ["namaskar", "namaste"],
                "cultural_traits": ["business_oriented", "progressive", "cultural_pride"],
                "common_terms": ["sahab", "tai", "dada"]
            },
            "east_india": {
                "states": ["west bengal", "odisha", "jharkhand", "bihar"],
                "greetings": ["namaskar", "adab"],
                "cultural_traits": ["intellectual", "artistic", "traditional"],
                "common_terms": ["dada", "didi", "babu"]
            }
        }
    
    async def interpret_cultural_context(
        self, 
        text: str, 
        user_profile: Optional[UserProfile] = None,
        regional_context: Optional[RegionalContextData] = None
    ) -> Dict[str, Any]:
        """Interpret cultural context from text and user information."""
        try:
            cultural_context = {
                "communication_style": "neutral",
                "formality_level": "medium",
                "cultural_references": [],
                "regional_influence": None,
                "response_tone": "helpful",
                "cultural_sensitivity": []
            }
            
            text_lower = text.lower()
            
            # Analyze communication patterns
            for pattern_name, pattern_info in self._cultural_patterns.items():
                for pattern in pattern_info["patterns"]:
                    if re.search(pattern, text_lower):
                        cultural_context["communication_style"] = pattern_info["context"]
                        cultural_context["response_tone"] = pattern_info["response_style"]
                        break
            
            # Determine formality level
            formal_indicators = ["sir", "madam", "sahab", "ji", "please", "kindly"]
            casual_indicators = ["yaar", "bro", "dude", "boss", "hey"]
            
            formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
            casual_count = sum(1 for indicator in casual_indicators if indicator in text_lower)
            
            if formal_count > casual_count:
                cultural_context["formality_level"] = "high"
            elif casual_count > formal_count:
                cultural_context["formality_level"] = "low"
            
            # Add regional influence
            if regional_context and regional_context.location:
                state = regional_context.location.state.lower()
                for region, info in self._regional_contexts.items():
                    if state in info["states"]:
                        cultural_context["regional_influence"] = region
                        cultural_context["cultural_references"] = info["common_terms"]
                        break
            
            # Add user profile influence
            if user_profile:
                if user_profile.location:
                    state = user_profile.location.state.lower()
                    for region, info in self._regional_contexts.items():
                        if state in info["states"]:
                            cultural_context["regional_influence"] = region
                            break
            
            # Identify cultural sensitivity areas
            sensitive_topics = []
            if any(term in text_lower for term in ["religion", "caste", "politics"]):
                sensitive_topics.append("handle_with_care")
            if any(term in text_lower for term in ["festival", "celebration", "puja"]):
                sensitive_topics.append("cultural_celebration")
            if any(term in text_lower for term in ["family", "marriage", "relationship"]):
                sensitive_topics.append("personal_family")
            
            cultural_context["cultural_sensitivity"] = sensitive_topics
            
            return cultural_context
            
        except Exception as e:
            self.logger.error(f"Error interpreting cultural context: {e}")
            return {
                "communication_style": "neutral",
                "formality_level": "medium",
                "cultural_references": [],
                "regional_influence": None,
                "response_tone": "helpful",
                "cultural_sensitivity": []
            }


class NLUService:
    """
    Main Natural Language Understanding service for BharatVoice Assistant.
    
    Integrates intent classification, entity extraction, colloquial term mapping,
    and cultural context interpretation for Indian users.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.colloquial_mapper = ColloquialTermMapper()
        self.entity_extractor = IndianEntityExtractor()
        self.intent_classifier = IndianIntentClassifier()
        self.cultural_interpreter = CulturalContextInterpreter()
    
    async def process_user_input(
        self,
        text: str,
        language: LanguageCode,
        conversation_state: Optional[ConversationState] = None,
        user_profile: Optional[UserProfile] = None,
        regional_context: Optional[RegionalContextData] = None
    ) -> Dict[str, Any]:
        """
        Process user input through complete NLU pipeline.
        
        Args:
            text: User input text
            language: Input language
            conversation_state: Current conversation context
            user_profile: User profile information
            regional_context: Regional context data
            
        Returns:
            Complete NLU analysis result
        """
        try:
            self.logger.info(f"Processing user input: '{text[:50]}...' in language {language}")
            
            # Step 1: Map colloquial terms
            mapped_text = await self.colloquial_mapper.map_colloquial_terms(text, language)
            
            # Step 2: Extract entities
            entities = await self.entity_extractor.extract_entities(mapped_text, language)
            
            # Step 3: Classify intent
            intent = await self.intent_classifier.classify_intent(mapped_text, conversation_state)
            
            # Step 4: Interpret cultural context
            cultural_context = await self.cultural_interpreter.interpret_cultural_context(
                text, user_profile, regional_context
            )
            
            # Step 5: Enhance entities with cultural context
            enhanced_entities = await self._enhance_entities_with_context(
                entities, cultural_context, regional_context
            )
            
            # Step 6: Adjust intent confidence based on cultural context
            adjusted_intent = await self._adjust_intent_with_context(
                intent, cultural_context, conversation_state
            )
            
            nlu_result = {
                "original_text": text,
                "processed_text": mapped_text,
                "language": language,
                "intent": adjusted_intent,
                "entities": enhanced_entities,
                "cultural_context": cultural_context,
                "confidence": adjusted_intent.confidence,
                "processing_metadata": {
                    "colloquial_terms_mapped": text != mapped_text,
                    "entities_count": len(enhanced_entities),
                    "cultural_indicators_found": len(cultural_context.get("cultural_references", [])),
                    "regional_influence": cultural_context.get("regional_influence"),
                    "formality_level": cultural_context.get("formality_level")
                }
            }
            
            self.logger.info(f"NLU processing completed. Intent: {adjusted_intent.name}, Confidence: {adjusted_intent.confidence:.2f}")
            return nlu_result
            
        except Exception as e:
            self.logger.error(f"Error in NLU processing: {e}")
            return {
                "original_text": text,
                "processed_text": text,
                "language": language,
                "intent": Intent(
                    name=IntentCategory.UNKNOWN.value,
                    confidence=0.1,
                    category=IntentCategory.UNKNOWN.value,
                    parameters={}
                ),
                "entities": [],
                "cultural_context": {},
                "confidence": 0.1,
                "processing_metadata": {
                    "error": str(e),
                    "colloquial_terms_mapped": False,
                    "entities_count": 0,
                    "cultural_indicators_found": 0
                }
            }
    
    async def _enhance_entities_with_context(
        self,
        entities: List[Entity],
        cultural_context: Dict[str, Any],
        regional_context: Optional[RegionalContextData]
    ) -> List[Entity]:
        """Enhance entities with cultural and regional context."""
        try:
            enhanced_entities = []
            
            for entity in entities:
                enhanced_entity = entity.copy()
                
                # Add cultural context to entities
                if entity.type == EntityType.FESTIVAL.value:
                    # Add festival significance
                    cultural_info = await self.colloquial_mapper.get_cultural_context(entity.value)
                    if cultural_info:
                        enhanced_entity.name = f"{entity.name}_with_context"
                        enhanced_entity.confidence = min(entity.confidence + 0.1, 1.0)
                
                elif entity.type == EntityType.CITY.value and regional_context:
                    # Boost confidence for local cities
                    if entity.value.lower() == regional_context.location.city.lower():
                        enhanced_entity.confidence = min(entity.confidence + 0.2, 1.0)
                
                elif entity.type == EntityType.RELATIONSHIP.value:
                    # Add family context
                    if cultural_context.get("communication_style") == "family_oriented":
                        enhanced_entity.confidence = min(entity.confidence + 0.1, 1.0)
                
                enhanced_entities.append(enhanced_entity)
            
            return enhanced_entities
            
        except Exception as e:
            self.logger.error(f"Error enhancing entities: {e}")
            return entities
    
    async def _adjust_intent_with_context(
        self,
        intent: Intent,
        cultural_context: Dict[str, Any],
        conversation_state: Optional[ConversationState]
    ) -> Intent:
        """Adjust intent confidence based on cultural context."""
        try:
            adjusted_intent = intent.copy()
            
            # Boost confidence for culturally appropriate intents
            if cultural_context.get("communication_style") == "formal_respectful":
                if intent.name in ["government_service", "hospital_inquiry", "help"]:
                    adjusted_intent.confidence = min(intent.confidence + 0.1, 1.0)
            
            elif cultural_context.get("communication_style") == "casual_friendly":
                if intent.name in ["food_order", "ride_booking", "cricket_scores"]:
                    adjusted_intent.confidence = min(intent.confidence + 0.1, 1.0)
            
            # Adjust based on conversation history
            if conversation_state and conversation_state.conversation_history:
                recent_intents = [interaction.intent for interaction in conversation_state.conversation_history[-3:]]
                if intent.name in recent_intents:
                    adjusted_intent.confidence = min(intent.confidence + 0.05, 1.0)
            
            return adjusted_intent
            
        except Exception as e:
            self.logger.error(f"Error adjusting intent: {e}")
            return intent
    
    async def get_intent_suggestions(
        self,
        partial_text: str,
        language: LanguageCode,
        context: Optional[ConversationState] = None
    ) -> List[Dict[str, Any]]:
        """Get intent suggestions for partial user input."""
        try:
            suggestions = []
            
            # Get potential intents based on partial text
            potential_intents = await self.intent_classifier.classify_intent(partial_text, context)
            
            # Add common follow-up intents based on conversation history
            if context and context.conversation_history:
                last_intent = context.conversation_history[-1].intent
                if last_intent == "weather_inquiry":
                    suggestions.extend([
                        {"intent": "time_inquiry", "confidence": 0.7, "suggestion": "What time is it?"},
                        {"intent": "festival_inquiry", "confidence": 0.6, "suggestion": "Any festivals coming up?"}
                    ])
                elif last_intent == "train_inquiry":
                    suggestions.extend([
                        {"intent": "weather_inquiry", "confidence": 0.6, "suggestion": "How's the weather?"},
                        {"intent": "food_order", "confidence": 0.5, "suggestion": "Order some food?"}
                    ])
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting intent suggestions: {e}")
            return []
    
    async def validate_cultural_appropriateness(
        self,
        text: str,
        intent: Intent,
        cultural_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate cultural appropriateness of the interaction."""
        try:
            validation_result = {
                "is_appropriate": True,
                "concerns": [],
                "suggestions": []
            }
            
            # Check for cultural sensitivity
            sensitive_areas = cultural_context.get("cultural_sensitivity", [])
            
            if "handle_with_care" in sensitive_areas:
                validation_result["concerns"].append("Sensitive topic detected")
                validation_result["suggestions"].append("Respond with cultural sensitivity")
            
            if intent.name == "religious_inquiry" and cultural_context.get("formality_level") == "low":
                validation_result["concerns"].append("Religious topic with casual tone")
                validation_result["suggestions"].append("Use respectful tone for religious topics")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating cultural appropriateness: {e}")
            return {"is_appropriate": True, "concerns": [], "suggestions": []}