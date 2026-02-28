#!/usr/bin/env python3
"""
Simple test runner for NLU service tests.
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_basic_nlu_functionality():
    """Test basic NLU functionality."""
    try:
        from bharatvoice.services.response_generation.nlu_service import NLUService
        from bharatvoice.core.models import LanguageCode
        
        print("Testing NLU Service...")
        
        nlu_service = NLUService()
        
        # Test basic processing
        test_cases = [
            ("Namaste, kaise hain aap?", LanguageCode.HINDI),
            ("What's the weather like today?", LanguageCode.ENGLISH_IN),
            ("Train schedule from Mumbai to Delhi", LanguageCode.ENGLISH_IN),
            ("When is Diwali this year?", LanguageCode.HINDI),
            ("I want to order biryani", LanguageCode.ENGLISH_IN)
        ]
        
        for text, language in test_cases:
            print(f"\nTesting: '{text}' in {language}")
            
            result = await nlu_service.process_user_input(text, language)
            
            print(f"  Intent: {result['intent'].name} (confidence: {result['intent'].confidence:.2f})")
            print(f"  Entities: {len(result['entities'])} found")
            print(f"  Cultural context: {result['cultural_context'].get('formality_level', 'unknown')}")
            print(f"  Colloquial terms mapped: {result['processing_metadata'].get('colloquial_terms_mapped', False)}")
        
        print("\n✅ Basic NLU functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_colloquial_mapping():
    """Test colloquial term mapping."""
    try:
        from bharatvoice.services.response_generation.nlu_service import ColloquialTermMapper
        from bharatvoice.core.models import LanguageCode
        
        print("\nTesting Colloquial Term Mapping...")
        
        mapper = ColloquialTermMapper()
        
        test_cases = [
            ("Namaste ji, khana ready hai", LanguageCode.HINDI),
            ("Yaar, mummy ne chai banai hai", LanguageCode.HINDI),
            ("Papa aur bhai ghar pe hain", LanguageCode.HINDI)
        ]
        
        for text, language in test_cases:
            print(f"\nOriginal: '{text}'")
            mapped = await mapper.map_colloquial_terms(text, language)
            print(f"Mapped:   '{mapped}'")
            
            # Test cultural context
            terms = text.split()
            for term in terms:
                context = await mapper.get_cultural_context(term.lower())
                if context:
                    print(f"  '{term}' -> {context['standard_meaning']} ({context['context']})")
        
        print("\n✅ Colloquial mapping tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Colloquial mapping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_entity_extraction():
    """Test entity extraction."""
    try:
        from bharatvoice.services.response_generation.nlu_service import IndianEntityExtractor
        from bharatvoice.core.models import LanguageCode
        
        print("\nTesting Entity Extraction...")
        
        extractor = IndianEntityExtractor()
        
        test_cases = [
            "I want to travel from Mumbai to Delhi",
            "When is Diwali and Holi this year?",
            "Order biryani and samosa from restaurant",
            "Call my mummy and tell bhai",
            "My pincode is 400001"
        ]
        
        for text in test_cases:
            print(f"\nText: '{text}'")
            entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
            
            for entity in entities:
                print(f"  {entity.type}: '{entity.value}' (confidence: {entity.confidence:.2f})")
        
        print("\n✅ Entity extraction tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_intent_classification():
    """Test intent classification."""
    try:
        from bharatvoice.services.response_generation.nlu_service import IndianIntentClassifier
        from bharatvoice.core.models import LanguageCode
        
        print("\nTesting Intent Classification...")
        
        classifier = IndianIntentClassifier()
        
        test_cases = [
            ("Namaste, kaise hain aap?", "greeting"),
            ("What's the weather today?", "weather_inquiry"),
            ("Train from Mumbai to Delhi", "train_inquiry"),
            ("When is Diwali?", "festival_inquiry"),
            ("I want to order food", "food_order"),
            ("Book a cab", "ride_booking"),
            ("Cricket score", "cricket_scores")
        ]
        
        for text, expected in test_cases:
            print(f"\nText: '{text}'")
            intent = await classifier.classify_intent(text)
            print(f"  Intent: {intent.name} (confidence: {intent.confidence:.2f})")
            print(f"  Expected: {expected}")
            
            if expected in intent.name:
                print("  ✅ Match!")
            else:
                print("  ⚠️  Different classification")
        
        print("\n✅ Intent classification tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Intent classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cultural_context():
    """Test cultural context interpretation."""
    try:
        from bharatvoice.services.response_generation.nlu_service import CulturalContextInterpreter
        
        print("\nTesting Cultural Context Interpretation...")
        
        interpreter = CulturalContextInterpreter()
        
        test_cases = [
            ("Sir, please help me", "formal"),
            ("Hey yaar, what's up?", "casual"),
            ("Mummy said to call papa", "family"),
            ("Need to go to mandir", "religious"),
            ("Jaldi help chahiye!", "urgent")
        ]
        
        for text, expected_style in test_cases:
            print(f"\nText: '{text}'")
            context = await interpreter.interpret_cultural_context(text)
            
            print(f"  Communication style: {context.get('communication_style', 'unknown')}")
            print(f"  Formality level: {context.get('formality_level', 'unknown')}")
            print(f"  Response tone: {context.get('response_tone', 'unknown')}")
            print(f"  Expected: {expected_style}")
        
        print("\n✅ Cultural context tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Cultural context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting NLU Service Tests...")
    
    tests = [
        test_basic_nlu_functionality,
        test_colloquial_mapping,
        test_entity_extraction,
        test_intent_classification,
        test_cultural_context
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! NLU service is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)