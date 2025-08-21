#!/usr/bin/env python3
"""
Test script to verify API key configuration
"""

import os
import sys
from config_loader import get_openrouter_api_key, get_config_value

def test_api_keys():
    """Test if API keys are properly configured"""
    print("🔍 Testing API Key Configuration...")
    print("=" * 50)
    
    # Test OpenRouter API key
    print("\n1. Testing OpenRouter API Key:")
    openrouter_key = get_openrouter_api_key()
    if openrouter_key and openrouter_key != "YOUR_VALID_OPENROUTER_API_KEY_HERE":
        print(f"   ✅ OpenRouter API key found: {openrouter_key[:20]}...")
    else:
        print("   ❌ OpenRouter API key not configured properly")
        print("   📝 Please update config.json with your valid OpenRouter API key")
    
    # Test OpenAI API key
    print("\n2. Testing OpenAI API Key:")
    openai_key = get_config_value('llm.openai_api_key')
    if openai_key and openai_key != "YOUR_OPENAI_API_KEY_HERE":
        print(f"   ✅ OpenAI API key found: {openai_key[:20]}...")
    else:
        print("   ⚠️  OpenAI API key not configured (optional)")
    
    print("\n" + "=" * 50)
    print("📋 Next Steps:")
    print("1. Get a valid OpenRouter API key from https://openrouter.ai/")
    print("2. Update the 'openrouter_api_key' in config.json")
    print("3. Restart your FastAPI application")
    print("4. Test your API endpoint again")

if __name__ == "__main__":
    test_api_keys()

