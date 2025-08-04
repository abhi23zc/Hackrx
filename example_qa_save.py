#!/usr/bin/env python3
"""
Example script to demonstrate how to use the QA save endpoint.
This script shows how to save question-answer pairs to MongoDB.
"""

import requests
import json
from typing import List

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "2931609bd36ec1a45cb577b3b831dc711c76ae157b3c6250c564284c93b062ff"

def save_qa_pairs(url: str, questions: List[str], answers: List[str]) -> dict:
    """
    Save question-answer pairs to MongoDB via the API endpoint.
    
    Args:
        url (str): URL of the PDF document
        questions (List[str]): List of questions
        answers (List[str]): List of corresponding answers
        
    Returns:
        dict: Response from the API
    """
    
    # Prepare the request payload
    payload = {
        "url": url,
        "questions": questions,
        "answers": answers
    }
    
    # Set up headers with API key
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/hackrx/save-qa",
            json=payload,
            headers=headers
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Return the response data
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

def main():
    """Main function to demonstrate the QA save functionality."""
    
    # Example data
    example_url = "https://example.com/sample-document.pdf"
    example_questions = [
        "What is the main topic of this document?",
        "What are the key findings?",
        "What recommendations are provided?"
    ]
    example_answers = [
        "The main topic is artificial intelligence and its applications in healthcare.",
        "The key findings include improved diagnostic accuracy and reduced processing time.",
        "The recommendations include implementing AI systems gradually and ensuring proper training."
    ]
    
    print("🚀 Example: Saving QA pairs to MongoDB")
    print(f"📄 URL: {example_url}")
    print(f"❓ Questions: {len(example_questions)}")
    print(f"✅ Answers: {len(example_answers)}")
    print("-" * 50)
    
    # Save the QA pairs
    result = save_qa_pairs(example_url, example_questions, example_answers)
    
    if result:
        print("✅ Success!")
        print(f"📊 Response: {json.dumps(result, indent=2)}")
    else:
        print("❌ Failed to save QA pairs")

if __name__ == "__main__":
    main() 