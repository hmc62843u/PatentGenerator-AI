# util_funcs.py
import requests
import streamlit as st
import json

# Configuration
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", "")
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# API URLs for alternative implementations
PATENT_ANALYSIS_API = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
TEXT_GENERATION_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

def analyze_patent_novelty(patent_description):
    """Alternative implementation using HuggingFace for patent novelty analysis"""
    if not HF_API_TOKEN:
        return "HuggingFace API token not configured"
    
    payload = {
        "inputs": f"Analyze the novelty of this patent concept: {patent_description}",
        "parameters": {
            "return_full_text": False,
            "max_new_tokens": 150
        }
    }
    
    try:
        response = requests.post(TEXT_GENERATION_API, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"API Error: {str(e)}"

def generate_technical_diagram_prompt(patent_concept):
    """Generate a prompt for creating technical diagrams"""
    prompt = f"""
    Based on this patent concept, create a detailed prompt for generating a technical architecture diagram:
    
    {patent_concept}
    
    The prompt should be suitable for AI image generation tools and should describe:
    - System components and their relationships
    - Data flow and processing steps
    - Key innovative elements
    - Technical interfaces
    """
    return prompt

def save_patent_concept(patent_data, filename=None):
    """Save patent concept to JSON file"""
    if not filename:
        from datetime import datetime
        filename = f"patent_concept_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(patent_data, f, indent=2)
    
    return filename

def load_patent_concept(filename):
    """Load patent concept from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def validate_patent_components(patent_description):
    """Validate that patent contains essential components"""
    required_components = [
        "technical description",
        "innovative features", 
        "applications",
        "implementation"
    ]
    
    validation_results = {}
    patent_lower = patent_description.lower()
    
    for component in required_components:
        validation_results[component] = component in patent_lower
    
    return validation_results

def get_patent_statistics():
    """Get statistics about generated patents"""
    try:
        with open("patent_usage_tracker.json", "r") as f:
            data = json.load(f)
        
        return {
            "total_patents": data.get("total_patents_generated", 0),
            "current_hour_usage": data.get("usage_count", 0),
            "usage_history": data.get("usage_history", [])
        }
    except FileNotFoundError:
        return {"total_patents": 0, "current_hour_usage": 0, "usage_history": []}