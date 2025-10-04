from transformers import pipeline
from google import genai
from google.genai import types
import wave
import os
import streamlit as st
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
import requests
import urllib.parse
from bs4 import BeautifulSoup
import re

# Load environment variables from Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets. Please configure it in the Secrets tab.")
    st.stop()

# Configure Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# Rate limiting functions
def init_usage_tracking():
    """Initialize the usage tracking file if it doesn't exist"""
    if not Path("patent_usage_tracker.json").exists():
        with open("patent_usage_tracker.json", "w") as f:
            json.dump({
                "usage_count": 0, 
                "last_reset": datetime.now().isoformat(), 
                "usage_history": [],
                "total_patents_generated": 0
            }, f)

def check_rate_limit():
    """Check if the rate limit has been exceeded (max 3 uses per hour)"""
    try:
        with open("patent_usage_tracker.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        init_usage_tracking()
        return True, 0
    
    # Check if we need to reset the counter (new hour)
    last_reset = datetime.fromisoformat(data["last_reset"])
    now = datetime.now()
    
    if now - last_reset >= timedelta(hours=1):
        # Reset the counter
        data["usage_count"] = 0
        data["last_reset"] = now.isoformat()
        data["usage_history"] = []
        
        with open("patent_usage_tracker.json", "w") as f:
            json.dump(data, f)
    
    # Check if limit exceeded
    if data["usage_count"] >= 3:
        return False, data["usage_count"]
    
    return True, data["usage_count"]

def increment_usage(patent_title=""):
    """Increment the usage counter"""
    try:
        with open("patent_usage_tracker.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        init_usage_tracking()
        with open("patent_usage_tracker.json", "r") as f:
            data = json.load(f)
    
    # Check if we need to reset the counter (new hour)
    last_reset = datetime.fromisoformat(data["last_reset"])
    now = datetime.now()
    
    if now - last_reset >= timedelta(hours=1):
        # Reset the counter
        data["usage_count"] = 0
        data["last_reset"] = now.isoformat()
        data["usage_history"] = []
    
    # Increment usage
    data["usage_count"] += 1
    data["total_patents_generated"] = data.get("total_patents_generated", 0) + 1
    data["usage_history"].append({
        "timestamp": now.isoformat(),
        "action": "generate_patent",
        "patent_title": patent_title
    })
    
    with open("patent_usage_tracker.json", "w") as f:
        json.dump(data, f)
    
    return data["usage_count"], data["total_patents_generated"]

# Website scraping and analysis functions
def scrape_website_content(url):
    """Scrape and extract meaningful content from a website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract meaningful content
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title found"
        
        # Get main content - try multiple strategies
        content_parts = []
        
        # Try to get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            content_parts.append(f"Meta Description: {meta_desc.get('content', '').strip()}")
        
        # Get headings
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            heading_text = heading.get_text().strip()
            if heading_text and len(heading_text) > 10:
                content_parts.append(f"Heading: {heading_text}")
        
        # Get paragraph content
        paragraphs = soup.find_all('p')
        for p in paragraphs[:10]:  # Limit to first 10 paragraphs
            text = p.get_text().strip()
            if len(text) > 50:  # Only include substantial paragraphs
                content_parts.append(text)
        
        # Combine all content
        full_content = f"Website Title: {title_text}\n\n" + "\n\n".join(content_parts[:15])  # Limit total content
        
        return full_content[:4000]  # Limit content length
    
    except Exception as e:
        return f"Error scraping website: {str(e)}"

def analyze_website_with_gemini(url, content):
    """Use Gemini to analyze the website and extract key information for patent generation"""
    prompt = f"""
    Analyze this website content and extract key information for generating a patent concept:
    
    WEBSITE URL: {url}
    WEBSITE CONTENT:
    {content}
    
    Please analyze and provide:
    1. **Technology Domain**: What main technology field does this website represent?
    2. **Core Problem**: What problem is this website/company trying to solve?
    3. **Key Innovations**: What innovative approaches or technologies are mentioned?
    4. **Business Context**: What industry/market does this operate in?
    5. **Technical Gaps**: What potential technical challenges or limitations might exist?
    
    Be concise but comprehensive. Focus on identifying patentable opportunities.
    """
    
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    
    return response.text

def generate_patent_from_analysis(website_analysis, url):
    """Generate a patent concept based on website analysis"""
    prompt = f"""
    Based on this website analysis, create a compelling patent concept:
    
    WEBSITE ANALYSIS:
    {website_analysis}
    
    Generate a comprehensive patent concept including:
    1. **Novel Patent Title**: Creative and descriptive
    2. **Technical Description**: Detailed explanation of the invention
    3. **Key Innovative Features**: What makes this novel and non-obvious
    4. **Technical Implementation**: How it would be built/implemented
    5. **Potential Applications**: Where and how it could be used
    6. **Competitive Advantages**: Why this is better than existing solutions
    
    Make it technically sound, commercially viable, and genuinely innovative.
    Focus on solving the core problems identified in the website analysis.
    """
    
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    
    return response.text

def analyze_patent_strength(patent_description):
    """Analyze the strength and novelty of the generated patent"""
    prompt = f"""
    As a patent analyst, evaluate this patent concept:
    
    {patent_description}
    
    Please provide:
    1. Novelty score (1-10)
    2. Commercial potential (1-10)
    3. Technical feasibility (1-10)
    4. Key strengths
    5. Potential weaknesses
    6. Suggested improvements
    
    Format your response clearly with headings.
    """
    
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    
    return response.text

def generate_patent_claims(patent_description):
    """Generate formal patent claims"""
    prompt = f"""
    Based on this patent description, draft 3-5 formal patent claims:
    
    {patent_description}
    
    Format each claim as:
    "1. [Independent claim describing the core invention]"
    "2. [Dependent claim narrowing the first claim]"
    etc.
    
    Use proper patent claim language and structure.
    """
    
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    
    return response.text

# Form submission handler
def submit_inquiry_form(form_data):
    """Submit inquiry form to Formspree"""
    try:
        response = requests.post(
            "https://formspree.io/f/mqaybnnb",
            data=form_data,
            headers={'Accept': 'application/json'}
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error submitting form: {e}")
        return False

# UI Components
def render_wpatent_promotion():
    """Render W&Patent promotional elements"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Powered by W&Patent")
    st.sidebar.markdown("""
    **AI-Powered Patent Solutions:**
    - Website-to-Patent Analysis
    - Competitive Intelligence
    - Patent Landscape Mapping
    - Strength Analysis
    - Commercialization Support
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📧 Contact for Full Access")
    st.sidebar.markdown("""
    **Email:** wp@wpatent.com  
    **Website:** www.wpatent.com  
    **Plans:** Explorer, Innovator, Enterprise
    """)
    
    if st.sidebar.button("📩 Request Enterprise Demo"):
        st.sidebar.success("Contact wp@wpatent.com for a custom demo!")

def render_rate_limit_message(current_usage, is_allowed):
    """Render rate limit information with W&Patent promotion"""
    if not is_allowed:
        st.error("""
        ❌ **Rate Limit Exceeded!**
        
        This demo allows **3 patent generations per hour** across all users.
        
        🚀 **For unlimited access and enterprise features:**
        - **Unlimited website analysis**
        - **Competitive patent intelligence**
        - **Portfolio management**
        - **Commercialization support**
        
        📧 **Contact:** wp@wpatent.com
        """)
        
        try:
            with open("patent_usage_tracker.json", "r") as f:
                data = json.load(f)
            last_reset_time = datetime.fromisoformat(data["last_reset"])
            next_reset_time = last_reset_time + timedelta(hours=1)
            st.info(f"⏰ **Next reset:** {next_reset_time.strftime('%H:%M:%S')}")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            st.info("⏰ **Next reset:** In 1 hour")
        
        return False
    else:
        if current_usage >= 2:  # Warning when approaching limit
            st.warning(f"⚠️ **Usage Alert:** {current_usage}/3 generations used this hour. Contact wp@wpatent.com for unlimited access.")
        return True

def render_inquiry_form():
    """Render the inquiry form using Formspree"""
    st.markdown("---")
    st.subheader("📧 Interested in Full Access?")
    
    with st.form("inquiry_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *", placeholder="Enter your full name")
            email = st.text_input("Email Address *", placeholder="your.email@example.com")
            company = st.text_input("Company", placeholder="Your company name")
        
        with col2:
            interest = st.selectbox(
                "I'm interested in: *",
                [
                    "Select an option",
                    "Website-to-Patent Analysis",
                    "Competitive Intelligence",
                    "Patent Portfolio Management",
                    "Commercialization Support",
                    "Enterprise Solution",
                    "Other"
                ]
            )
            
            project_type = st.selectbox(
                "Project Type",
                [
                    "Select a project type",
                    "Competitor Analysis",
                    "Technology Landscape",
                    "Product Innovation",
                    "Market Expansion",
                    "Patent Strategy",
                    "Other"
                ]
            )
        
        message = st.text_area(
            "Message *", 
            placeholder="Tell us about your patent needs, specific websites you want to analyze, or any questions...",
            height=100
        )
        
        submitted = st.form_submit_button("🚀 Submit Inquiry", type="primary")
        
        if submitted:
            if not all([name, email, interest, message]) or interest == "Select an option":
                st.error("Please fill in all required fields (*)")
            else:
                form_data = {
                    "name": name,
                    "email": email,
                    "company": company,
                    "interest": interest,
                    "project_type": project_type,
                    "message": message,
                    "source": "WPatent AI Generator - Website Analysis"
                }
                
                if submit_inquiry_form(form_data):
                    st.success("✅ Thank you! Your inquiry has been submitted. We'll contact you within 24 hours.")
                else:
                    st.error("❌ There was an error submitting your form. Please email us directly at wp@wpatent.com")

def validate_url(url):
    """Validate and format URL"""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def main():
    st.set_page_config(
        page_title="W&Patent AI Website Analyzer", 
        page_icon="⚖️",
        layout="wide"
    )
    
    # Initialize usage tracking
    init_usage_tracking()
    
    # Main header with W&Patent branding
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("🌐 W&Patent AI Website Analyzer")
        st.caption("Transform Any Website into Patent Opportunities • AI-Powered Competitive Intelligence")
    with col2:
        st.markdown("""
        <div style='text-align: right; padding: 10px; background: #f0f2f6; border-radius: 5px;'>
            <strong>W&Patent</strong><br>
            <small>The World's Smarter Patent Marketplace</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Display usage information
    is_allowed, current_usage = check_rate_limit()
    
    # Sidebar content
    with st.sidebar:
        st.title("📊 Usage Dashboard")
        
        try:
            with open("patent_usage_tracker.json", "r") as f:
                data = json.load(f)
            total_patents = data.get("total_patents_generated", 0)
        except:
            total_patents = 0
            
        st.metric("Analyses This Hour", f"{current_usage}/3")
        st.metric("Total Patents Generated", total_patents)
        
        if not is_allowed:
            st.error("❌ Limit Exceeded")
        elif current_usage >= 2:
            st.warning("⚠️ Approaching Limit")
        else:
            st.success("✅ Within Limit")
        
        # W&Patent promotion
        render_wpatent_promotion()
        
        # Example websites
        st.markdown("---")
        st.markdown("### 🎯 Example Websites to Analyze")
        st.markdown("""
        Try these for testing:
        - `https://www.tesla.com`
        - `https://www.openai.com`
        - `https://www.spacex.com`
        - `https://www.deepmind.com`
        - `https://www.notion.so`
        """)
    
    # Main website analyzer interface
    st.markdown("### 🔍 Analyze Website for Patent Opportunities")
    
    with st.form("website_analyzer"):
        website_url = st.text_input(
            "Website URL *",
            placeholder="https://example.com or www.competitor-website.com",
            help="Enter the full URL of the website you want to analyze for patent opportunities"
        )
        
        analysis_focus = st.selectbox(
            "Analysis Focus",
            [
                "General Technology Analysis",
                "Competitive Intelligence",
                "Product Innovation",
                "Technical Architecture",
                "Business Model Innovation"
            ],
            help="What aspect of the website would you like to focus on for patent generation?"
        )
        
        generate_button = st.form_submit_button(
            "🔬 Analyze Website & Generate Patent", 
            type="primary",
            disabled=not is_allowed
        )
    
    if generate_button:
        if not website_url:
            st.error("Please enter a website URL")
        elif not validate_url(website_url):
            st.error("Please enter a valid URL (e.g., https://example.com)")
        else:
            # Check rate limit again before processing
            is_allowed, current_usage = check_rate_limit()
            if not render_rate_limit_message(current_usage, is_allowed):
                return
            
            # Step 1: Scrape website content
            with st.spinner("🌐 Scraping website content..."):
                website_content = scrape_website_content(website_url)
                
                if website_content.startswith("Error"):
                    st.error(f"Failed to scrape website: {website_content}")
                    return
                
                # Show scraped content in expander
                with st.expander("📄 View Scraped Website Content"):
                    st.text_area("Extracted Content", website_content, height=200)
            
            # Step 2: Analyze website with Gemini
            with st.spinner("🤖 Analyzing website content and identifying opportunities..."):
                website_analysis = analyze_website_with_gemini(website_url, website_content)
                
                with st.expander("📊 View Website Analysis"):
                    st.markdown(website_analysis)
            
            # Step 3: Generate patent concept
            with st.spinner("💡 Generating patent concept based on analysis..."):
                patent_idea = generate_patent_from_analysis(website_analysis, website_url)
            
            # Step 4: Analyze patent strength
            with st.spinner("📈 Analyzing patent strength and commercial potential..."):
                strength_analysis = analyze_patent_strength(patent_idea)
            
            # Step 5: Generate patent claims
            with st.spinner("⚖️ Drafting formal patent claims..."):
                patent_claims = generate_patent_claims(patent_idea)
            
            # Extract patent title for tracking
            patent_title = f"Patent from {website_url}"
            if "**Patent Title:**" in patent_idea:
                patent_title = patent_idea.split("**Patent Title:**")[1].split("\n")[0].strip()
            elif "Title:" in patent_idea:
                patent_title = patent_idea.split("Title:")[1].split("\n")[0].strip()
            
            # Increment usage counter after successful generation
            new_usage_count, total_patents = increment_usage(patent_title)
            
            # Update sidebar metrics
            st.sidebar.metric("Analyses This Hour", f"{new_usage_count}/3")
            st.sidebar.metric("Total Patents Generated", total_patents)
            
            # Display results
            st.success("🎉 Patent concept generated successfully from website analysis!")
            st.info("💡 **Ready to protect these innovations?** Contact wp@wpatent.com for full patent services!")
            
            # Results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["💡 Patent Concept", "📊 Strength Analysis", "⚖️ Patent Claims", "🌐 Website Insights"])
            
            with tab1:
                st.subheader("Generated Patent Concept")
                st.markdown(patent_idea)
                
                # Download patent concept
                st.download_button(
                    label="📥 Download Patent Concept",
                    data=patent_idea,
                    file_name=f"wpatent_from_{website_url.replace('https://', '').replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with tab2:
                st.subheader("Patent Strength Analysis")
                st.markdown(strength_analysis)
            
            with tab3:
                st.subheader("Formal Patent Claims")
                st.markdown(patent_claims)
            
            with tab4:
                st.subheader("Original Website Analysis")
                st.markdown(website_analysis)
            
            # Competitive intelligence insights
            st.markdown("---")
            st.markdown("### 🎯 Competitive Intelligence Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🔍 What We Discovered:**
                - Technology trends from the website
                - Potential innovation gaps
                - Competitive positioning
                - Market opportunities
                """)
            
            with col2:
                st.markdown("""
                **🚀 Next Steps:**
                - Deep dive competitive analysis
                - Patent landscape mapping
                - Innovation strategy development
                - Portfolio optimization
                """)
    
    # Example analysis section
    st.markdown("---")
    st.markdown("### 💡 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🌐 Website Input**
        1. Enter any website URL
        2. AI scrapes and analyzes content
        3. Identifies key technologies
        4. Extracts business context
        """)
    
    with col2:
        st.markdown("""
        **🤖 AI Analysis**
        1. Technology domain mapping
        2. Problem identification
        3. Innovation opportunity spotting
        4. Competitive gap analysis
        """)
    
    with col3:
        st.markdown("""
        **⚖️ Patent Output**
        1. Novel patent concepts
        2. Strength assessment
        3. Formal claims drafting
        4. Commercial potential
        """)
    
    # Always show the inquiry form
    render_inquiry_form()
    
    # Footer with W&Patent branding
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <strong>W&Patent AI Website Analyzer</strong> • The World's Smarter Patent Marketplace • 
        <a href="mailto:wp@wpatent.com" style='color: #007bff;'>wp@wpatent.com</a> • 
        <a href="https://www.wpatent.com" style='color: #007bff;'>www.wpatent.com</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()