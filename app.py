# app.py
import streamlit as st
import google.generativeai as genai
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import requests

# Load environment variables from Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets. Please configure it in the Secrets tab.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

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

# Patent generation functions
def generate_patent_idea(domain, problem_statement):
    """Generate a patent idea using Gemini"""
    prompt = f"""
    You are a patent expert and innovation strategist. Create a compelling patent idea based on the following:
    
    DOMAIN: {domain}
    PROBLEM STATEMENT: {problem_statement}
    
    Please generate a comprehensive patent concept including:
    1. A novel patent title
    2. Detailed technical description
    3. Key innovative features
    4. Potential applications
    5. Technical implementation approach
    
    Make it realistic, technically sound, and commercially viable.
    Focus on genuine innovation and avoid generic solutions.
    """
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
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
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
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
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
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
    - Patent Idea Generation
    - Strength Analysis
    - Prior Art Research
    - Patent Portfolio Management
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
        - **Unlimited patent generation**
        - **Advanced patent analysis**
        - **Prior art research**
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
                    "Patent Generation Service",
                    "Strength Analysis",
                    "Portfolio Management",
                    "Commercialization Support",
                    "Enterprise Solution",
                    "Other"
                ]
            )
            
            project_type = st.selectbox(
                "Project Type",
                [
                    "Select a project type",
                    "Software/App",
                    "Hardware/Device",
                    "Biotech/Medical",
                    "AI/Machine Learning",
                    "Green Tech",
                    "Consumer Product",
                    "Other"
                ]
            )
        
        message = st.text_area(
            "Message *", 
            placeholder="Tell us about your patent needs, specific requirements, or any questions you have...",
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
                    "source": "WPatent AI Generator"
                }
                
                if submit_inquiry_form(form_data):
                    st.success("✅ Thank you! Your inquiry has been submitted. We'll contact you within 24 hours.")
                else:
                    st.error("❌ There was an error submitting your form. Please email us directly at wp@wpatent.com")

def main():
    st.set_page_config(
        page_title="W&Patent AI Patent Generator", 
        page_icon="⚖️",
        layout="wide"
    )
    
    # Initialize usage tracking
    init_usage_tracking()
    
    # Main header with W&Patent branding
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("⚖️ W&Patent AI Patent Generator")
        st.caption("Transform Your Ideas into Patent-Ready Concepts • Powered by AI & Expert Analysis")
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
            
        st.metric("Generations This Hour", f"{current_usage}/3")
        st.metric("Total Patents Generated", total_patents)
        
        if not is_allowed:
            st.error("❌ Limit Exceeded")
        elif current_usage >= 2:
            st.warning("⚠️ Approaching Limit")
        else:
            st.success("✅ Within Limit")
        
        # W&Patent promotion
        render_wpatent_promotion()
    
    # Main patent generator interface
    st.markdown("### 🎯 Generate Your Patent Concept")
    
    with st.form("patent_generator"):
        col1, col2 = st.columns(2)
        
        with col1:
            domain = st.selectbox(
                "Technology Domain *",
                [
                    "Select a domain",
                    "Artificial Intelligence & Machine Learning",
                    "Blockchain & Cryptocurrency",
                    "Internet of Things (IoT)",
                    "Biotechnology & Healthcare",
                    "Renewable Energy & Green Tech",
                    "Robotics & Automation",
                    "Quantum Computing",
                    "Augmented/Virtual Reality",
                    "Cybersecurity",
                    "Fintech & Financial Services",
                    "EdTech & Education",
                    "AgriTech & Agriculture",
                    "Space Technology",
                    "Autonomous Vehicles",
                    "Other"
                ]
            )
            
        with col2:
            complexity = st.select_slider(
                "Innovation Complexity",
                options=["Incremental", "Moderate", "Breakthrough"],
                value="Moderate"
            )
        
        problem_statement = st.text_area(
            "Problem Statement *",
            placeholder="Describe the specific problem you want to solve, current limitations, and why existing solutions are inadequate...",
            height=100
        )
        
        additional_context = st.text_area(
            "Additional Context (Optional)",
            placeholder="Any specific technical requirements, target market, or special considerations...",
            height=80
        )
        
        generate_button = st.form_submit_button(
            "⚡ Generate Patent Concept", 
            type="primary",
            disabled=not is_allowed
        )
    
    if generate_button:
        if not all([domain != "Select a domain", problem_statement]):
            st.error("Please fill in all required fields (*)")
        else:
            # Check rate limit again before processing
            is_allowed, current_usage = check_rate_limit()
            if not render_rate_limit_message(current_usage, is_allowed):
                return
            
            # Generate patent concept
            with st.spinner("🔬 Analyzing problem space and generating patent concept..."):
                full_context = f"{problem_statement}"
                if additional_context:
                    full_context += f"\n\nAdditional Context: {additional_context}"
                
                patent_idea = generate_patent_idea(domain, full_context)
            
            # Analyze patent strength
            with st.spinner("📊 Analyzing patent strength and commercial potential..."):
                strength_analysis = analyze_patent_strength(patent_idea)
            
            # Generate patent claims
            with st.spinner("⚖️ Drafting formal patent claims..."):
                patent_claims = generate_patent_claims(patent_idea)
            
            # Extract patent title for tracking
            patent_title = "Generated Patent Concept"
            if "**Title:**" in patent_idea:
                patent_title = patent_idea.split("**Title:**")[1].split("\n")[0].strip()
            elif "Patent Title:" in patent_idea:
                patent_title = patent_idea.split("Patent Title:")[1].split("\n")[0].strip()
            
            # Increment usage counter after successful generation
            new_usage_count, total_patents = increment_usage(patent_title)
            
            # Update sidebar metrics
            st.sidebar.metric("Generations This Hour", f"{new_usage_count}/3")
            st.sidebar.metric("Total Patents Generated", total_patents)
            
            # Display results
            st.success("🎉 Patent concept generated successfully!")
            st.info("💡 **Ready to protect your innovation?** Contact wp@wpatent.com for full patent services!")
            
            # Results in tabs
            tab1, tab2, tab3 = st.tabs(["📄 Patent Concept", "📊 Strength Analysis", "⚖️ Patent Claims"])
            
            with tab1:
                st.subheader("Generated Patent Concept")
                st.markdown(patent_idea)
                
                # Download patent concept
                st.download_button(
                    label="📥 Download Patent Concept",
                    data=patent_idea,
                    file_name=f"wpatent_concept_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with tab2:
                st.subheader("Patent Strength Analysis")
                st.markdown(strength_analysis)
            
            with tab3:
                st.subheader("Formal Patent Claims")
                st.markdown(patent_claims)
            
            # Success metrics and next steps
            st.markdown("---")
            st.markdown("### 🚀 Next Steps for Your Patent")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🔍 Prior Art Search**
                - Comprehensive patent database search
                - Identify potential conflicts
                - Validate novelty
                """)
            
            with col2:
                st.markdown("""
                **📝 Patent Drafting**
                - Professional patent attorneys
                - Formal specification
                - Claims optimization
                """)
            
            with col3:
                st.markdown("""
                **🎯 Commercialization**
                - Market analysis
                - Licensing opportunities
                - Investment readiness
                """)
    
    # Always show the inquiry form
    render_inquiry_form()
    
    # Footer with W&Patent branding
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <strong>W&Patent AI Patent Generator</strong> • The World's Smarter Patent Marketplace • 
        <a href="mailto:wp@wpatent.com" style='color: #007bff;'>wp@wpatent.com</a> • 
        <a href="https://www.wpatent.com" style='color: #007bff;'>www.wpatent.com</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()