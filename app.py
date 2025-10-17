from transformers import pipeline
import groq
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
from ddgs import DDGS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables from Streamlit secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("❌ GROQ_API_KEY not found in Streamlit secrets. Please configure it in the Secrets tab.")
    st.stop()

# Configure Groq client
client = groq.Client(api_key=GROQ_API_KEY)

# Groq model configuration
GROQ_MODEL = "qwen/qwen3-32b"  # Options: "llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"

def groq_generate_content(prompt, model=GROQ_MODEL):
    """Generate content using Groq API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API error: {str(e)}")
        return f"Error generating content: {str(e)}"

def groq_embed_content(text):
    """Generate embeddings using Groq - Note: Groq doesn't have native embedding API yet"""
    # For now, we'll use TF-IDF as a fallback for semantic similarity
    # In production, you might want to use a separate embedding service
    vectorizer = TfidfVectorizer(max_features=1000)
    try:
        # Create a simple embedding using TF-IDF
        tfidf_matrix = vectorizer.fit_transform([text])
        return tfidf_matrix.toarray()[0]
    except:
        return np.zeros(1000)  # Return zero vector as fallback

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

# Robust Tiered Industry Detection (Updated for Groq)
def keyword_industry_detection(patent_claims, website_analysis):
    """Tier 1: Fast keyword-based industry detection"""
    combined_text = f"{website_analysis} {patent_claims}".lower()
    
    # Enhanced industry taxonomy with weights
    industry_taxonomy = {
        'fintech': ['financial', 'banking', 'payment', 'fintech', 'investment', 'lending', 'insurance', 'wealth', 'portfolio', 'transaction', 'digital wallet', 'cryptocurrency'],
        'healthtech': ['healthcare', 'medical', 'patient', 'clinical', 'telemedicine', 'biotech', 'pharmaceutical', 'diagnostic', 'health tech', 'medical device', 'hospital', 'treatment'],
        'edtech': ['education', 'learning', 'course', 'student', 'educational', 'online learning', 'edtech', 'curriculum', 'teaching', 'academic', 'school', 'university'],
        'saas': ['software', 'service', 'cloud', 'subscription', 'enterprise', 'platform', 'api', 'integration', 'dashboard', 'workflow', 'automation', 'business intelligence'],
        'iot': ['internet of things', 'iot', 'connected', 'sensor', 'smart device', 'embedded', 'wireless', 'smart home', 'industrial iot', 'sensor network'],
        'ai_ml': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'ai model', 'algorithm', 'predictive', 'natural language', 'computer vision', 'llm'],
        'ecommerce': ['ecommerce', 'e-commerce', 'online store', 'shopping cart', 'marketplace', 'retail', 'inventory', 'checkout', 'product catalog', 'digital storefront'],
        'cleantech': ['renewable', 'solar', 'wind', 'energy', 'sustainability', 'green tech', 'carbon', 'environmental', 'clean energy', 'climate', 'emissions']
    }
    
    industry_scores = {}
    for industry, keywords in industry_taxonomy.items():
        score = 0
        for keyword in keywords:
            if keyword in combined_text:
                # Weight longer, more specific keywords higher
                score += len(keyword.split()) * 2
        if score > 0:
            industry_scores[industry] = score
    
    return dict(sorted(industry_scores.items(), key=lambda x: x[1], reverse=True)[:5])

def llm_industry_classifier(patent_claims, website_content):
    """Tier 2: LLM-based industry classification for precision using Groq"""
    prompt = f"""
    Analyze this patent context and website content to determine the primary industry.
    
    PATENT CONTEXT:
    {patent_claims[:1000]}  # Limit length
    
    WEBSITE CONTENT: 
    {website_content[:1000]}  # Limit length
    
    Classify into ONE primary industry from this list:
    - fintech (financial technology, banking, payments)
    - healthtech (healthcare, medical technology)
    - edtech (education technology)
    - saas (software as a service, enterprise software)
    - iot (internet of things, connected devices)
    - ai_ml (artificial intelligence, machine learning)
    - ecommerce (online retail, marketplaces)
    - cleantech (renewable energy, sustainability)
    - other (if none of the above fit well)
    
    Respond in this exact JSON format:
    {{
        "primary_industry": "industry_name",
        "confidence": "High/Medium/Low",
        "reasoning": "brief explanation",
        "alternative_industries": ["industry1", "industry2"]
    }}
    """
    
    try:
        response = groq_generate_content(prompt)
        
        # Parse JSON response
        import json
        # Extract JSON from response if there's additional text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response)
        return result
        
    except Exception as e:
        # Fallback to keyword analysis if Groq fails
        return {
            "primary_industry": "other",
            "confidence": "Low", 
            "reasoning": f"LLM analysis failed: {str(e)}",
            "alternative_industries": []
        }

def semantic_similarity_analysis(patent_claims, website_analysis):
    """Tier 3: Semantic similarity analysis using TF-IDF"""
    try:
        combined_text = f"{patent_claims} {website_analysis}"[:2000]  # Limit length
        
        # Industry domain descriptions
        industry_domains = {
            "fintech": "financial technology digital payments banking investment cryptocurrency blockchain fintech insurance wealth management",
            "healthtech": "healthcare medical technology patient care telemedicine biotech pharmaceuticals medical devices diagnostics treatment clinical",
            "edtech": "education learning educational technology online courses digital learning students teachers academic curriculum school university",
            "saas": "software as a service cloud computing enterprise business platform subscription api integration workflow automation",
            "iot": "internet of things connected devices sensors smart home industrial iot wireless embedded systems sensor networks",
            "ai_ml": "artificial intelligence machine learning neural networks deep learning algorithms predictive analytics natural language processing computer vision",
            "ecommerce": "ecommerce online shopping retail marketplace digital storefront inventory management checkout payment gateway",
            "cleantech": "renewable energy sustainability green technology environmental solar wind power carbon emissions clean energy climate"
        }
        
        # Calculate similarities using TF-IDF
        similarities = {}
        vectorizer = TfidfVectorizer()
        
        # Create document corpus
        documents = [combined_text] + list(industry_domains.values())
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarities between patent text and each industry
        patent_vector = tfidf_matrix[0]
        for i, (industry, description) in enumerate(industry_domains.items(), 1):
            industry_vector = tfidf_matrix[i]
            similarity = cosine_similarity(patent_vector, industry_vector)[0][0]
            similarities[industry] = similarity
        
        return dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5])
        
    except Exception as e:
        st.warning(f"Semantic analysis failed: {str(e)}")
        return {}

def consolidate_industries(keyword_results, llm_results, semantic_results):
    """Consolidate results from all three tiers with weighted scoring"""
    
    consolidated = {}
    
    # Weight the different methods
    weights = {
        'llm': 0.5,      # Most reliable
        'semantic': 0.3,  # Good for nuance
        'keyword': 0.2    # Baseline
    }
    
    # Process LLM results (highest weight)
    if llm_results.get('primary_industry') and llm_results.get('primary_industry') != 'other':
        industry = llm_results['primary_industry']
        confidence_weight = {'High': 1.0, 'Medium': 0.7, 'Low': 0.4}.get(llm_results.get('confidence', 'Medium'), 0.5)
        consolidated[industry] = weights['llm'] * confidence_weight
    
    # Add LLM alternative industries with lower weight
    for alt_industry in llm_results.get('alternative_industries', [])[:2]:
        if alt_industry in consolidated:
            consolidated[alt_industry] += weights['llm'] * 0.3
        else:
            consolidated[alt_industry] = weights['llm'] * 0.3
    
    # Process semantic results
    for industry, score in semantic_results.items():
        if industry in consolidated:
            consolidated[industry] += weights['semantic'] * score
        else:
            consolidated[industry] = weights['semantic'] * score
    
    # Process keyword results
    max_keyword_score = max(keyword_results.values()) if keyword_results else 1
    for industry, score in keyword_results.items():
        normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
        if industry in consolidated:
            consolidated[industry] += weights['keyword'] * normalized_score
        else:
            consolidated[industry] = weights['keyword'] * normalized_score
    
    # Filter and return top industries
    final_industries = dict(sorted(consolidated.items(), key=lambda x: x[1], reverse=True)[:3])
    
    # Convert to percentage confidence
    total = sum(final_industries.values()) if final_industries else 1
    return {industry: round(score/total * 100, 1) for industry, score in final_industries.items()}

def robust_industry_detection(patent_claims, website_analysis, original_url):
    """Main tiered industry detection function"""
    
    st.info("🔍 Analyzing industry context...")
    
    # Tier 1: Fast keyword analysis
    with st.spinner("Tier 1: Keyword analysis..."):
        keyword_industries = keyword_industry_detection(patent_claims, website_analysis)
    
    # Tier 2: LLM classification
    with st.spinner("Tier 2: AI classification..."):
        llm_industries = llm_industry_classifier(patent_claims, website_analysis)
    
    # Tier 3: Semantic similarity
    with st.spinner("Tier 3: Semantic analysis..."):
        semantic_industries = semantic_similarity_analysis(patent_claims, website_analysis)
    
    # Consolidate results
    final_industries = consolidate_industries(keyword_industries, llm_industries, semantic_industries)
    
    # Display analysis results in expander
    with st.expander("📊 Industry Analysis Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔤 Keyword Analysis**")
            for industry, score in keyword_industries.items():
                st.write(f"- {industry}: {score}")
        
        with col2:
            st.markdown("**🤖 LLM Classification**")
            st.write(f"Primary: {llm_industries.get('primary_industry', 'N/A')}")
            st.write(f"Confidence: {llm_industries.get('confidence', 'N/A')}")
            st.write(f"Alternatives: {', '.join(llm_industries.get('alternative_industries', []))}")
        
        with col3:
            st.markdown("**🎯 Semantic Similarity**")
            for industry, score in semantic_industries.items():
                st.write(f"- {industry}: {score:.3f}")
    
    return final_industries

# Infringement Analysis Functions
def find_industry_competitors(industry_keywords, max_competitors=8):
    """Find competitor websites in the same industry"""
    try:
        with DDGS() as ddgs:
            competitors = []
            query = f"top companies in {industry_keywords} industry website"
            search_results = list(ddgs.text(query, max_results=max_competitors))
            
            for result in search_results:
                url = result.get("href", "")
                if url and any(domain in url for domain in ['.com', '.org', '.io', '.net']):
                    competitors.append({
                        "name": result.get("title", "Unknown Company"),
                        "url": url,
                        "description": result.get("body", "")[:200] + "..."
                    })
            
            return competitors
    except Exception as e:
        st.error(f"Error finding competitors: {str(e)}")
        return []

def infringement_risk_analysis(patent_claims, competitor_url):
    """Analyze a specific competitor website for infringement risks using Groq"""
    try:
        # Scrape the competitor site
        site_content = scrape_website_content(competitor_url)
        
        if site_content.startswith("Error"):
            return {
                "competitor_url": competitor_url,
                "risk_level": "Unknown",
                "error": site_content,
                "overlapping_features": [],
                "recommendations": []
            }
        
        # Analyze for infringement risks using Groq
        analysis_prompt = f"""
        PATENT CLAIMS TO ANALYZE:
        {patent_claims}
        
        COMPETITOR WEBSITE CONTENT:
        {site_content}
        
        Analyze for potential patent infringement risks. Focus on:
        
        1. **DIRECT PRODUCT MATCH**: Does the website show products/services that directly implement the claimed invention?
        2. **FEATURE OVERLAP**: Which specific claim elements appear to be implemented in their products?
        3. **TECHNICAL SIMILARITY**: How similar are the technical approaches and implementations?
        4. **COMMERCIAL USE EVIDENCE**: Is there evidence of actual commercial use of similar technology?
        
        Provide your analysis in this exact JSON format:
        {{
            "risk_level": "Low/Medium/High",
            "overlapping_features": ["list of specific overlapping features"],
            "infringement_confidence": "Low/Medium/High",
            "key_findings": "Detailed analysis of potential infringement",
            "recommendations": ["list of recommendations for further action"]
        }}
        
        Be thorough and evidence-based in your assessment.
        """
        
        response = groq_generate_content(analysis_prompt)
        
        # Parse the JSON response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
            else:
                analysis_result = json.loads(response)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response from text
            analysis_result = {
                "risk_level": "Medium",
                "overlapping_features": ["Unable to parse detailed features"],
                "infringement_confidence": "Medium", 
                "key_findings": response[:500] + "...",
                "recommendations": ["Conduct manual review of this competitor"]
            }
        
        # Add competitor info to result
        analysis_result["competitor_url"] = competitor_url
        analysis_result["scraped_content_preview"] = site_content[:300] + "..."
        
        return analysis_result
        
    except Exception as e:
        return {
            "competitor_url": competitor_url,
            "risk_level": "Error",
            "error": str(e),
            "overlapping_features": [],
            "recommendations": ["Analysis failed - manual review required"]
        }

def industry_wide_infringement_scan(patent_claims, website_analysis, original_url, max_competitors=5):
    """Search across multiple industry players for infringement risks"""
    
    # Use robust industry detection
    detected_industries = robust_industry_detection(patent_claims, website_analysis, original_url)
    
    if not detected_industries:
        st.error("❌ Could not detect relevant industries for scanning")
        return {
            "error": "Industry detection failed",
            "scanned_competitors": 0,
            "high_risk_findings": 0,
            "detailed_results": []
        }
    
    # Show detected industries
    st.success(f"🎯 Detected Industries: {', '.join(detected_industries.keys())}")
    
    # Use the highest confidence industry for competitor search
    primary_industry = next(iter(detected_industries))
    industry_confidence = detected_industries[primary_industry]
    
    st.info(f"🔍 Scanning {primary_industry} industry (confidence: {industry_confidence}%)...")
    
    # Find competitors in this industry
    competitors = find_industry_competitors(primary_industry, max_competitors)
    
    if not competitors:
        return {
            "error": f"No competitors found in {primary_industry} industry",
            "scanned_competitors": 0,
            "high_risk_findings": 0,
            "detailed_results": []
        }
    
    infringement_findings = []
    high_risk_count = 0
    
    # Analyze each competitor
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, competitor in enumerate(competitors):
        status_text.text(f"Analyzing {competitor['name']}... ({i+1}/{len(competitors)})")
        
        risk_analysis = infringement_risk_analysis(patent_claims, competitor["url"])
        infringement_findings.append({
            "competitor_info": competitor,
            "risk_analysis": risk_analysis
        })
        
        if risk_analysis.get("risk_level") == "High":
            high_risk_count += 1
        
        progress_bar.progress((i + 1) / len(competitors))
    
    status_text.text("Analysis complete!")
    
    return {
        "scanned_competitors": len(competitors),
        "high_risk_findings": high_risk_count,
        "detected_industries": detected_industries,
        "primary_industry": primary_industry,
        "detailed_results": infringement_findings,
        "overall_risk_level": "High" if high_risk_count > 0 else "Medium" if len(competitors) > 0 else "Low"
    }

def display_infringement_results(infringement_results, patent_claims):
    """Display infringement analysis results in Streamlit"""
    st.subheader("⚖️ Infringement Risk Analysis")
    
    if infringement_results.get('error'):
        st.error(f"Infringement analysis failed: {infringement_results['error']}")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Competitors Scanned", infringement_results['scanned_competitors'])
    with col2:
        st.metric("High Risk Findings", infringement_results['high_risk_findings'])
    with col3:
        st.metric("Overall Risk", infringement_results['overall_risk_level'])
    
    # Detailed results
    st.markdown("### 📊 Detailed Competitor Analysis")
    
    for i, result in enumerate(infringement_results['detailed_results']):
        competitor = result['competitor_info']
        analysis = result['risk_analysis']
        
        # Color code based on risk level
        risk_color = {
            "High": "red",
            "Medium": "orange", 
            "Low": "green",
            "Error": "gray"
        }.get(analysis.get('risk_level', 'Unknown'), 'gray')
        
        with st.expander(f"🎯 {competitor['name']} - Risk: :{risk_color}[{analysis.get('risk_level', 'Unknown')}]", expanded=i < 2):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Website:** {competitor['url']}")
                st.markdown(f"**Description:** {competitor.get('description', 'N/A')}")
                
                if analysis.get('key_findings'):
                    st.markdown("**Key Findings:**")
                    st.info(analysis['key_findings'])
                
                if analysis.get('overlapping_features'):
                    st.markdown("**Overlapping Features:**")
                    for feature in analysis['overlapping_features']:
                        st.write(f"• {feature}")
            
            with col2:
                st.markdown(f"**Confidence:** {analysis.get('infringement_confidence', 'N/A')}")
                
                if analysis.get('recommendations'):
                    st.markdown("**Recommendations:**")
                    for rec in analysis['recommendations'][:2]:  # Show top 2
                        st.write(f"📌 {rec}")
            
            # Visit competitor button
            st.markdown(f"[🌐 Visit Competitor Website]({competitor['url']})")
    
    # Download report
    infringement_report = {
        "patent_claims": patent_claims,
        "analysis_summary": {
            "scanned_competitors": infringement_results['scanned_competitors'],
            "high_risk_findings": infringement_results['high_risk_findings'],
            "overall_risk_level": infringement_results['overall_risk_level'],
            "industry_keywords": infringement_results.get('detected_industries', {})
        },
        "detailed_results": infringement_results['detailed_results'],
        "generated_at": datetime.now().isoformat()
    }
    
    st.download_button(
        label="📥 Download Infringement Analysis Report",
        data=json.dumps(infringement_report, indent=2),
        file_name=f"infringement_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def search_prior_art(patent_claims, max_results=5):
    """Search for prior art using DuckDuckGo based on patent claims"""
    try:
        # Extract key terms from patent claims for search
        search_terms = extract_search_terms_from_claims(patent_claims)
        
        with DDGS() as ddgs:
            results = []
            for term in search_terms[:3]:  # Use top 3 search terms
                query = f"patent {term} technology prior art"
                search_results = list(ddgs.text(query, max_results=max_results))

                for r in search_results:
                    url = r.get("href")
                    title = r.get("title")
                    content = r.get("body")

                    if not all([url, title, content]):
                        continue

                    # Add result to list WITH SOURCE INFORMATION
                    result = {
                        "title": title,
                        "url": url,
                        "content": content,
                        "search_term": term,
                        "source": "DuckDuckGo Search",
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)

            return {"results": results, "search_terms": search_terms}
    except Exception as e:
        return {"error": f"Prior art search failed: {str(e)}", "results": []}

def extract_search_terms_from_claims(patent_claims):
    """Extract key search terms from patent claims for prior art research"""
    # Simple extraction of key terms - you could enhance this with NLP
    claims_text = patent_claims.lower()
    
    # Remove common patent claim language
    stop_words = ['claim', 'comprising', 'wherein', 'method', 'system', 'device', 'apparatus']
    for word in stop_words:
        claims_text = claims_text.replace(word, '')
    
    # Extract potential technical terms (words with capital letters or technical sounding)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', claims_text)
    
    # Filter for potentially technical terms (longer words often more specific)
    technical_terms = [word for word in words if len(word) > 5][:10]
    
    return list(set(technical_terms))  # Remove duplicates

def analyze_prior_art_with_groq(patent_claims, prior_art_results):
    """Use Groq to analyze prior art search results WITH SOURCE CITATIONS"""
    if not prior_art_results.get('results'):
        return "No prior art results to analyze."
    
    # Format results with clear source citations
    search_results_text = "\n\n".join([
        f"[SOURCE {i+1}]\n"
        f"Title: {result['title']}\n"
        f"URL: {result['url']}\n"
        f"Relevance: {result['content'][:300]}...\n"
        f"Search Term: {result['search_term']}\n"
        for i, result in enumerate(prior_art_results['results'])
    ])
    
    prompt = f"""
    Analyze these prior art search results in relation to the patent claims below.
    For each assessment point, REFERENCE THE SPECIFIC SOURCE(S) that support your analysis.
    
    PATENT CLAIMS:
    {patent_claims}
    
    PRIOR ART SEARCH RESULTS:
    {search_results_text}
    
    Please provide a thorough analysis with SOURCE REFERENCES:
    
    1. **NOVELTY ASSESSMENT** 
       - Overall novelty score (1-10)
       - Specific elements that appear novel vs. existing art
       - Reference specific sources that support this assessment
    
    2. **SIMILAR TECHNOLOGIES FOUND**
       - List similar technologies/patents found, citing sources
       - Describe how they relate to the claims
    
    3. **POTENTIAL CONFLICTS & OVERLAPS**
       - Identify potential infringement risks with source evidence
       - Highlight overlapping technical approaches
    
    4. **RECOMMENDATIONS FOR CLAIM MODIFICATION**
       - Suggest specific claim language changes to improve novelty
       - Reference the prior art that necessitates these changes
    
    5. **SEARCH QUALITY ASSESSMENT**
       - Evaluate comprehensiveness of search
       - Suggest additional search terms
    
    Always cite sources using format: [SOURCE X] when referencing specific findings.
    """
    
    try:
        return groq_generate_content(prompt)
    except Exception as e:
        return f"Error analyzing prior art: {str(e)}"

def display_prior_art_results(prior_art_results, prior_art_analysis, patent_claims):
    """Display prior art results with proper source attribution"""
    st.subheader("🔍 Prior Art Research Analysis")
    
    if prior_art_results.get('error'):
        st.error(f"Prior art search failed: {prior_art_results['error']}")
        return
    
    # Show search terms used
    if prior_art_results.get('search_terms'):
        st.markdown(f"**🔎 Search Terms Used:** `{', '.join(prior_art_results['search_terms'])}`")
    
    # 1. Show the AI analysis with citations
    st.markdown("### 📊 Prior Art Assessment")
    st.markdown(prior_art_analysis)
    
    # 2. Show detailed source references
    st.markdown("### 📚 Source References")
    
    for i, result in enumerate(prior_art_results.get('results', [])[:5]):
        with st.expander(f"📄 Source {i+1}: {result['title']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Title:** {result['title']}")
                st.markdown(f"**URL:** {result['url']}")
                st.markdown(f"**Search Term:** `{result.get('search_term', 'N/A')}`")
                st.markdown(f"**Content Preview:** {result['content'][:500]}...")
            
            with col2:
                # Add a button to visit the source
                st.markdown(f"[🌐 Visit Source]({result['url']})")
                st.markdown(f"**Reference:** [SOURCE {i+1}]")
    
    # 3. Add download capability for sources
    sources_data = {
        "patent_claims": patent_claims,
        "search_terms": prior_art_results.get('search_terms', []),
        "sources": prior_art_results.get('results', []),
        "analysis": prior_art_analysis,
        "generated_at": datetime.now().isoformat()
    }
    
    st.download_button(
        label="📥 Download Prior Art Report",
        data=json.dumps(sources_data, indent=2),
        file_name=f"prior_art_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

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

def analyze_website_with_groq(url, content):
    """Use Groq to analyze the website and extract key information for patent generation"""
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
    
    return groq_generate_content(prompt)

def generate_patent_from_analysis(website_analysis, url):
    """Generate a patent concept based on website analysis using Groq"""
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
    
    return groq_generate_content(prompt)

def analyze_patent_strength(patent_description):
    """Analyze the strength and novelty of the generated patent using Groq"""
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
    
    return groq_generate_content(prompt)

def generate_patent_claims(patent_description):
    """Generate formal patent claims using Groq"""
    prompt = f"""
    Based on this patent description, draft 3-5 formal patent claims:
    
    {patent_description}
    
    Format each claim as:
    "1. [Independent claim describing the core invention]"
    "2. [Dependent claim narrowing the first claim]"
    etc.
    
    Use proper patent claim language and structure.
    """
    
    return groq_generate_content(prompt)

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
    - **NEW: Infringement Risk Analysis**
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
        - **Infringement risk analysis**
        
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
                    "Infringement Risk Analysis",
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
                    "Infringement Assessment",
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
        st.caption("Transform Any Website into Patent Opportunities • AI-Powered Competitive Intelligence • Prior Art Research • Infringement Risk Analysis")
    with col2:
        st.markdown("""
        <div style='
            text-align: right; 
            padding: 10px; 
            background: rgba(128, 128, 128, 0.1); 
            border-radius: 5px; 
            border: 1px solid rgba(128, 128, 128, 0.2);
        '>
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
        
        # Groq model info
        st.markdown("---")
        st.markdown("### 🤖 AI Model")
        st.markdown(f"**Using:** {GROQ_MODEL}")
        st.markdown("**Provider:** Groq")
        st.markdown("**Speed:** ⚡ Ultra-fast")
        
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
        
        include_prior_art = st.checkbox(
            "🔍 Include Prior Art Research", 
            value=True,
            help="Search for existing similar patents/technologies using DuckDuckGo"
        )
        
        include_infringement_analysis = st.checkbox(
            "⚖️ Include Infringement Risk Analysis", 
            value=True,
            help="NEW: Scan competitor websites for potential infringement risks"
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
            
            # Step 2: Analyze website with Groq
            with st.spinner("🤖 Analyzing website content and identifying opportunities..."):
                website_analysis = analyze_website_with_groq(website_url, website_content)
                
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
            
            # Step 6: Prior Art Research (if enabled)
            prior_art_results = None
            prior_art_analysis = None
            
            if include_prior_art:
                with st.spinner("🔍 Searching for prior art and similar technologies..."):
                    prior_art_results = search_prior_art(patent_claims, max_results=5)
                    
                    if not prior_art_results.get('error'):
                        prior_art_analysis = analyze_prior_art_with_groq(patent_claims, prior_art_results)
            
            # NEW: Step 7 - Infringement Risk Analysis (if enabled)
            infringement_results = None
            
            if include_infringement_analysis:
                with st.spinner("⚖️ Analyzing infringement risks across industry competitors..."):
                    infringement_results = industry_wide_infringement_scan(
                        patent_claims, 
                        website_analysis,
                        website_url,
                        max_competitors=5
                    )
            
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
            
            # Results in tabs - UPDATED TO INCLUDE INFRINGEMENT ANALYSIS TAB
            tab_names = ["💡 Patent Concept", "📊 Strength Analysis", "⚖️ Patent Claims", "🌐 Website Insights"]
            if include_prior_art and prior_art_results:
                tab_names.append("🔍 Prior Art Research")
            if include_infringement_analysis and infringement_results:
                tab_names.append("⚖️ Infringement Risks")
            
            tabs = st.tabs(tab_names)
            
            with tabs[0]:
                st.subheader("Generated Patent Concept")
                st.markdown(patent_idea)
                
                # Download patent concept
                st.download_button(
                    label="📥 Download Patent Concept",
                    data=patent_idea,
                    file_name=f"wpatent_from_{website_url.replace('https://', '').replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with tabs[1]:
                st.subheader("Patent Strength Analysis")
                st.markdown(strength_analysis)
            
            with tabs[2]:
                st.subheader("Formal Patent Claims")
                st.markdown(patent_claims)
            
            with tabs[3]:
                st.subheader("Original Website Analysis")
                st.markdown(website_analysis)
            
            # Prior Art Research Tab
            if include_prior_art and prior_art_results and len(tabs) > 4:
                tab_index = 4
                if include_infringement_analysis and infringement_results:
                    tab_index = 4  # Prior art comes before infringement
                    infringement_tab_index = 5
                else:
                    infringement_tab_index = None
                
                with tabs[tab_index]:
                    display_prior_art_results(prior_art_results, prior_art_analysis, patent_claims)
            
            # Infringement Analysis Tab
            if include_infringement_analysis and infringement_results and infringement_tab_index and len(tabs) > infringement_tab_index:
                with tabs[infringement_tab_index]:
                    display_infringement_results(infringement_results, patent_claims)

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
                - Infringement risk assessment
                """)
            
            with col2:
                st.markdown("""
                **🚀 Next Steps:**
                - Deep dive competitive analysis
                - Patent landscape mapping
                - Innovation strategy development
                - Portfolio optimization
                - Freedom-to-operate analysis
                """)
    
    # Example analysis section
    st.markdown("---")
    st.markdown("### 💡 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
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
        
    with col4:
        st.markdown("""
        **🛡️ Risk Analysis**
        1. Competitor scanning
        2. Infringement risk assessment
        3. Industry landscape
        4. Risk mitigation
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