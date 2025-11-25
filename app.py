"""
Streamlit Frontend for Constitutional Compliance Checker
A beautiful, user-friendly interface for analyzing policy compliance.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple

# Page configuration
st.set_page_config(
    page_title="Constitutional Compliance Checker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-compliant {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .status-non-compliant {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .conflict-card {
        background-color: #fff;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .article-card {
        background-color: #f8f9fa;
        border-left: 3px solid #007bff;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .severity-high {
        color: #dc3545;
        font-weight: bold;
    }
    .severity-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .severity-low {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .confidence-high {
        background-color: #dc3545;
        color: white;
    }
    .confidence-medium {
        background-color: #ffc107;
        color: #000;
    }
    .confidence-low {
        background-color: #6c757d;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

@st.cache_resource
def load_pipeline():
    """Load the compliance pipeline (cached for performance)."""
    try:
        from src.orchestrator import load_pipeline
        import os
        
        # Check if LLM should be enabled
        use_llm = bool(os.getenv("GROQ_API_KEY"))
        
        pipeline = load_pipeline(
            root_dir=".",
            engine="langgraph",
            use_llm=use_llm,
            llm_model="llama-3.3-70b-versatile"
        )
        return pipeline, use_llm
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        return None, False

def format_confidence(confidence: float) -> Tuple[str, str]:
    """Format confidence score with color coding."""
    if confidence >= 0.75:
        return "High", "confidence-high"
    elif confidence >= 0.60:
        return "Medium", "confidence-medium"
    else:
        return "Low", "confidence-low"

def format_severity(severity: str) -> str:
    """Format severity with color."""
    severity_lower = severity.lower()
    if severity_lower == "high":
        return f'<span class="severity-high">🔴 High</span>'
    elif severity_lower == "medium":
        return f'<span class="severity-medium">🟡 Medium</span>'
    else:
        return f'<span class="severity-low">🟢 Low</span>'

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Constitutional Compliance Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze policies for compliance with the Constitution of Pakistan</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Load pipeline
        if st.session_state.pipeline is None:
            with st.spinner("Loading compliance checker..."):
                pipeline, use_llm = load_pipeline()
                st.session_state.pipeline = pipeline
                st.session_state.use_llm = use_llm
        
        if st.session_state.pipeline is None:
            st.error("Failed to load pipeline. Please check your configuration.")
            return
        
        if st.session_state.use_llm:
            st.success("✅ LLM Analysis Enabled")
        else:
            st.warning("⚠️ LLM Analysis Disabled (No GROQ_API_KEY)")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This tool analyzes policy documents for compliance with the Constitution of Pakistan.
        
        **Features:**
        - Detects constitutional violations
        - Identifies relevant articles
        - Provides detailed analysis
        - Suggests remediation steps
        """)
    
    # Main content area
    tab1, tab2 = st.tabs(["📝 Input Policy", "📊 View Results"])
    
    with tab1:
        st.header("Policy Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📄 Upload File", "✍️ Type Text"],
            horizontal=True
        )
        
        policy_text = None
        
        if input_method == "📄 Upload File":
            uploaded_file = st.file_uploader(
                "Upload a policy document (.txt)",
                type=['txt'],
                help="Upload a text file containing the policy to analyze"
            )
            
            if uploaded_file is not None:
                policy_text = uploaded_file.read().decode('utf-8')
                st.text_area("Preview:", policy_text[:500] + "..." if len(policy_text) > 500 else policy_text, height=150, disabled=True)
        
        else:  # Type Text
            policy_text = st.text_area(
                "Enter policy text:",
                height=300,
                placeholder="Paste or type the policy text here...",
                help="Enter the full text of the policy you want to analyze"
            )
        
        # Analyze button
        if st.button("🔍 Analyze Policy", type="primary", use_container_width=True):
            if not policy_text or not policy_text.strip():
                st.error("Please provide policy text or upload a file.")
                return
            
            if st.session_state.pipeline is None:
                st.error("Pipeline not loaded. Please check configuration.")
                return
            
            # Show progress
            with st.spinner("Analyzing policy for constitutional compliance..."):
                try:
                    result = st.session_state.pipeline.evaluate_text(
                        policy_text,
                        policy_id="user_policy"
                    )
                    st.session_state.last_result = result
                    st.success("Analysis complete! Switch to 'View Results' tab to see details.")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)
    
    with tab2:
        st.header("Analysis Results")
        
        if st.session_state.last_result is None:
            st.info("👆 Please analyze a policy first using the 'Input Policy' tab.")
        else:
            result = st.session_state.last_result
            
            # Overall Status
            st.subheader("📋 Overall Status")
            
            if result.diagnosis:
                is_compliant = result.diagnosis.is_constitutional
                confidence = result.diagnosis.confidence
                
                if is_compliant:
                    st.markdown(f"""
                    <div class="status-compliant">
                        <h3>✅ Policy is Constitutional</h3>
                        <p>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-non-compliant">
                        <h3>❌ Policy has Constitutional Violations</h3>
                        <p>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Summary
                if result.diagnosis.summary:
                    with st.expander("📝 Summary", expanded=True):
                        st.write(result.diagnosis.summary)
            else:
                st.warning("No diagnosis available.")
            
            # Conflicts Section
            if result.conflicts:
                st.subheader(f"⚠️ Detected Violations ({len(result.conflicts)})")
                
                for i, conflict in enumerate(result.conflicts, 1):
                    conf_level, conf_class = format_confidence(conflict.confidence)
                    
                    with st.expander(
                        f"Article {conflict.article_id}: {conflict.article_title}",
                        expanded=(i == 1)
                    ):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Severity", format_severity(conflict.severity), delta=None)
                        with col2:
                            st.metric("Confidence", f"{conflict.confidence:.1%}", delta=None)
                        with col3:
                            st.metric("Jurisdiction", conflict.jurisdiction.replace("_", " ").title(), delta=None)
                        
                        st.markdown("**Description:**")
                        st.write(conflict.description)
                        
                        st.markdown("**Evidence:**")
                        st.info(conflict.evidence)
            else:
                st.success("✅ No constitutional violations detected!")
            
            # Retrieved Articles Section
            if result.retrieved_articles:
                st.subheader(f"📚 Relevant Articles ({len(result.retrieved_articles)})")
                
                # Group by category
                articles_by_category = {}
                for article in result.retrieved_articles:
                    category = article.category.replace("_", " ").title()
                    if category not in articles_by_category:
                        articles_by_category[category] = []
                    articles_by_category[category].append(article)
                
                for category, articles in articles_by_category.items():
                    with st.expander(f"{category} ({len(articles)} articles)", expanded=False):
                        for article in articles:
                            st.markdown(f"""
                            <div class="article-card">
                                <strong>Article {article.article_id}:</strong> {article.title}<br>
                                <small>Relevance: {article.relevance:.2%}</small>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("View snippet"):
                                st.text(article.text_snippet[:500] + "..." if len(article.text_snippet) > 500 else article.text_snippet)
            
            # Recommended Actions
            if result.diagnosis and result.diagnosis.recommended_actions:
                st.subheader("💡 Recommended Actions")
                for i, action in enumerate(result.diagnosis.recommended_actions, 1):
                    st.markdown(f"{i}. {action}")
            
            # Detected Domains
            if result.detected_domains:
                st.subheader("🏷️ Detected Policy Domains")
                cols = st.columns(len(result.detected_domains))
                for i, domain in enumerate(result.detected_domains):
                    with cols[i]:
                        st.info(domain.replace("_", " ").title())
            
            # LLM Analysis (if available)
            if result.llm_analysis:
                st.subheader("🤖 LLM Analysis")
                with st.expander("View detailed LLM analysis", expanded=False):
                    st.json(result.llm_analysis)
            
            # Download Results
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            import json
            results_json = json.dumps(result.to_dict(), indent=2, default=str)
            
            st.download_button(
                label="📥 Download Results as JSON",
                data=results_json,
                file_name=f"compliance_analysis_{result.policy_id}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()

