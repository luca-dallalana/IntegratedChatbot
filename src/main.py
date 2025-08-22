# src/main.py
import streamlit as st
import os
import sys

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.streamlit_app.ui import create_dual_chatbot_interface_with_langsmith, show_langsmith_analytics, create_standard_comparison_interface
from src.core.langsmith_analyzer import LANGSMITH_AVAILABLE

def create_streamlit_app():
    """Create enhanced Streamlit application with optional LangSmith integration."""
    
    st.set_page_config(
        page_title="Investment Chatbot Model Comparison with LangSmith",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Investment Chatbot Model Comparison")
    if LANGSMITH_AVAILABLE:
        st.markdown("Advanced AI model comparison with optional LangSmith analytics")
    else:
        st.markdown("Advanced AI model comparison")
    
    # Sidebar configuration
    st.sidebar.header("API Configuration")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    langsmith_api_key = None
    langsmith_project_name = "investment-chatbot-comparison"
    if LANGSMITH_AVAILABLE:
        langsmith_api_key = st.sidebar.text_input("LangSmith API Key (Optional)", type="password")
        if langsmith_api_key:
            # Check for environment variable override
            env_project = os.environ.get("LANGSMITH_PROJECT")
            if env_project:
                st.sidebar.info(f"Using project from environment variable: `{env_project}`")
                langsmith_project_name = st.sidebar.text_input("LangSmith Project Name", value=env_project, disabled=True)
            else:
                langsmith_project_name = st.sidebar.text_input("LangSmith Project Name", value="investment-chatbot-comparison")
            
            st.sidebar.success("LangSmith integration enabled")

            # LangSmith Diagnostics Section
            with st.sidebar.expander("LangSmith Diagnostics"):
                st.write("Checking for LangSmith environment variables...")
                env_vars = {
                    "LANGCHAIN_TRACING_V2": os.environ.get("LANGCHAIN_TRACING_V2"),
                    "LANGCHAIN_API_KEY": os.environ.get("LANGCHAIN_API_KEY"),
                    "LANGCHAIN_ENDPOINT": os.environ.get("LANGCHAIN_ENDPOINT"),
                    "LANGCHAIN_PROJECT": os.environ.get("LANGCHAIN_PROJECT"),
                }
                
                found_vars = False
                for var, value in env_vars.items():
                    if value:
                        st.warning(f"Found: `{var}`=`{value}`. This may override app settings.")
                        found_vars = True
                
                if not found_vars:
                    st.info("No LangSmith environment variables found.")

                if st.button("Test LangSmith Connection"):
                    try:
                        # Use the detected project name for the test
                        from src.core.langsmith_analyzer import LangSmithAnalyzer
                        project_to_test = env_project if env_project else langsmith_project_name
                        analyzer = LangSmithAnalyzer(langsmith_api_key, project_to_test)
                        analyzer.test_connection()
                    except (ImportError, ConnectionError) as e:
                        st.error(f"Test failed: {e}")

        else:
            st.sidebar.info("Enter LangSmith API key for enhanced analytics")
    
    if not openai_api_key:
        st.warning("Enter your OpenAI API key to begin")
        if LANGSMITH_AVAILABLE:
            st.info("Get your LangSmith API key from: https://smith.langchain.com/")
        return
    
    # Navigation
    st.sidebar.header("Navigation")
    pages = ["Enhanced Dual Chatbot", "Standard Test Cases"]
    if LANGSMITH_AVAILABLE and langsmith_api_key:
        pages.insert(1, "LangSmith Analytics")
    
    page = st.sidebar.selectbox("Select Mode:", pages)
    
    if page == "Enhanced Dual Chatbot":
        create_dual_chatbot_interface_with_langsmith(openai_api_key, langsmith_api_key, langsmith_project_name)
    elif page == "LangSmith Analytics" and LANGSMITH_AVAILABLE and langsmith_api_key:
        show_langsmith_analytics(langsmith_api_key)
    else:
        create_standard_comparison_interface(openai_api_key)


def main():
    """Main function to run the integrated system."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "streamlit":
            create_streamlit_app()
        elif sys.argv[1] == "api":
            import uvicorn
            from src.api.server import app
            uvicorn.run(app, host="0.0.0.0", port=8000)
        else:
            print(f"Usage: {sys.argv[0]} [streamlit|api]")
    else:
        print("Investment Chatbot Comparison System with LangSmith Integration")
        print("Usage:")
        print(f"  streamlit run {sys.argv[0]}")
        print(f"  python {sys.argv[0]} api")
        if not LANGSMITH_AVAILABLE:
            print("\nNote: Install LangSmith for enhanced analytics:")
            print("  pip install langsmith langchain langchain-openai")

if __name__ == "__main__":
    main()
