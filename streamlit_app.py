import os
from dotenv import load_dotenv
import streamlit as st
from models.embeddings import create_embeddings_from_files, retrieve
from models.llm import chat_with_context
from utils.web_search import serpapi_search

load_dotenv()

st.set_page_config(page_title="NeoStats AI Chatbot", layout="wide")
st.title("NeoStats — RAG Chatbot (MVP)")

# Sidebar: API keys and settings
st.sidebar.header("Settings & Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or "")
serp_key = st.sidebar.text_input("SerpAPI Key", type="password", value=os.getenv("SERPAPI_KEY") or "")
mode = st.sidebar.radio("Response mode", ("Concise", "Detailed"))

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if serp_key:
    os.environ["SERPAPI_KEY"] = serp_key

# File upload / ingestion
st.sidebar.markdown("---")
st.sidebar.header("Upload docs (PDF / TXT)")
uploaded_files = st.sidebar.file_uploader("Upload one or more files", type=["pdf","txt"], accept_multiple_files=True)
if uploaded_files:
    if st.sidebar.button("Create embeddings from uploaded files"):
        with st.spinner("Reading files and creating embeddings..."):
            create_embeddings_from_files(uploaded_files, openai_api_key=openai_key)
            st.success("Embeddings created and saved locally.")

# Main chat UI
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([3,1])
with col1:
    query = st.text_input("Ask anything (about uploaded docs / web):")
    if st.button("Ask") and query:
        with st.spinner("Thinking..."):
            docs = retrieve(query, k=4, openai_api_key=openai_key)
            if docs:
                # Build context and call LLM
                context = "\n\n".join([f"{d.page_content}\nSource: {d.metadata.get('source') or d.metadata.get('filename','unknown')}" for d in docs])
                system = {"role":"system","content":"You are an AI assistant. Use context if available. Always include a short SOURCES section."}
                user_msg = {"role":"user","content":f"Question: {query}\n\nContext:\n{context}\n\nRespond in {mode} mode."}
                ans = chat_with_context([system, user_msg], openai_api_key=openai_key)
                st.session_state.history.append((query, ans, [d.metadata for d in docs]))
            else:
                # Fallback to web search
                results = serpapi_search(query, num_results=5, serpapi_key=serp_key)
                web_text = "\n\n".join([f"{r['title']} - {r['link']}\n{r['snippet']}" for r in results])
                system = {"role":"system","content":"You are an AI assistant. Use the web search results to answer. Always include a short SOURCES section."}
                user_msg = {"role":"user","content":f"Question: {query}\n\nWeb results:\n{web_text}\n\nRespond in {mode} mode."}
                ans = chat_with_context([system, user_msg], openai_api_key=openai_key)
                st.session_state.history.append((query, ans, results))

    # Display history
    for q,a,sources in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("**Sources:**")
        for s in sources:
            st.write(s)
        st.markdown("---")

with col2:
    st.markdown("### Tips")
    st.markdown("- Upload files first and click 'Create embeddings' before asking questions about them.")
    st.markdown("- If you ask something outside uploaded docs, the app will search the web.")
    st.markdown("- Use the sidebar to switch Concise/Detailed mode.")