import streamlit as st
from typing import Optional

from app_api import index_url, query_text


st.set_page_config(page_title="AI PDF Assistant", layout="wide")

st.title("AI PDF Assistant (Streamlit)")

with st.sidebar:
    st.header("Index a PDF URL")
    url = st.text_input("PDF URL", value="https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf")
    print(url)
    if st.button("Index URL"):
        with st.spinner("Indexing URL... this may take a while"):
            res = index_url(url)
        if res.get("status") == "ok":
            st.success(f"Indexed: {res.get('url')}")
        else:
            st.error(f"Indexing failed: {res.get('error')}\n{res.get('traceback','')}")

    st.markdown("---")
    st.header("Quick Tools")
    if st.button("Reload knowledge base (default)"):
        # Call index on the default URL by reusing index_url on the default that's in pdf_assistant
        from pdf_assistant import knowledge_base

        default_urls = getattr(knowledge_base, "urls", None)
        if default_urls:
            with st.spinner("Re-loading knowledge base..."):
                # re-run load on the existing knowledge base
                knowledge_base.load(upsert=True)
            st.success("Knowledge base reloaded")
        else:
            st.info("No default URLs available in knowledge_base to reload.")

st.header("Ask the assistant")
query: Optional[str] = st.text_input("Your question:")
if st.button("Ask"):
    if not query or not query.strip():
        st.error("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            res = query_text(query)
        if res.get("status") == "ok":
            st.markdown("**Answer:**")
            # Show the assistant response as preformatted text to preserve newlines
            st.text(res.get("result"))
        else:
            st.error(f"Query failed: {res.get('error')}\n{res.get('traceback','')}")

st.markdown("---")
st.write("This Streamlit app calls into the project code. Long indexing operations may block the UI; consider running reindexing as a background job for production.")
