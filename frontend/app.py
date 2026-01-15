import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/rag/stream"

st.set_page_config(page_title="MedWise RAG", layout="wide")
st.title("🩺 MedWise Medical Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a medical question...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        metrics_placeholder = st.empty()

        full_response = ""
        metrics = None

        with requests.get(
            BACKEND_URL,
            params={"question": prompt},
            stream=True,
            timeout=300,
        ) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if line.strip() == "<END>":
                    metrics = next(r.iter_lines(decode_unicode=True))
                    break

                full_response += line
                response_placeholder.markdown(full_response)

        if metrics:
            stats = eval(metrics)
            metrics_placeholder.markdown(
                f"""
**⏱ Performance**
- Retrieval time: `{stats['retrieval_time']}s`
- LLM generation time: `{stats['llm_time']}s`
- End-to-end time: `{stats['e2e_time']}s`
"""
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )