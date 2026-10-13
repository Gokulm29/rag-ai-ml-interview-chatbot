
import streamlit as st

st.set_page_config(
    page_title="RAG Chatbot Test",
    page_icon="🤖",
)

st.title("AI/ML Interview Chatbot")
st.success("Streamlit is working!")

question = st.chat_input("Ask a question")

if question:
    st.write("Your question:", question)
