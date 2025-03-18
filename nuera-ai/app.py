import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from knowledge_base import KnowledgeBase
import torch
import wikipedia
import time
import logging
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)

class ResearchAssistant:
    def __init__(self):
        # Load FLAN-T5 model
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    
    def generate_answer(self, context, question):
        """Generate answer with improved prompt structure"""
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer the question using ONLY the information in the context above. "
            "If the answer isn't in the context, respond with 'I don't have enough information to answer this.'"
        )
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                num_beams=5,
                temperature=0.1,
                do_sample=False
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer.replace("Answer:", "").strip()
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return "An error occurred while generating the answer."

# --------------------- Streamlit App ---------------------
st.title("🧠 Nuera-AI : A Cognitive Agent")
st.markdown("""
**How to use:**
1. Add research material through any input method
2. Use the "View All Entries" button to inspect stored knowledge
3. Ask questions in the chat interface
""")

# Initialize session state
if 'kb' not in st.session_state:
    st.session_state.kb = KnowledgeBase()
if 'assistant' not in st.session_state:
    st.session_state.assistant = ResearchAssistant()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --------------------- Knowledge Base Viewer ---------------------
st.header("📂 Knowledge Base Manager")
if st.button("📚 View All Entries"):
    entries = st.session_state.kb.get_all_knowledge()
    if entries:
        st.subheader("📖 Stored Knowledge")
        for i, (title, content) in enumerate(entries, 1):
            st.markdown(f"**{i}. {title}**")
            st.caption(content[:200] + "...")
    else:
        st.warning("No knowledge entries found.")

# Manual Text Input
with st.expander("📝 Add Research Material"):
    research_text = st.text_area("Paste research content:", height=200)
    if st.button("Add to Knowledge Base"):
        if research_text.strip():
            st.session_state.kb.add_text(research_text)
            st.success("Knowledge base updated!")
        else:
            st.error("Please enter valid content.")

# Wikipedia Integration
st.header("📚 Auto-update from Wikipedia")
topic = st.text_input("Search Wikipedia:")
if st.button("Fetch and Add"):
    if topic.strip():
        try:
            page = wikipedia.page(topic)
            st.session_state.kb.add_text(page.summary, title=page.title)
            st.success(f"Added Wikipedia entry: {page.title}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# File Upload
st.header("📄 Upload Documents")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            pdf = PdfReader(uploaded_file)
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        else:
            text = uploaded_file.read().decode("utf-8")
        
        if text.strip():
            st.session_state.kb.add_text(text, title=uploaded_file.name)
            st.success(f"Added: {uploaded_file.name}")
    except Exception as e:
        st.error(f"File processing error: {e}")

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question about your research"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Retrieve context and generate answer
    with st.spinner("Searching knowledge base..."):
        contexts = st.session_state.kb.search(query, k=5)
        context_str = "\n\n".join([f"**{title}**\n{content}" for title, content in contexts])
    
    with st.spinner("Generating answer..."):
        answer = st.session_state.assistant.generate_answer(context_str, query)
    
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})