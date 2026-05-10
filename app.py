import streamlit as st
import os

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="MFU Academy AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 MFU Academy AI Assistant")

st.markdown("""
AI Chatbot สำหรับตอบคำถามเกี่ยวกับคอร์สเรียนออนไลน์  
โดยใช้เทคนิค RAG Pipeline
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("📌 Group Information")

st.sidebar.markdown("""
### Group No:
BDA_Project2_Group10

### Members:
- 6631501148 Kanphong Nasuriwong
- 6631501158 Nitiwat Chatturong
- 6631501168 Worada Suyawa
- 6631501169 Supison Kingjuntrasin
""")

# =====================================================
# API KEY
# =====================================================

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("❌ กรุณาเพิ่ม GOOGLE_API_KEY ใน Streamlit Secrets")
    st.stop()

# =====================================================
# LOAD DATA
# =====================================================

COURSE_FILE = "course_detail.txt"

GUIDE_FILE = "MFU Academy Q&A.txt"

@st.cache_resource
def build_rag():

    # -----------------------------
    # Load course dataset
    # -----------------------------

    with open(COURSE_FILE, "r", encoding="utf-8") as f:
        course_text = f.read()

    docs = [Document(page_content=course_text)]

    # -----------------------------
    # Load guideline
    # -----------------------------

    with open(GUIDE_FILE, "r", encoding="utf-8") as f:
        guide_text = f.read()

    # -----------------------------
    # Split text
    # -----------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40
    )

    splits = splitter.split_documents(docs)

    # -----------------------------
    # Embeddings
    # -----------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # FAISS
    # -----------------------------

    vectorstore = FAISS.from_documents(
        splits,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )

    # -----------------------------
    # Gemini
    # -----------------------------

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    # -----------------------------
    # Prompt
    # -----------------------------

    template = f"""
คุณคือ AI Assistant ของ MFU Academy

จงตอบด้วยสไตล์สุภาพ เป็นกันเอง
ตอบสั้น กระชับ เข้าใจง่าย

ตัวอย่างแนวทางการตอบ:
{guide_text}

ให้ใช้ข้อมูลจาก Context เท่านั้น

หากไม่มีข้อมูลให้ตอบว่า:
"ขออภัยครับ ไม่พบข้อมูลในระบบ"

Context:
{{context}}

Question:
{{input}}

Answer:
"""

    prompt = PromptTemplate.from_template(template)

    # -----------------------------
    # Format docs
    # -----------------------------

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # -----------------------------
    # RAG Pipeline
    # -----------------------------

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, course_text

rag_chain, course_text = build_rag()

# =====================================================
# SESSION
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs([
    "💬 Q&A Chatbot",
    "📚 Course Information"
])

# =====================================================
# TAB 1
# =====================================================

with tab1:

    st.subheader("💬 Ask About Courses")

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ask something about the courses..."
    )

    if user_input:

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            with st.spinner("🔍 Searching..."):

                try:

                    response = rag_chain.invoke(user_input)

                except Exception as e:

                    response = f"⚠️ Error: {str(e)}"

                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.subheader("📚 All Course Information")

    st.text(course_text[:20000])
