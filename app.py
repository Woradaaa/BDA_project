import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain.chains.retrieval import (
    create_retrieval_chain
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="MFU Academy AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 MFU Academy AI Assistant")

st.markdown("""
ระบบ AI Chatbot สำหรับตอบคำถามเกี่ยวกับคอร์สออนไลน์  
โดยใช้เทคนิค RAG Pipeline + Gemini AI
""")

# --------------------------------------------------
# GROUP INFO
# --------------------------------------------------

st.sidebar.header("📌 Group Information")

st.sidebar.markdown("""
### Group No:
BDA_Project2_10

### Members:
- 6631501148 Kanphong Nasuriwong
- 6631501158 Nitiwat Chatturong
- 6631501168 Worada Suyawa
- 6631501169 Supison Kingjuntrasin
""")

# --------------------------------------------------
# API KEY
# --------------------------------------------------

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("Please add GOOGLE_API_KEY in Streamlit Secrets")
    st.stop()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

DATA_FILE = "รายละเอียดคอร์สออนไลน์.txt"

# --------------------------------------------------
# CACHE VECTOR DATABASE
# --------------------------------------------------

@st.cache_resource
def load_vectorstore():

    loader = TextLoader(
        DATA_FILE,
        encoding="utf-8"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=20
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore

# --------------------------------------------------
# LOAD RAG CHAIN
# --------------------------------------------------

@st.cache_resource
def load_rag():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    system_prompt = """
    คุณคือ AI Assistant ของ MFU Academy
    
    ตอบโดยใช้ข้อมูลจาก context เท่านั้น
    
    หากไม่มีข้อมูลให้ตอบว่า:
    "ขออภัย ไม่พบข้อมูลในระบบ"

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        qa_chain
    )

    return rag_chain

rag_chain = load_rag()

# --------------------------------------------------
# SESSION MEMORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2 = st.tabs([
    "💬 Q&A Chatbot",
    "📚 Course Information"
])

# ==================================================
# TAB 1 : CHATBOT
# ==================================================

with tab1:

    st.subheader("💬 Ask About Courses")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input(
        "Ask about online courses..."
    )

    if user_input:

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            with st.spinner("Searching..."):

                try:

                    response = rag_chain.invoke({
                        "input": user_input
                    })

                    answer = response["answer"]

                except Exception as e:

                    answer = f"⚠️ Error: {str(e)}"

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

# ==================================================
# TAB 2 : COURSE INFORMATION
# ==================================================

with tab2:

    st.subheader("📚 All Course Information")

    if os.path.exists(DATA_FILE):

        with open(DATA_FILE, "r", encoding="utf-8") as file:

            course_text = file.read()

        st.text(course_text)

    else:

        st.error("Dataset file not found")
