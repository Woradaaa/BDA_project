import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain.chains.retrieval import (
    create_retrieval_chain
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MFU Academy AI",
    page_icon="🎓"
)

st.title("🎓 MFU Academy AI Assistant")

st.write(
    "ระบบ AI Chatbot สำหรับตอบคำถามเกี่ยวกับคอร์สออนไลน์ "
    "โดยใช้เทคนิค RAG Pipeline"
)

# ---------------- GROUP INFO ----------------

st.markdown("## Group No: BDA_Project2_10")

st.sidebar.header("👥 สมาชิกกลุ่ม")

st.sidebar.markdown("""
- 6631501148 Kanphong Nasuriwong
- 6631501158 Nitiwat Chatturong
- 6631501168 Worada Suyawa
- 6631501169 Supison Kingjuntrasin
""")

# ---------------- API KEY ----------------

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("กรุณาตั้งค่า GOOGLE_API_KEY ใน Streamlit Secrets")
    st.stop()

# ---------------- LOAD RAG ----------------

@st.cache_resource
def load_rag():

    if not os.path.exists("รายละเอียดคอร์สออนไลน์.txt"):
        st.error("ไม่พบ dataset")
        return None

    # Load dataset
    loader = TextLoader(
        "รายละเอียดคอร์สออนไลน์.txt",
        encoding="utf-8"
    )

    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_documents(documents)

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector database
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    # Prompt
    system_prompt = """
    คุณคือ AI Assistant ของ MFU Academy

    ตอบโดยอ้างอิงจากข้อมูลที่ได้รับเท่านั้น

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

# ---------------- CHAT ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input(
    "สอบถามเกี่ยวกับคอร์สออนไลน์..."
)

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("กำลังค้นหาข้อมูล..."):

            response = rag_chain.invoke({
                "input": user_input
            })

            answer = response["answer"]

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
