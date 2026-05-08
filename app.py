import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 1. การตั้งค่าหน้าเว็บและข้อมูลกลุ่ม ---
st.set_page_config(page_title="MFU Academy AI", page_icon="🎓")
st.title("🎓 MFU Academy AI Assistant")
st.write("ระบบผู้ช่วยตอบคำถามคอร์สเรียนออนไลน์ด้วยเทคนิค RAG Pipeline (Powered by Gemini API)")

st.markdown("### Group No: BDA_Project2_10")
st.sidebar.header("รายชื่อสมาชิกกลุ่ม")
st.sidebar.markdown("""
- 6631501148 Kanphong Nasuriwong
- 6631501158 Nitiwat Chatturong
- 6631501168 Worada Suyawa
- 6631501169 Supison Kingjuntrasin
""")

# --- 2. ดึง API Key จากตั้งค่าความลับของ Streamlit ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("⚠️ ยังไม่ได้ตั้งค่า GOOGLE_API_KEY ใน Streamlit Secrets! ไปที่ Manage app -> Settings -> Secrets")
    st.stop()

# --- 3. ตั้งค่าระบบ RAG ---
@st.cache_resource
def load_rag_model():
    if not os.path.exists("รายละเอียดคอร์สออนไลน์.txt"):
        st.error("⚠️ ไม่พบไฟล์ 'รายละเอียดคอร์สออนไลน์.txt' ใน GitHub")
        return None

    # โหลดและหั่นข้อมูล
    loader = TextLoader("รายละเอียดคอร์สออนไลน์.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 🌟 ใช้ Google Embeddings (เบาและไม่กิน RAM)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # โหลดโมเดล LLM ผ่าน API
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=GOOGLE_API_KEY, 
        temperature=0.3
    )

    # ตั้งค่า Prompt
    system_prompt = (
        "คุณคือผู้ช่วยอัจฉริยะของ MFU Academy ตอบคำถามอย่างสุภาพโดยใช้ข้อมูลที่ให้มาเท่านั้น "
        "หากไม่มีข้อมูลให้ตอบว่า 'ขออภัยครับ ไม่พบข้อมูลในระบบ'\n\nContext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

rag_chain = load_rag_model()

# --- 4. ระบบแชท ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("สอบถามรายละเอียดคอร์สเรียน..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔄 กำลังประมวลผล..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response['answer']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
